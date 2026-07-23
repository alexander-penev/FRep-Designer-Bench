cmake_minimum_required(VERSION 3.28)
project(frep VERSION 4.53.0 LANGUAGES C CXX)

# ═════════════════════════════════════════════════════════════════════════════
# C++26
# ═════════════════════════════════════════════════════════════════════════════
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # for shared plugin libs

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ═════════════════════════════════════════════════════════════════════════════
# LLVM 22+ (minimum; warns if older)
# ═════════════════════════════════════════════════════════════════════════════
option(FREP_STRICT_LLVM_VERSION
       "Require exactly LLVM 22+" OFF)
option(FREP_BUILD_GUI
       "Build the Qt6 GUI application" ON)
option(FREP_BUILD_DYNAMIC_PLUGINS
       "Build dynamic plugins (.so)" ON)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

# ═════════════════════════════════════════════════════════════════════════════
# Vulkan (optional — for the GPU compute path)
# ═════════════════════════════════════════════════════════════════════════════
option(FREP_BUILD_GPU "Build the Vulkan GPU compute path" ON)
find_package(Vulkan QUIET)
if(FREP_BUILD_GPU AND NOT Vulkan_FOUND)
    message(WARNING "Vulkan not found; FREP_BUILD_GPU disabled")
    set(FREP_BUILD_GPU OFF)
endif()
if(FREP_BUILD_GPU)
    message(STATUS "Vulkan ${Vulkan_VERSION} from ${Vulkan_LIBRARY}")
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(NOT GLSLANG_VALIDATOR)
        message(WARNING "glslangValidator not found; FREP_BUILD_GPU disabled")
        set(FREP_BUILD_GPU OFF)
    endif()
endif()

# ═════════════════════════════════════════════════════════════════════════════
# OpenCL (optional — for the GPU-IR retargeting path; runs the IR→SPIR-V
# kernel on the GPU via OpenCL, distinct from the GLSL→Vulkan path).
# ═════════════════════════════════════════════════════════════════════════════
option(FREP_BUILD_OPENCL "Build the OpenCL GPU-IR path" ON)
# Prefer a real find_package; fall back to the loader lib by soname and the
# bundled Khronos headers (third_party/CL) when the dev package is absent.
find_library(OPENCL_LIB NAMES OpenCL libOpenCL.so.1 PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
if(FREP_BUILD_OPENCL AND NOT OPENCL_LIB)
    message(WARNING "libOpenCL not found; FREP_BUILD_OPENCL disabled")
    set(FREP_BUILD_OPENCL OFF)
endif()
if(FREP_BUILD_OPENCL)
    message(STATUS "OpenCL loader: ${OPENCL_LIB}")
endif()

# ═════════════════════════════════════════════════════════════════════════════
# CUDA Driver API (optional — for the GPU-IR path on NVIDIA via PTX). We use
# only the driver API (libcuda.so, shipped with the NVIDIA driver) and
# declare the slice we need in cuda_ctx.cpp, so no CUDA toolkit headers are
# required — only the loader library.
# ═════════════════════════════════════════════════════════════════════════════
option(FREP_BUILD_CUDA "Build the CUDA GPU-IR path (NVPTX→PTX)" ON)
find_library(CUDA_DRIVER_LIB NAMES cuda PATHS
    /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu/stubs /usr/local/cuda/lib64/stubs)
if(FREP_BUILD_CUDA AND NOT CUDA_DRIVER_LIB)
    message(WARNING "libcuda not found; FREP_BUILD_CUDA disabled")
    set(FREP_BUILD_CUDA OFF)
endif()
if(FREP_BUILD_CUDA)
    message(STATUS "CUDA driver: ${CUDA_DRIVER_LIB}")
endif()

# ═════════════════════════════════════════════════════════════════════════════
# libpng (optional — PNG texture loading)
# Without libpng we still build, but Material::Pattern::Texture only
# accepts BMP files.
# ═════════════════════════════════════════════════════════════════════════════
find_package(PNG QUIET)
if(PNG_FOUND)
    message(STATUS "libpng ${PNG_VERSION_STRING} — PNG textures enabled")
else()
    message(STATUS "libpng not found — PNG textures disabled (BMP still works)")
endif()

if(LLVM_VERSION_MAJOR LESS 22)
    if(FREP_STRICT_LLVM_VERSION)
        message(FATAL_ERROR "frep requires LLVM 22+. Found: ${LLVM_PACKAGE_VERSION}")
    else()
        message(WARNING "LLVM ${LLVM_PACKAGE_VERSION} < 22; building in compatibility mode")
    endif()
endif()

llvm_map_components_to_libnames(LLVM_LIBS
    core orcjit native passes support irreader transformutils
)
if(LLVM_LINK_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
endif()

# ═════════════════════════════════════════════════════════════════════════════
# Compile options
# ═════════════════════════════════════════════════════════════════════════════
add_library(frep_opts INTERFACE)
target_compile_options(frep_opts INTERFACE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    $<$<CONFIG:Debug>:-O0 -g>
)
target_compile_definitions(frep_opts INTERFACE ${LLVM_DEFINITIONS_LIST})
target_include_directories(frep_opts INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
)

# ═════════════════════════════════════════════════════════════════════════════
# Core library
# ═════════════════════════════════════════════════════════════════════════════
add_library(frep_core STATIC
    core/compiler/codegen.cpp
    core/frep/custom_expr.cpp
    core/frep/expr_ast.cpp
    core/frep/template_fn.cpp
    core/frep/node.cpp
    core/frep/primitives_ad.cpp
    core/frep/operations_ad.cpp
    core/frep/deformations_ad.cpp
    core/frep/aabb.cpp
    core/frep/scene.cpp
    core/frep/mesh_sdf.cpp
    core/accel/guard_calibration.cpp
    core/exec/multipath.cpp
    core/exec/executor_factory.cpp
    core/power/energy_meter.cpp
    core/io/scene_io.cpp
    core/mesh/marching_cubes.cpp
    core/mesh/marching_cubes_tables.cpp
)
target_link_libraries(frep_core PUBLIC frep_opts ${LLVM_LIBS} ${CMAKE_DL_LIBS})
# -fno-rtti only for core (because of LLVM headers); PRIVATE so it does not
# propagate to the GUI target, which needs RTTI for dynamic_cast on QGraphicsItem.
target_compile_options(frep_core PRIVATE -fno-rtti)

# Threading. TileScheduler uses std::jthread / std::stop_token (C++20).
# Always link the threads library (pthread on Linux).
find_package(Threads REQUIRED)
target_link_libraries(frep_core PUBLIC Threads::Threads)

# ── std::jthread / std::stop_token supplemental library ─────────────────────
#
# On recent libstdc++ (GCC 13+, including the libstdc++ that ships with
# Clang 19/20/22), the std::stop_token atomic-wait machinery — the
# symbol std::__detail::__notify_impl(..., __wait_args_base const&) — was
# moved OUT of the main libstdc++ into a separate library, variously
# named libstdc++exp.a (most common) or libstdc++_libbacktrace.a.
# Without linking it, any *executable* that pulls in tile_scheduler.hpp
# (which uses std::jthread) fails at the final link with:
#   undefined reference to 'std::__detail::__notify_impl(...)'
#
# This is purely a link-line issue; the code is standard and correct.
#
# Detection is done by actually *link-testing* a real std::jthread
# program against each candidate library, using the project's true
# compile flags. We can't rely on -print-file-name alone because some
# toolchains report a path for a library that the driver then refuses
# to auto-link, and we can't rely on a no-library probe because the
# static lib frep_core hides the unresolved symbol until exe-link time.
#
# Whatever candidate links cleanly is attached PUBLIC to frep_core so
# it propagates (in the right order, after frep_core's own objects) to
# every executable: frep_designer, frep_tests, frep_render, etc.
include(CheckCXXSourceCompiles)

set(_frep_jthread_src "
#include <stop_token>
#include <thread>
#include <chrono>
int main() {
    std::jthread t([](std::stop_token st){
        while (!st.stop_requested()) {}
    });
    t.request_stop();
    return 0;
}
")

# Save and restore CMAKE_REQUIRED_* around the probes.
set(_frep_saved_req_libs "${CMAKE_REQUIRED_LIBRARIES}")

# Strategy 0: does it already link with no extra library? (older
# toolchains where the symbol is still in the main libstdc++.)
set(CMAKE_REQUIRED_LIBRARIES "")
check_cxx_source_compiles("${_frep_jthread_src}" FREP_JTHREAD_NO_EXTRA_LIB)

set(_frep_jthread_extra_lib "")
if(NOT FREP_JTHREAD_NO_EXTRA_LIB)
    # Strategy 1..N: try each candidate library name as a raw -l flag.
    # Index suffix keeps the cache-variable names unique per attempt.
    set(_idx 0)
    foreach(_cand stdc++exp stdc++_libbacktrace stdc++fs)
        math(EXPR _idx "${_idx}+1")
        set(CMAKE_REQUIRED_LIBRARIES "-l${_cand}")
        check_cxx_source_compiles("${_frep_jthread_src}"
                                  FREP_JTHREAD_LIB_${_idx})
        if(FREP_JTHREAD_LIB_${_idx})
            set(_frep_jthread_extra_lib "${_cand}")
            break()
        endif()
    endforeach()
endif()

# Restore.
set(CMAKE_REQUIRED_LIBRARIES "${_frep_saved_req_libs}")

if(_frep_jthread_extra_lib)
    message(STATUS
        "frep: linking -l${_frep_jthread_extra_lib} for std::jthread / "
        "std::stop_token support")
    # Attach to frep_core PUBLIC so dependents see it...
    target_link_libraries(frep_core PUBLIC ${_frep_jthread_extra_lib})
    # ...AND append to the global executable link flags so it also
    # appears at the very END of every executable's link line. Static-
    # library symbol resolution is order-sensitive: the library that
    # *provides* __notify_impl must come after the object files that
    # *reference* it. PUBLIC propagation alone can place it too early
    # in some generator/linker combinations (this is exactly what bit
    # the clang-22 build — the symbol stayed unresolved despite the
    # PUBLIC dep). Appending to the exe link flags guarantees a final,
    # correctly-ordered occurrence. A duplicate -l is harmless.
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -l${_frep_jthread_extra_lib}"
        CACHE STRING "" FORCE)
elseif(FREP_JTHREAD_NO_EXTRA_LIB)
    message(STATUS "frep: std::jthread links with the default libstdc++")
else()
    # Neither the bare link nor any candidate library worked. Emit a
    # loud warning so the failure mode is obvious, and fall back to
    # forcing -lstdc++exp (the most likely answer on GCC 13+), which
    # at least produces a clear "cannot find -lstdc++exp" if that's
    # genuinely the problem rather than a silent undefined-reference.
    message(WARNING
        "frep: could not find a library that resolves std::jthread's "
        "std::__detail::__notify_impl. Forcing -lstdc++exp as a last "
        "resort. If the build still fails to link, run:\n"
        "    ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++exp.a\n"
        "    ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++exp.so\n"
        "and report the paths so the candidate list can be extended.")
    target_link_libraries(frep_core PUBLIC stdc++exp)
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -lstdc++exp" CACHE STRING "" FORCE)
endif()

# Optional libpng for PNG texture loading.
if(PNG_FOUND)
    target_link_libraries(frep_core PUBLIC PNG::PNG)
    target_compile_definitions(frep_core PUBLIC FREP_HAS_LIBPNG)
endif()

# ═════════════════════════════════════════════════════════════════════════════
# CLI tools
# ═════════════════════════════════════════════════════════════════════════════
add_executable(frep_render tools/render_ppm.cpp)
target_link_libraries(frep_render PRIVATE frep_core)
target_compile_options(frep_render PRIVATE -fno-rtti)

add_executable(frep_advanced tools/render_advanced.cpp)
target_link_libraries(frep_advanced PRIVATE frep_core)
target_compile_options(frep_advanced PRIVATE -fno-rtti)

# ═════════════════════════════════════════════════════════════════════════════
# GPU compute path: Vulkan host + GLSL compute shader → SPIR-V
# ═════════════════════════════════════════════════════════════════════════════
if(FREP_BUILD_GPU)
    # Add Vulkan host code to the core library.
    target_sources(frep_core PRIVATE
        core/gpu/vulkan_ctx.cpp
        core/gpu/glsl_emitter.cpp
        core/gpu/rtx_caps.cpp
        core/gpu/rtx_ctx.cpp
        core/gpu/rtx_accel.cpp
        core/gpu/rtx_shaders.cpp
        core/gpu/rtx_pipeline.cpp
        core/gpu/rtx_csg_groups.cpp
        core/gpu/rtx_query.cpp
    )
    target_link_libraries(frep_core PUBLIC Vulkan::Vulkan)

    # Compile sphere_trace.comp to SPIR-V at build time.
    set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
    set(SPHERE_TRACE_SPV ${SHADER_OUT_DIR}/sphere_trace.spv)
    add_custom_command(
        OUTPUT ${SPHERE_TRACE_SPV}
        COMMAND ${GLSLANG_VALIDATOR} -V
                ${CMAKE_CURRENT_SOURCE_DIR}/gpu/sphere_trace.comp
                -o ${SPHERE_TRACE_SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/sphere_trace.comp
        COMMENT "Compiling sphere_trace.comp → sphere_trace.spv"
        VERBATIM
    )
    add_custom_target(frep_shaders ALL DEPENDS ${SPHERE_TRACE_SPV})

    # Examples gallery generator — produces seven showcase PPMs that
    # exercise different combinations of features (CSG, deformations,
    # MeshSDF, patterns, textures, CPU vs GPU comparison).
    add_executable(frep_gallery tools/gallery.cpp)
    target_link_libraries(frep_gallery PRIVATE frep_core)

    # GPU tile-cull benchmark: renders with cull off vs on, reports ms/frame.
    add_executable(gpu_cull_bench tools/gpu_cull_bench.cpp)
    target_link_libraries(gpu_cull_bench PRIVATE frep_core)
    set_property(TARGET gpu_cull_bench PROPERTY CXX_STANDARD 26)
    target_compile_options(frep_gallery PRIVATE -fno-rtti)

    # Benchmarks tool — prints a markdown table of timings.
    add_executable(frep_bench tools/benchmarks.cpp)
    target_link_libraries(frep_bench PRIVATE frep_core)
    target_compile_options(frep_bench PRIVATE -fno-rtti)

    # Benchmark scene generator — writes examples/benchmark_*.json through
    # the SceneGraph API (schema-valid by construction). Run as
    # `./frep_gen_benchmark_scenes <output-dir>`.
    add_executable(frep_gen_benchmark_scenes tools/gen_benchmark_scenes.cpp)
    target_link_libraries(frep_gen_benchmark_scenes PRIVATE frep_core)
    target_compile_options(frep_gen_benchmark_scenes PRIVATE -fno-rtti)

    # Artifact dump tool — emits IR (pre/post-opt), GLSL, SPIR-V, scene,
    # and a stats JSON for a scene, for system analysis and write-ups.
    add_executable(frep_dump tools/dump_artifacts.cpp)
    target_link_libraries(frep_dump PRIVATE frep_core)
    target_compile_options(frep_dump PRIVATE -fno-rtti)

    # Multi-path executor driver (Model D) — runs decompose/dispatch/merge
    # across selected paths and writes the merged image + a report.
    add_executable(frep_multipath tools/multipath_driver.cpp)
    target_link_libraries(frep_multipath PRIVATE frep_core)
    target_compile_options(frep_multipath PRIVATE -fno-rtti)

    # Distributed render driver (Direction 2b) — master serves tiles to remote
    # worker processes over TCP; workers render with their chosen path.
    add_executable(frep_dist_render tools/dist_render.cpp)
    target_link_libraries(frep_dist_render PRIVATE frep_core)
    target_compile_options(frep_dist_render PRIVATE -fno-rtti)

    # Cross-path equivalence harness — renders every parity scene on two
    # paths and reports per-scene mean/max |Δ|.
    add_executable(frep_parity_check tools/parity_check.cpp)
    target_link_libraries(frep_parity_check PRIVATE frep_core)
    target_compile_options(frep_parity_check PRIVATE -fno-rtti)

    # Validates the opt-in runtime-parameter fast path on GPU_GLSL: a parameter
    # edit must leave the shader byte-identical (no recompile) yet match the
    # full-recompile image.
    add_executable(frep_validate_runtime_params tools/validate_runtime_params.cpp)
    target_link_libraries(frep_validate_runtime_params PRIVATE frep_core)
    target_compile_options(frep_validate_runtime_params PRIVATE -fno-rtti)

    add_executable(frep_diag_handwritten tools/diag_handwritten.cpp)
    target_link_libraries(frep_diag_handwritten PRIVATE frep_core)
    target_compile_options(frep_diag_handwritten PRIVATE -fno-rtti)

    add_executable(frep_diag_shadow tools/diag_shadow.cpp)
    target_link_libraries(frep_diag_shadow PRIVATE frep_core)
    target_compile_options(frep_diag_shadow PRIVATE -fno-rtti)

    add_executable(frep_diag_horizon tools/diag_horizon.cpp)
    target_link_libraries(frep_diag_horizon PRIVATE frep_core)
    target_compile_options(frep_diag_horizon PRIVATE -fno-rtti)

    # RT capability diagnostic — prints what Vulkan reports about ray-tracing
    # support (device names, RT extensions). Only meaningful with Vulkan.
    if(FREP_BUILD_GPU)
        add_executable(frep_rtx_probe tools/rtx_probe.cpp)
        target_link_libraries(frep_rtx_probe PRIVATE frep_core)
        target_compile_options(frep_rtx_probe PRIVATE -fno-rtti)

        # Phase 3 broad-phase scaling benchmark: RT multi-BLAS vs O(N) compute.
        add_executable(frep_rtx_bench tools/rtx_bench.cpp)
        target_link_libraries(frep_rtx_bench PRIVATE frep_core)
        target_compile_options(frep_rtx_bench PRIVATE -fno-rtti)
    endif()
endif()

# ═════════════════════════════════════════════════════════════════════════════
# OpenCL GPU-IR path — runs the IR→SPIR-V kernel on the GPU via OpenCL.
# Independent of FREP_BUILD_GPU (Vulkan): a machine may have one and not the
# other. The bundled Khronos headers in third_party/CL let this compile even
# without the OpenCL dev package installed.
# ═════════════════════════════════════════════════════════════════════════════
if(FREP_BUILD_OPENCL)
    target_sources(frep_core PRIVATE core/gpu/opencl_ctx.cpp)
    target_include_directories(frep_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(frep_core PUBLIC ${OPENCL_LIB})
    target_compile_definitions(frep_core PUBLIC FREP_HAS_OPENCL)
endif()

# CUDA GPU-IR path (NVPTX → PTX → CUDA Driver API). The NVPTX backend is
# part of LLVM (no extra dep); the runtime links libcuda.so.
if(FREP_BUILD_CUDA)
    target_sources(frep_core PRIVATE core/gpu/cuda_ctx.cpp)
    target_link_libraries(frep_core PUBLIC ${CUDA_DRIVER_LIB})
    target_compile_definitions(frep_core PUBLIC FREP_HAS_CUDA)
endif()

# ═════════════════════════════════════════════════════════════════════════════
# Qt6 GUI (frep_designer)
# ═════════════════════════════════════════════════════════════════════════════
if(FREP_BUILD_GUI)
    find_package(Qt6 6.4 COMPONENTS Core Widgets Concurrent QUIET)
    if(Qt6_FOUND)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)

        # Qt 6.4.2 AutoMoc has a bug where Q_OBJECT classes inheriting
        # from a custom QObject-derived interface (rather than from
        # QWidget directly) sometimes produce empty moc_*.cpp output.
        # Symptom: linker errors about missing vtable/staticMetaObject
        # for IViewport / OffscreenViewportAdapter / VulkanViewport.
        #
        # Workaround: invoke moc manually for those three headers via
        # qt6_wrap_cpp and add the generated sources to the executable.
        # The headers are *also* still listed in the source list below
        # so AutoMoc doesn't trip on them when generating its
        # mocs_compilation.cpp (it'll produce empty files, which the
        # compiler is happy to compile as no-ops).
        qt6_wrap_cpp(MANUAL_MOC_SOURCES
            gui/iviewport.hpp
            gui/offscreen_viewport_adapter.hpp
            gui/executor_viewport.hpp
            gui/checkable_combo_box.hpp
            gui/dist_viewport.hpp
            gui/vulkan_viewport.hpp
            gui/frep_vulkan_window.hpp
            gui/main_window.hpp
        )

        add_executable(frep_designer
            gui/main.cpp
            gui/viewport.hpp
            gui/viewport.cpp
            gui/iviewport.hpp
            gui/iviewport.cpp
            gui/offscreen_viewport_adapter.hpp
            gui/offscreen_viewport_adapter.cpp
            gui/executor_viewport.hpp
            gui/executor_viewport.cpp
            gui/checkable_combo_box.hpp
            gui/checkable_combo_box.cpp
            gui/dist_viewport.hpp
            gui/dist_viewport.cpp
            gui/main_window.hpp
            gui/main_window.cpp
            gui/scene_inspector.hpp
            gui/scene_inspector.cpp
            gui/expression_editor.hpp
            gui/expression_editor.cpp
            gui/material_editor.hpp
            gui/material_editor.cpp
            gui/vulkan_viewport.hpp
            gui/vulkan_viewport.cpp
            gui/frep_vulkan_window.hpp
            gui/node_types.hpp
            gui/node_types.cpp
            gui/node_graph.hpp
            gui/node_graph.cpp
            ${MANUAL_MOC_SOURCES}
        )
        # Qt 6.4's MOC parser is older — fall back to C++23 for the GUI.
        # The engine itself stays on C++26.
        set_target_properties(frep_designer PROPERTIES
            CXX_STANDARD 23
            # Plugins (capsule_plugin.so, etc.) are linked as SHARED
            # libraries with `frep_core` as a private dependency — but
            # frep_core is itself a STATIC archive, so plugins refer to
            # the host executable's copy of FRepNode::codegen_grad,
            # PluginRegistry::instance, etc. ENABLE_EXPORTS adds
            # `-rdynamic` so plugins can dlsym those host symbols at
            # load time. We additionally link frep_core with
            # --whole-archive so the linker does not prune unused base
            # class symbols (e.g. FRepNode::codegen_grad's default
            # impl) that plugins need.
            ENABLE_EXPORTS ON
        )
        target_include_directories(frep_designer PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(frep_designer PRIVATE
            ${LLVM_DEFINITIONS_LIST}
            QT_NO_KEYWORDS  # avoids the `emit` macro clashing with the LLVM API
        )
        target_compile_options(frep_designer PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
            -fno-rtti  # consistent ABI with frep_core; the GUI uses
                       # qgraphicsitem_cast instead of dynamic_cast
            $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g>
            $<$<CONFIG:Debug>:-O0 -g>
        )
        target_link_libraries(frep_designer PRIVATE
            # Use --whole-archive so the linker keeps every symbol in
            # frep_core, including base-class virtual definitions
            # (FRepNode::codegen_grad, etc.) that plugins refer to via
            # dlopen. Without this, ld prunes unreferenced symbols and
            # plugin loading fails with "undefined symbol".
            -Wl,--whole-archive frep_core -Wl,--no-whole-archive
            Qt6::Core Qt6::Widgets Qt6::Concurrent
            Vulkan::Vulkan
        )
        message(STATUS "Qt6 GUI: enabled (frep_designer)")

        # ── Headless test for the node graph logic ────────────────────────────
        # Tests NodeGraphScene::build_tree / load_from_tree / cycle detection
        # without an X server. automoc processes node_graph.hpp.
        add_executable(test_node_graph
            tests/test_node_graph.cpp
            gui/node_graph.hpp
            gui/node_graph.cpp
            gui/node_types.hpp
            gui/node_types.cpp
        )
        set_target_properties(test_node_graph PROPERTIES CXX_STANDARD 23)
        target_include_directories(test_node_graph PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(test_node_graph PRIVATE
            ${LLVM_DEFINITIONS_LIST}
            QT_NO_KEYWORDS
        )
        target_compile_options(test_node_graph PRIVATE
            -Wall -Wextra -Wno-unused-parameter -fno-rtti
            # No -DNDEBUG — assert() must stay active in the test.
            $<$<CONFIG:Release>:-O2>
        )
        target_link_libraries(test_node_graph PRIVATE
            frep_core
            Qt6::Core Qt6::Widgets Qt6::Concurrent
        )
        add_test(NAME node_graph_logic COMMAND test_node_graph)
        set_tests_properties(node_graph_logic PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

        # ── Headless reproduction of the multi-view cpu_ir-dark report ─────────
        # Moc the Q_OBJECT headers explicitly via qt6_wrap_cpp, exactly like
        # frep_designer above. AUTOMOC does not reliably pick these up because the
        # .cpp files include their headers with a `gui/` path prefix, so it emits
        # "no files require moc" and the vtables/signals go undefined at link.
        qt6_wrap_cpp(REPRO_MOC_SOURCES
            gui/iviewport.hpp
            gui/executor_viewport.hpp
        )
        add_executable(repro_cpu_ir_black
            tools/repro_cpu_ir_black.cpp
            gui/executor_viewport.cpp
            gui/iviewport.cpp
            ${REPRO_MOC_SOURCES}
        )
        set_target_properties(repro_cpu_ir_black PROPERTIES
            CXX_STANDARD 23
            AUTOMOC OFF)
        target_include_directories(repro_cpu_ir_black PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(repro_cpu_ir_black PRIVATE
            ${LLVM_DEFINITIONS_LIST}
            QT_NO_KEYWORDS)
        target_compile_options(repro_cpu_ir_black PRIVATE
            -Wno-unused-parameter -fno-rtti
            $<$<CONFIG:Release>:-O2>)
        target_link_libraries(repro_cpu_ir_black PRIVATE
            frep_core
            Qt6::Core Qt6::Widgets Qt6::Concurrent)
    else()
        message(STATUS "Qt6 not found — GUI will be skipped")
    endif()
endif()

# ═════════════════════════════════════════════════════════════════════════════
# Dynamic plugins (shared libs)
# ═════════════════════════════════════════════════════════════════════════════
if(FREP_BUILD_DYNAMIC_PLUGINS)
    add_library(capsule_plugin SHARED plugins/dynamic/capsule_plugin.cpp)
    target_link_libraries(capsule_plugin PRIVATE frep_opts ${LLVM_LIBS})
    target_compile_options(capsule_plugin PRIVATE -fno-rtti)
    set_target_properties(capsule_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
        PREFIX ""  # no "lib" prefix — just "capsule_plugin.so"
    )
endif()

# ═════════════════════════════════════════════════════════════════════════════
# Tests — GoogleTest
# ═════════════════════════════════════════════════════════════════════════════
include(CTest)
find_package(GTest REQUIRED)

add_executable(frep_tests
    tests/test_nodes.cpp
    tests/test_codegen.cpp
    tests/test_phase4.cpp
    tests/test_mesh.cpp
    tests/test_mesh_sdf.cpp
    tests/test_triangle_bvh.cpp
    tests/test_sparse_sdf.cpp
    tests/test_undo.cpp
    tests/test_incremental.cpp
    tests/test_spirv.cpp
    tests/test_deformations_patterns.cpp
    tests/test_custom_expr_eval.cpp
    tests/test_expr_backends.cpp
    tests/test_plugin_runtime.cpp
    tests/test_scene_set_geometry.cpp
    tests/test_mixed_nodes_integration.cpp
    tests/test_gpu_shadow_regression.cpp
    tests/test_scene_io_relative_textures.cpp
    tests/test_codegen_texture_fallback.cpp
    tests/test_reflections_soft_shadows.cpp
    tests/test_clone_node.cpp
    tests/test_transform_gizmo.cpp
    tests/test_scene_bvh.cpp
    tests/test_sdf_invariant.cpp
    tests/test_guard_calibration.cpp
    tests/test_multipath.cpp
    tests/test_nvptx.cpp
    tests/test_energy.cpp
    tests/test_compile_policy.cpp
    tests/test_post_process.cpp
    tests/test_dist_transport.cpp
    tests/test_dist_render.cpp
    tests/test_path_parity.cpp
)
if(FREP_BUILD_GPU)
    target_sources(frep_tests PRIVATE
        tests/test_vulkan.cpp
        tests/test_glsl_emitter.cpp
        tests/test_gpu_patterns_mesh.cpp
        tests/test_textures.cpp
        tests/test_plugin_glsl.cpp
        tests/test_rtx_path.cpp
    )
    # Ensure the shader exists before running Vulkan tests.
    add_dependencies(frep_tests frep_shaders)
endif()
target_link_libraries(frep_tests
    PRIVATE
        # Same --whole-archive trick as frep_designer — keeps all
        # FRepNode base symbols visible to plugins that this process
        # might dlopen at test time (see test_plugin_runtime.cpp).
        -Wl,--whole-archive frep_core -Wl,--no-whole-archive
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)
set_target_properties(frep_tests PROPERTIES ENABLE_EXPORTS ON)
target_compile_options(frep_tests PRIVATE -fno-rtti)

# Tests that dlopen the dynamic capsule_plugin.so need it built first.
if(TARGET capsule_plugin)
    add_dependencies(frep_tests capsule_plugin)
endif()

include(GoogleTest)
gtest_discover_tests(frep_tests)

# ═════════════════════════════════════════════════════════════════════════════
# Doxygen API documentation
# Optional — only enabled if doxygen is on PATH. Builds to docs/api/html/.
# Trigger via: cmake --build build --target docs
# ═════════════════════════════════════════════════════════════════════════════
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND Doxygen::doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen ${DOXYGEN_VERSION} — `docs` target available")
else()
    message(STATUS "Doxygen not found — `docs` target will be unavailable. "
                   "Install with: sudo apt install doxygen graphviz")
endif()

# ═════════════════════════════════════════════════════════════════════════════
# Performance regression check (item 4: perf CI)
# Convenience targets that wire frep_bench + tools/perf_check.py together,
# so local developers run the same regression gate that CI runs without
# memorising the two-step shell invocation.
#
# Usage:
#   cmake --build build --target perf-check             # compare to baseline
#   cmake --build build --target perf-update-baseline   # rewrite baseline
#
# The thresholds match the values used by .github/workflows/ci.yml so a
# green local run implies a green CI run (modulo CI runner noise).
# ═════════════════════════════════════════════════════════════════════════════
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND TARGET frep_bench)
    set(_PERF_REPORT ${CMAKE_BINARY_DIR}/perf_report.json)

    add_custom_target(perf-check
        COMMAND $<TARGET_FILE:frep_bench> --json > ${_PERF_REPORT}
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/perf_check.py
                ${_PERF_REPORT}
                --baseline ${CMAKE_CURRENT_SOURCE_DIR}/tools/perf_baseline.json
                --cpu-threshold 2.0
                --gpu-threshold 3.0
        DEPENDS frep_bench
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running performance regression check"
        VERBATIM
        USES_TERMINAL
    )

    add_custom_target(perf-update-baseline
        COMMAND $<TARGET_FILE:frep_bench> --json
                > ${CMAKE_CURRENT_SOURCE_DIR}/tools/perf_baseline.json
        DEPENDS frep_bench
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Refreshing tools/perf_baseline.json from current build"
        VERBATIM
        USES_TERMINAL
    )

    message(STATUS "Python3 ${Python3_VERSION} — `perf-check` and "
                   "`perf-update-baseline` targets available")
else()
    message(STATUS "Python3 not found — `perf-check` target will be unavailable")
endif()

# CPU sphere-tracer benchmark (scalar vs SIMD packet vs Lipschitz step).
add_executable(bench_trace tools/bench_trace.cpp)
target_include_directories(bench_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_trace PRIVATE frep_core)
set_property(TARGET bench_trace PROPERTY CXX_STANDARD 26)

# Standalone randomized soundness test for the interval twin.
add_executable(test_interval_enclosure tests/test_interval_enclosure.cpp)
target_include_directories(test_interval_enclosure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_interval_enclosure PRIVATE frep_core)
set_property(TARGET test_interval_enclosure PROPERTY CXX_STANDARD 26)

# Interval tile-cull: emits the GLSL compute shader and measures its cull rate.
add_executable(bench_tile_cull tools/bench_tile_cull.cpp)
target_include_directories(bench_tile_cull PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_tile_cull PRIVATE frep_core)
set_property(TARGET bench_tile_cull PROPERTY CXX_STANDARD 26)
