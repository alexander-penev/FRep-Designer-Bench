cmake_minimum_required(VERSION 3.20)
project(frep4_bench CXX)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)  # try_compile uses same toolchain
# FREP4_ROOT: FRep Designer 4.0 source tree (env or -DFREP4_ROOT=...).
if(NOT DEFINED FREP4_ROOT)
  set(FREP4_ROOT $ENV{FREP4_ROOT})
endif()
if(NOT FREP4_ROOT)
  set(FREP4_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../frep4)  # default: sibling clone
endif()
if(NOT EXISTS ${FREP4_ROOT}/CMakeLists.txt)
  message(FATAL_ERROR "FRep Designer not found at ${FREP4_ROOT}; run frep4_bench/run.sh or clone https://github.com/alexander-penev/FRep-Designer.git there")
endif()
# The SDF-eval benchmarks (gen_scenes, bench_grid) only need the CPU JIT, so GPU
# is off by default. The render benchmark (bench_render) needs the GPU/RTX
# executors — enable it with -DFREP4_BENCH_RENDER=ON, which turns on FREP_BUILD_GPU.
option(FREP4_BENCH_RENDER "Build the render-path benchmark (needs Vulkan)" OFF)

set(FREP_BUILD_GUI OFF CACHE BOOL "" FORCE)
if(FREP4_BENCH_RENDER)
  set(FREP_BUILD_GPU ON CACHE BOOL "" FORCE)
else()
  set(FREP_BUILD_GPU OFF CACHE BOOL "" FORCE)
endif()
set(FREP_BUILD_OPENCL OFF CACHE BOOL "" FORCE)
set(FREP_BUILD_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(${FREP4_ROOT} ${CMAKE_BINARY_DIR}/frep4 EXCLUDE_FROM_ALL)
foreach(t gen_scenes bench_grid)
  add_executable(frep4_${t} ${t}.cpp)
  target_include_directories(frep4_${t} PRIVATE ${FREP4_ROOT})
  target_link_libraries(frep4_${t} PRIVATE frep_core)
  set_property(TARGET frep4_${t} PROPERTY CXX_STANDARD 26)
endforeach()

if(FREP4_BENCH_RENDER)
  add_executable(frep4_bench_render bench_render.cpp)
  target_include_directories(frep4_bench_render PRIVATE ${FREP4_ROOT})
  # frep_core plus whatever GPU/RTX libs the tree exposes when FREP_BUILD_GPU=ON.
  target_link_libraries(frep4_bench_render PRIVATE frep_core)
  set_property(TARGET frep4_bench_render PROPERTY CXX_STANDARD 26)
endif()
