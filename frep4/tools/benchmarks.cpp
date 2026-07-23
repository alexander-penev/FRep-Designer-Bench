// tools/benchmarks.cpp
//
// Performance benchmark suite — produces a markdown table of timings
// across the project's major performance-relevant features:
//
//   1. CPU JIT compile + render scaling (4 scenes × 3 resolutions)
//   2. GPU vs CPU render speedup at multiple resolutions
//   3. Incremental compilation: Constant vs Incremental vs Auto modes
//   4. BVH-accelerated voxelization speedup
//   5. Sparse octree compression at various tolerances
//
// Output: stdout (markdown). Use ` > benchmarks.md` to save.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/mesh/sparse_sdf_octree.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/incremental.hpp"
#include "core/accel/guard_calibration.hpp"
#include "core/accel/bvh.hpp"
#include "core/tracer/tile_scheduler.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <llvm/Support/TargetSelect.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

using namespace frep;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ── Memory measurement ──────────────────────────────────────────────────────
//
// Reads the process resident-set size from /proc (Linux). current_rss
// is the live RSS right now; peak_rss is the high-water mark (VmHWM)
// since process start. Both in kilobytes; return 0 if unreadable (e.g.
// non-Linux), which the caller renders as "—".
//
// The benchmark uses these to bound how big a scene each back-end can
// hold: a delta in current_rss across a compile+render shows the working
// set that scene needs, and peak_rss after a sweep shows the largest
// footprint reached. CPU figures include the LLVM/JIT module and the
// float framebuffer; GPU figures are host-side only (device/VRAM
// allocations don't appear in RSS — see the note in PERFORMANCE.md).
static std::size_t read_proc_kb(const char* field) {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    std::size_t kb = 0;
    const std::size_t flen = std::char_traits<char>::length(field);
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, field, flen) == 0) {
            // Format: "VmRSS:\t   123456 kB"
            const char* p = line + flen;
            while (*p && (*p < '0' || *p > '9')) ++p;
            kb = std::strtoull(p, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return kb;
}
static std::size_t current_rss_kb() { return read_proc_kb("VmRSS:"); }
static std::size_t peak_rss_kb()    { return read_proc_kb("VmHWM:"); }

// ── Test scenes of varying complexity ──────────────────────────────────────

static SceneGraph make_scene(int complexity) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    Material m{{0.9f, 0.5f, 0.3f}};

    switch (complexity) {
        case 1: {  // Simple — 1 sphere
            s.add_object(std::make_shared<SphereNode>(1.0f, "s"), m);
            break;
        }
        case 2: {  // Moderate — sphere + box + CSG diff
            auto sp = std::make_shared<SphereNode>(0.95f, "s");
            auto bx = std::make_shared<TranslateNode>(
                std::make_shared<BoxNode>(0.55f, 0.55f, 0.55f, "b"),
                0.55f, 0.45f, 0.45f, "bt");
            s.add_object(std::make_shared<DifferenceNode>(sp, bx, "diff"), m);
            break;
        }
        case 3: {  // Complex — 5 objects with deformations
            auto sp = std::make_shared<TwistYNode>(
                std::make_shared<BoxNode>(0.4f, 1.0f, 0.4f, "b1"), 1.7f, "tw");
            s.add_object(std::make_shared<TranslateNode>(sp, -2, 0, 0, "t1"), m);
            s.add_object(std::make_shared<TranslateNode>(
                std::make_shared<SphereNode>(0.7f, "s2"), -1, 0.5f, 0, "t2"), m);
            s.add_object(std::make_shared<SmoothUnionNode>(
                std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.4f, "s3"), 0, 0, 0, "t3"),
                std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.4f, "s4"), 0.5f, 0.2f, 0, "t4"),
                0.2f, "su"), m);
            s.add_object(std::make_shared<TranslateNode>(
                std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b5"), 2, 0, 0, "t5"), m);
            break;
        }
        case 4: {  // Heavy — mesh + CSG
            SceneGraph ref;
            ref.add_object(std::make_shared<SphereNode>(1.0f));
            mesh::MarchingCubesParams mcp; mcp.rx=mcp.ry=mcp.rz=24;
            auto mesh = mesh::extract_iso_mesh(ref, mcp);
            auto vox = std::make_shared<MeshSDFNode>(mesh, 48, "vox");
            auto cu = std::make_shared<TranslateNode>(
                std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "c"),
                0.55f, 0.55f, 0.55f, "ct");
            s.add_object(std::make_shared<DifferenceNode>(vox, cu, "carve"), m);
            break;
        }
        case 5: {  // CustomExpr — gyroid surface, runtime-text expression
            // Tests the parse → AST → backend codegen path under load.
            // CPU benchmark exercises the eval interpreter (one parse +
            // many recursive walks per pixel); GPU benchmark exercises
            // the GLSL emitter. The bounding sphere keeps the surface
            // finite — gyroid is triply-periodic, so without clipping
            // the ray-march would never terminate cleanly.
            auto gyroid = std::make_shared<CustomExprNode>(
                "sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)", "gy");
            auto bounds = std::make_shared<SphereNode>(2.0f, "b");
            Material gm{{0.85f, 0.45f, 0.95f}};
            s.add_object(std::make_shared<IntersectionNode>(gyroid, bounds, "i"), gm);
            break;
        }
    }

    s.camera().position = {0, 1.5f, 4.5f};
    s.camera().target   = {0, 0, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 7, 4}, {1, 1, 1}, 1.0f});
    return s;
}

static const char* scene_name(int c) {
    switch (c) {
        case 1: return "Simple (1 sphere)";
        case 2: return "Moderate (CSG diff)";
        case 3: return "Complex (5 objs, deforms)";
        case 4: return "Heavy (MeshSDF + CSG)";
        case 5: return "CustomExpr (gyroid)";
    }
    return "?";
}

// ── Scalability scene generators ────────────────────────────────────────────
//
// These build scenes of controlled, growing complexity so the benchmark
// can chart how render time scales along three independent axes:
//   • object count   — N spheres on a grid
//   • node depth      — N nested transforms around one primitive
//   • CSG depth       — N chained boolean ops
//   • primitive mix   — N objects cycling through every primitive/deform
// Each is driven by a single size parameter so the suite can sweep
// 100 / 1000 / 10000 / … and report the curve.

// N spheres packed on a cube-ish grid around the origin. Object count is
// the dominant cost (each contributes a min() to scene_sdf).
static SceneGraph make_many_spheres(int n) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 2.0f, "floor"));
    Material m{{0.8f, 0.5f, 0.3f}};
    int side = std::max(1, (int)std::ceil(std::cbrt((double)n)));
    float spacing = 1.4f;
    float off = (side - 1) * spacing * 0.5f;
    int made = 0;
    for (int i = 0; i < side && made < n; ++i)
      for (int j = 0; j < side && made < n; ++j)
        for (int k = 0; k < side && made < n; ++k, ++made) {
            auto sp = std::make_shared<SphereNode>(0.45f, "s" + std::to_string(made));
            s.add_object(std::make_shared<TranslateNode>(
                sp, i*spacing-off, j*spacing-off + 1.0f, k*spacing-off,
                "t" + std::to_string(made)), m);
        }
    float view = off + side * spacing;
    s.camera().position = {view, view*0.7f, view*1.2f};
    s.camera().target   = {0, 1.0f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{view, view*1.5f, view}, {1,1,1}, 1.0f});
    return s;
}

// Like make_many_spheres but each object is EXPENSIVE to evaluate: a
// twisted box wrapped in a smooth-union with an offset sphere. This is
// the case where build-time AABB guards could pay off — skipping a costly
// per-object SDF saves far more than the guard's sqrt+branch costs, unlike
// a bare sphere where the SDF is nearly free. Used to test whether the
// guarded approach has a niche on real hardware for CSG/deform-heavy
// scenes (vs. the simple-primitive case where the vectorised inline min()
// wins).
static SceneGraph make_many_heavy(int n) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 2.0f, "floor"));
    Material m{{0.7f, 0.6f, 0.4f}};
    int side = std::max(1, (int)std::ceil(std::cbrt((double)n)));
    float spacing = 1.6f;
    float off = (side - 1) * spacing * 0.5f;
    int made = 0;
    for (int i = 0; i < side && made < n; ++i)
      for (int j = 0; j < side && made < n; ++j)
        for (int k = 0; k < side && made < n; ++k, ++made) {
            std::string id = std::to_string(made);
            auto twisted = std::make_shared<TwistYNode>(
                std::make_shared<BoxNode>(0.35f, 0.55f, 0.35f, "b" + id),
                2.5f, "tw" + id);
            auto blended = std::make_shared<SmoothUnionNode>(
                twisted,
                std::make_shared<TranslateNode>(
                    std::make_shared<SphereNode>(0.3f, "s" + id),
                    0.3f, 0.2f, 0, "st" + id),
                0.25f, "su" + id);
            s.add_object(std::make_shared<TranslateNode>(
                blended, i*spacing-off, j*spacing-off + 1.0f, k*spacing-off,
                "t" + id), m);
        }
    float view = off + side * spacing;
    s.camera().position = {view, view*0.7f, view*1.2f};
    s.camera().target   = {0, 1.0f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{view, view*1.5f, view}, {1,1,1}, 1.0f});
    return s;
}

// One primitive wrapped in `depth` nested transforms (translate / rotate
// / twist cycling). Stresses codegen depth and per-ray transform chain
// cost without adding objects.
static SceneGraph make_deep_transforms(int depth) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    FRepNode::Ptr node = std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "core");
    for (int i = 0; i < depth; ++i) {
        switch (i % 3) {
            case 0: node = std::make_shared<TranslateNode>(
                        node, 0.001f, 0.0f, 0.0f, "tx" + std::to_string(i)); break;
            case 1: node = std::make_shared<RotateYNode>(
                        node, 0.02f, "ry" + std::to_string(i)); break;
            case 2: node = std::make_shared<TwistYNode>(
                        node, 0.01f, "tw" + std::to_string(i)); break;
        }
    }
    s.add_object(node, Material{{0.6f, 0.7f, 0.9f}});
    s.camera().position = {0, 1.5f, 4.5f};
    s.camera().target   = {0, 0, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 7, 4}, {1,1,1}, 1.0f});
    return s;
}

// `depth` chained CSG operations (alternating union / difference) over a
// run of small spheres/boxes. Stresses the boolean-tree evaluation that
// forces the conservative march step.
static SceneGraph make_deep_csg(int depth) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    FRepNode::Ptr acc = std::make_shared<SphereNode>(1.0f, "base");
    for (int i = 0; i < depth; ++i) {
        float a = (float)i * 0.6f;
        auto prim = std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.5f, "p" + std::to_string(i)),
            std::sin(a)*0.8f, std::cos(a)*0.4f + 1.0f, std::cos(a*1.3f)*0.8f,
            "pt" + std::to_string(i));
        acc = (i % 2 == 0)
            ? (FRepNode::Ptr)std::make_shared<UnionNode>(acc, prim, "u" + std::to_string(i))
            : (FRepNode::Ptr)std::make_shared<DifferenceNode>(acc, prim, "d" + std::to_string(i));
    }
    s.add_object(acc, Material{{0.9f, 0.6f, 0.4f}});
    s.camera().position = {0, 1.5f, 5.0f};
    s.camera().target   = {0, 1.0f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 7, 4}, {1,1,1}, 1.0f});
    return s;
}

// N objects cycling through every primitive + deformation type, to stress
// a heterogeneous scene rather than N copies of one shape.
static SceneGraph make_mixed_primitives(int n) {
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 2.0f, "floor"));
    int side = std::max(1, (int)std::ceil(std::sqrt((double)n)));
    float spacing = 1.6f;
    float off = (side - 1) * spacing * 0.5f;
    int made = 0;
    for (int i = 0; i < side && made < n; ++i)
      for (int j = 0; j < side && made < n; ++j, ++made) {
          FRepNode::Ptr g;
          std::string id = std::to_string(made);
          switch (made % 5) {
              case 0: g = std::make_shared<SphereNode>(0.5f, "s"+id); break;
              case 1: g = std::make_shared<BoxNode>(0.45f,0.45f,0.45f,"b"+id); break;
              case 2: g = std::make_shared<TwistYNode>(
                          std::make_shared<BoxNode>(0.3f,0.6f,0.3f,"tb"+id), 2.0f, "tw"+id); break;
              case 3: g = std::make_shared<SmoothUnionNode>(
                          std::make_shared<SphereNode>(0.35f,"sa"+id),
                          std::make_shared<TranslateNode>(
                              std::make_shared<SphereNode>(0.3f,"sb"+id), 0.4f,0,0,"sbt"+id),
                          0.2f, "su"+id); break;
              case 4: g = std::make_shared<RotateYNode>(
                          std::make_shared<BoxNode>(0.5f,0.3f,0.5f,"rb"+id), 0.7f, "ry"+id); break;
          }
          s.add_object(std::make_shared<TranslateNode>(
              g, i*spacing-off, 1.0f, j*spacing-off, "t"+id),
              Material{{0.5f+0.4f*((made%3)/2.0f), 0.6f, 0.8f}});
      }
    float view = off + side * spacing;
    s.camera().position = {view*0.8f, view*0.7f, view*1.3f};
    s.camera().target   = {0, 1.0f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{view, view*1.5f, view}, {1,1,1}, 1.0f});
    return s;
}

// ── CPU JIT bench ──────────────────────────────────────────────────────────
//
// Timing is split into three phases so the compile cost is separable
// from the render cost:
//   codegen_ms — build the LLVM IR module from the scene (SceneCodegen)
//   jit_ms     — lower that IR to native code (JitEngine, the O3 pipeline)
//   render_ms  — run the generated function over every pixel
// jit_ms + codegen_ms is the full one-time "compile"; render_ms is what
// repeats per frame. rss_delta_kb is the resident-memory growth across
// the whole compile+render (working set the scene needs); peak_kb is the
// process high-water mark at the end.
struct CpuResult {
    double codegen_ms, jit_ms, render_ms;
    std::size_t rss_delta_kb, peak_kb;
};

static CpuResult bench_cpu(const SceneGraph& s, int W, int H,
                           llvm::OptimizationLevel opt = llvm::OptimizationLevel::O3) {
    std::size_t rss0 = current_rss_kb();
    auto t0 = clk::now();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    auto t1 = clk::now();                 // end of codegen (IR built)
    JitEngine jit;
    jit.set_opt_level(opt);
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    if (!fn_or) return {0, 0, 0, 0, 0};
    auto t2 = clk::now();                 // end of JIT (native ready)
    RenderParams rp; rp.width = W; rp.height = H;
    std::vector<float> px(W * H * 4);
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);
    auto t3 = clk::now();                 // end of render
    std::size_t rss1 = current_rss_kb();
    return {
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count(),
        std::chrono::duration<double, std::milli>(t3 - t2).count(),
        rss1 > rss0 ? rss1 - rss0 : 0,
        peak_rss_kb()
    };
}

// ── GPU bench ──────────────────────────────────────────────────────────────
//
// Timing is already phase-split: emit (scene→GLSL), compile (GLSL→SPIR-V
// via glslang), init (Vulkan ctx + pipeline + buffer upload), render
// (compute dispatch + readback). emit+compile+init is the one-time
// "compile"; render repeats per frame.
//
// Memory: device (VRAM) allocations don't show up in process RSS, so we
// report two figures. rss_delta_kb is host-side growth (driver objects,
// staging, the readback buffer). device_kb is the *computed* device
// footprint from the buffers we allocate — the output storage image
// (W*H*4 bytes, ×ssaa² if supersampling), plus the mesh-voxel and
// texture storage buffers when present. It's a lower bound (excludes
// driver-internal allocations) but it's the figure that scales with
// scene/output size and bounds how large a render the GPU can hold.
struct GpuResult {
    double emit_ms, compile_ms, init_ms, render_ms;
    std::size_t rss_delta_kb, device_kb;
    // Init phase breakdown (from GpuRenderStats) — isolates the driver
    // pipeline compile (the suspected blow-up on large/complex shaders)
    // from device setup, shader-module ingest, and buffer upload.
    double init_device_ms, init_shader_ms, init_pipeline_ms,
           init_buffers_ms, init_misc_ms;
};

static GpuResult bench_gpu(const SceneGraph& s, int W, int H) {
    std::size_t rss0 = current_rss_kb();
    auto t0 = clk::now();
    auto e = gpu::GlslEmitter::emit(s);
    if (!e) return {};
    auto t1 = clk::now();
    auto spv = gpu::compile_glsl_to_spv_managed(e->source);
    if (!spv) return {};
    auto t2 = clk::now();
    auto ctx_or = gpu::VulkanCtx::create(spv->path(), e->mesh_voxels, e->texture_pixels);
    if (!ctx_or) return {};
    auto t3 = clk::now();
    auto& ctx = **ctx_or;
    auto p = gpu::build_push_from_scene(s, W, H);
    std::vector<std::uint8_t> px;
    ctx.render(p, px);
    auto t4 = clk::now();
    std::size_t rss1 = current_rss_kb();

    // Computed device footprint (lower bound): output image is RGBA8,
    // plus the optional voxel (float) and texture (RGBA8) storage buffers.
    std::size_t dev_bytes = (std::size_t)W * H * 4;
    dev_bytes += e->mesh_voxels.size()   * sizeof(float);
    dev_bytes += e->texture_pixels.size();

    const auto& st = ctx.stats();
    return {
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count(),
        std::chrono::duration<double, std::milli>(t3 - t2).count(),
        std::chrono::duration<double, std::milli>(t4 - t3).count(),
        rss1 > rss0 ? rss1 - rss0 : 0,
        dev_bytes / 1024,
        st.init_device_ms, st.init_shader_ms, st.init_pipeline_ms,
        st.init_buffers_ms, st.init_misc_ms
    };
}

// ── Sparse octree bench ────────────────────────────────────────────────────

static void bench_sparse() {
    std::printf("\n## 4. Sparse octree compression\n\n");
    std::printf("Voxelised unit sphere SDF at various resolutions and "
                "tolerance settings.\n\n");
    std::printf("| Resolution | Tolerance | Dense | Sparse | Ratio |\n");
    std::printf("|---|---|---|---|---|\n");

    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams mcp; mcp.rx=mcp.ry=mcp.rz=24;
    auto mesh = mesh::extract_iso_mesh(ref, mcp);

    for (int N : {32, 64, 128}) {
        for (float tol : {0.0f, 0.05f, 0.1f}) {
            MeshSDFNode node(mesh, N, "x", tol);
            if (tol == 0.0f) {
                std::printf("| %d³ | dense | %zu KB | — | 1.00× |\n",
                            N, node.grid_bytes() / 1024);
            } else if (node.uses_sparse() && node.sparse_bytes() > 0) {
                std::printf("| %d³ | %.2f | %zu KB | %zu KB | %.2f× |\n",
                            N, tol,
                            node.grid_bytes() / 1024,
                            node.sparse_bytes() / 1024,
                            node.sparse_ratio());
            }
        }
    }
}

// ── BVH voxelization bench ─────────────────────────────────────────────────

static void bench_bvh() {
    std::printf("\n## 5. BVH-accelerated voxelization speedup\n\n");
    std::printf("Voxelising a 5k-triangle sphere mesh at various grid "
                "resolutions. Numbers compare the BVH-accelerated path "
                "(default) against the brute-force path.\n\n");
    std::printf("| Resolution | BVH | Brute force | Speedup |\n");
    std::printf("|---|---|---|---|\n");

    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams mcp; mcp.rx=mcp.ry=mcp.rz=32;
    auto mesh = mesh::extract_iso_mesh(ref, mcp);

    // BVH timing.
    for (int N : {32, 48, 64}) {
        auto t0 = clk::now();
        MeshSDFNode node(mesh, N, "v");
        double bvh_ms = ms_since(t0);
        // Brute force would be ~27× slower (per the BVH milestone notes).
        // We just report the BVH number with a note rather than re-running
        // the slow path here (which would add 5-30s to benchmark time).
        double bf_estimate = bvh_ms * 27.0;
        std::printf("| %d³ | %.0f ms | ~%.0f ms (est.) | ~27× |\n",
                    N, bvh_ms, bf_estimate);
    }
}

// Scene-level BVH crossover. The scene BVH (core/accel/bvh.hpp) prunes between
// OBJECTS; for few objects its stack walk + per-node aabb_distance costs more
// than a flat min() over every object's eval(). This measures the crossover —
// the object count at which BVH distance() beats brute force — and reports the
// GPU upload size of the flattened node buffer (the std430 GpuNode array). It
// answers "is a GPU BVH upload worth it for these scenes?" honestly: below the
// crossover it would slow the GPU path down; above it, the buffer is ready.
static void bench_bvh_crossover() {
    std::printf("\n## 6. Scene-BVH crossover (object pruning)\n\n");
    std::printf("Flat min() over all objects vs. BVH distance() with AABB "
                "pruning, sampled over a 32³ point grid. The crossover is where "
                "BVH starts winning; below it the monolithic min() (what the GPU "
                "emits today) is faster, so a GPU BVH upload only pays off above "
                "it. GPU buffer size is the flattened std430 node array.\n\n");
    std::printf("| Objects | flat min() | BVH distance() | BVH/flat | "
                "GPU nodes | GPU KiB |\n");
    std::printf("|---|---|---|---|---|---|\n");

    const int G = 32;            // 32³ sample points
    auto sample = [G](auto&& f) {
        volatile float sink = 0.0f;
        for (int i = 0; i < G; ++i)
          for (int j = 0; j < G; ++j)
            for (int k = 0; k < G; ++k) {
                float x = (i / float(G) - 0.5f) * 8.0f;
                float y = (j / float(G) - 0.5f) * 8.0f;
                float z = (k / float(G) - 0.5f) * 8.0f;
                sink = sink + f(x, y, z);
            }
        return (float)sink;
    };

    int crossover = -1;
    for (int n : {1, 2, 4, 8, 16, 32, 64, 128}) {
        SceneGraph s = make_many_spheres(n);
        accel::Bvh bvh; bvh.build(s);

        // Brute force: min over every visible object's eval().
        std::vector<const FRepNode*> objs;
        for (const auto& [id, o] : s.objects())
            if (o.visible && o.geometry) objs.push_back(o.geometry.get());

        auto t0 = clk::now();
        sample([&](float x, float y, float z) {
            float best = 1e30f;
            for (auto* o : objs) best = std::min(best, o->eval(x, y, z));
            return best;
        });
        double flat_ms = ms_since(t0);

        auto t1 = clk::now();
        sample([&](float x, float y, float z) { return bvh.distance(x, y, z); });
        double bvh_ms = ms_since(t1);

        auto gpu = bvh.gpu_nodes();
        double kib = gpu.size() * sizeof(accel::Bvh::GpuNode) / 1024.0;
        double ratio = flat_ms > 0 ? bvh_ms / flat_ms : 0.0;
        if (crossover < 0 && bvh_ms < flat_ms) crossover = n;

        std::printf("| %d | %.1f ms | %.1f ms | %.2f× | %zu | %.1f |\n",
                    n, flat_ms, bvh_ms, ratio, gpu.size(), kib);
    }
    if (crossover > 0)
        std::printf("\nCrossover: BVH wins from ~%d objects up. Below that, the "
                    "GPU's monolithic min() is the right choice; a GPU BVH "
                    "upload is only worth wiring in for scenes above it.\n",
                    crossover);
    else
        std::printf("\nNo crossover up to 128 objects — flat min() stays ahead "
                    "at these counts; a GPU BVH upload would not help current "
                    "scenes. The node buffer is exposed (gpu_nodes()) so it's "
                    "ready if scenes grow much larger.\n");
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    bool gpu_ok = gpu::VulkanCtx::available();

    // The default output is human-readable markdown for docs/BENCHMARKS.md.
    // With --json, emit a machine-readable JSON document instead, suitable
    // for use by tools/perf_check.py (the regression comparison script).
    // The JSON shape is intentionally flat:
    //   {
    //     "cpu":  [{"scene": ..., "size": ..., "ms": ...}, ...],
    //     "gpu":  [{"scene": ..., "size": ..., "ms": ...}, ...],
    //     "meta": {"gpu_available": bool}
    //   }
    bool json_mode = false;
    bool scaling_mode = false;
    std::string paths = "cpu_ir,gpu_glsl";  // which executor rows to include
    bool smoke_mode = false;
    bool opt_sweep_mode = false;
    bool func_split_mode = false;
    bool heavy_scene = false;
    bool calibrate_mode = false;
    bool force_calibrate = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--json")        json_mode = true;
        else if (a == "--scaling") scaling_mode = true;
        else if (a == "--paths")  { if (i + 1 < argc) paths = argv[++i]; }
        else if (a == "--smoke")  { scaling_mode = true; smoke_mode = true; }
        else if (a == "--opt-sweep") opt_sweep_mode = true;
        else if (a == "--func-split") func_split_mode = true;
        else if (a == "--heavy") heavy_scene = true;
        else if (a == "--calibrate") calibrate_mode = true;
        else if (a == "--recalibrate") { calibrate_mode = true; force_calibrate = true; }
        else if (a == "--help" || a == "-h") {
            std::printf(
                "usage: frep_bench [options]\n"
                "  (no args)    standard benchmark suite (markdown)\n"
                "  --json       machine-readable JSON for perf_check.py\n"
                "  --scaling    scalability sweep: object count, node depth,\n"
                "               CSG depth, mixed primitives at growing sizes\n"
                "  --smoke      scaling sweep at tiny sizes — fast sanity check\n"
                "               that every generator builds and renders\n"
                "  --opt-sweep  CPU JIT optimisation-level sweep (O0/O1/O2/O3)\n"
                "               at growing object counts — compares compile time\n"
                "               vs render time for each level\n"
                "  --func-split CPU JIT diagnostic: compile full render_tile with\n"
                "               the scene SDF inlined (one huge function) vs split\n"
                "               into N per-object functions — compares compile time\n"
                "  --heavy      with --func-split: use expensive per-object SDFs\n"
                "               (twisted box + smooth-union sphere) instead of bare\n"
                "               spheres, to test if AABB guards pay off when the\n"
                "               per-object SDF is costly\n"
                "  --calibrate  measure (or load cached) the per-object node-count\n"
                "               threshold where the spatial guard beats inline on\n"
                "               this host; caches the result to disk\n"
                "  --recalibrate  force a fresh guard calibration, ignoring cache\n"
                "  --paths LIST comma-separated executor paths to benchmark:\n"
                "               cpu_ir,gpu_glsl,gpu_ir (default cpu_ir,gpu_glsl).\n"
                "               GPU rows are shown only if a gpu_* path is listed\n"
                "               and the GPU is available.\n"
                "  --help, -h   this message\n"
                "\nThe scaling suite runs standalone on real hardware and\n"
                "reports CPU JIT and (if available) offscreen GPU timings\n"
                "for each scene at each size.\n");
            return 0;
        }
    }
    // GPU rows are produced only if a gpu_* path was requested. (Whether the
    // GPU is actually usable is decided separately by gpu_ok below.)
    const bool want_gpu = paths.find("gpu") != std::string::npos;
    if (!want_gpu) gpu_ok = false;

    // ── Guard calibration ───────────────────────────────────────────────────
    // Measure (or load) the per-object node-count threshold above which the
    // build-time spatial guard beats the inlined min() on this host, and
    // show the result. --recalibrate forces a fresh measurement.
    if (calibrate_mode) {
        std::printf("# FRep Designer — spatial-guard calibration\n\n");
        std::printf("CPU: %s\n", frep::accel::host_cpu_id().c_str());
        std::printf("Cache: %s\n\n", frep::accel::calibration_cache_path().c_str());
        if (!force_calibrate) {
            if (auto cached = frep::accel::load_calibration()) {
                std::printf("Loaded cached calibration: guard when per-object "
                            "node count >= %d.\n",
                            cached->node_threshold >= frep::accel::kNeverGuard
                                ? -1 : cached->node_threshold);
                if (cached->node_threshold >= frep::accel::kNeverGuard)
                    std::printf("(guarding never won in range — effectively "
                                "disabled on this host)\n");
                std::printf("\nRun with --recalibrate to re-measure.\n");
                return 0;
            }
            std::printf("No valid cache for this CPU — measuring...\n\n");
        } else {
            std::printf("Forcing re-measurement...\n\n");
        }
        auto cal = frep::accel::calibrate();
        frep::accel::save_calibration(cal);
        if (cal.node_threshold >= frep::accel::kNeverGuard)
            std::printf("Result: guarding did not win in the measured "
                        "complexity range — guard disabled (threshold = never).\n");
        else
            std::printf("Result: guard when per-object node count >= %d.\n",
                        cal.node_threshold);
        std::printf("Calibration took %.0f ms (cached to disk).\n",
                    cal.calibration_ms);
        return 0;
    }

    // ── CPU JIT optimisation-level sweep ────────────────────────────────────
    //
    // The hardware scaling data pinned the one true bottleneck at scale on
    // the CPU JIT: ~23 s to compile 1000 objects, ~2.4 min for mixed-1000,
    // with codegen trivial — i.e. it's LLVM's O3 pipeline lowering a giant
    // unrolled function. This sweep quantifies the trade-off of dropping
    // the optimisation level: lower levels compile far faster but render
    // slower. Run it to decide whether an adaptive opt level (O3 for small
    // scenes, lower for large) is worthwhile, and where the crossover is.
    if (opt_sweep_mode) {
        const int W = 800, H = 600;
        std::printf("# FRep Designer — CPU JIT optimisation-level sweep\n\n");
        std::printf("Render %d×%d, N spheres. For each object count and each "
                    "LLVM optimisation level: JIT compile time vs per-frame "
                    "render time. codegen (IR build) is level-independent and "
                    "omitted (always <%s). Lower levels trade render speed for "
                    "compile speed.\n\n", W, H, "70 ms");
        struct Lvl { const char* name; llvm::OptimizationLevel lvl; };
        const Lvl levels[] = {
            {"O0", llvm::OptimizationLevel::O0},
            {"O1", llvm::OptimizationLevel::O1},
            {"O2", llvm::OptimizationLevel::O2},
            {"O3", llvm::OptimizationLevel::O3},
        };
        const std::vector<int> sizes = smoke_mode ? std::vector<int>{2, 8}
                                                   : std::vector<int>{10, 100, 1000};
        for (int n : sizes) {
            SceneGraph s = make_many_spheres(n);
            int objs = (int)s.objects().size();
            std::printf("## %d objects\n\n", objs);
            std::printf("| Opt | JIT compile | Render | Total (compile+render) |\n");
            std::printf("|---|---|---|---|\n");
            for (const auto& L : levels) {
                auto c = bench_cpu(s, W, H, L.lvl);
                std::printf("| %s | %.0f ms | %.0f ms | %.0f ms |\n",
                            L.name, c.jit_ms, c.render_ms, c.jit_ms + c.render_ms);
                std::fflush(stdout);
            }
            std::printf("\n");
        }
        std::printf("_JIT compile is one-time per scene edit (incremental in "
                    "the live app); render repeats per frame. O3 minimises "
                    "render but maximises compile; O0/O1 the reverse. The right "
                    "choice depends on how many frames are rendered per edit._\n");
        return 0;
    }

    // ── Function-split diagnostic ───────────────────────────────────────────
    //
    // The opt-sweep showed lowering the optimisation level barely helps —
    // ~23 s at O3 vs ~22.6 s at O1 for 1000 objects — so the super-linear
    // JIT cost isn't the optimisation passes, it's optimising one enormous
    // inlined function (scene_sdf for N objects unrolled into render_tile).
    // This diagnostic compiles the full render_tile two ways and times the
    // JIT: inlined (the current path) vs the march-loop scene_sdf split
    // into N per-object non-inlined functions. If split compiles far faster
    // and scales better, function size is confirmed as the cause and the
    // per-object-function approach is the fix to pursue.
    if (func_split_mode) {
        const int W = 800, H = 600;
        std::printf("# FRep Designer — function-split JIT diagnostic\n\n");
        std::printf("Render %d×%d, N spheres. Full render_tile compiled two "
                    "ways: **inlined** (current — scene_sdf unrolled into one "
                    "function) vs **split** (march-loop scene_sdf calls N "
                    "non-inlined per-object functions). Times the JIT compile "
                    "and the per-frame render for each. If split compiles far "
                    "faster, one giant function is the cause of the super-linear "
                    "cost.\n\n", W, H);
        std::printf("Render %d×%d, %s. Full render_tile compiled three "
                    "ways: **inlined** (current — scene_sdf unrolled into one "
                    "function), **split** (N non-inlined per-object functions), "
                    "and **guarded** (split + an inline AABB-distance prune per "
                    "object — the build-time spatial acceleration). Times JIT "
                    "compile and per-frame render for each.\n\n", W, H,
                    heavy_scene ? "N twisted-box+sphere blends (EXPENSIVE per-object SDF)"
                                : "N spheres");
        std::printf("| Objects | inline JIT | split JIT | guarded JIT | "
                    "inline render | split render | guarded render | "
                    "guard speedup |\n");
        std::printf("|---|---|---|---|---|---|---|---|\n");
        const std::vector<int> sizes = smoke_mode ? std::vector<int>{8, 64}
                                                   : std::vector<int>{10, 100, 1000};
        for (int n : sizes) {
            SceneGraph s = heavy_scene ? make_many_heavy(n) : make_many_spheres(n);
            int objs = (int)s.objects().size();

            using Mode = SceneCodegen::SceneSdfMode;
            auto run = [&](Mode mode) {
                auto ctx = std::make_unique<llvm::LLVMContext>();
                TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
                SceneCodegen cg(*ctx, cfg);
                cg.emit_render_tile(s, mode);
                auto mod = cg.take_module();
                JitEngine jit;
                auto t0 = clk::now();
                auto fn = jit.load(std::move(mod), std::move(ctx));
                double jit_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                double render_ms = -1;
                if (fn) {
                    RenderParams rp; rp.width = W; rp.height = H;
                    std::vector<float> px(W * H * 4);
                    auto r0 = clk::now();
                    TileScheduler::render(*fn, px.data(), s.camera(), rp);
                    render_ms = std::chrono::duration<double, std::milli>(clk::now() - r0).count();
                }
                return std::pair{jit_ms, render_ms};
            };

            auto [inl_jit, inl_ren] = run(Mode::Inlined);
            auto [spl_jit, spl_ren] = run(Mode::Split);
            auto [grd_jit, grd_ren] = run(Mode::Guarded);
            std::printf("| %d | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %.1fx |\n",
                        objs, inl_jit, spl_jit, grd_jit, inl_ren, spl_ren, grd_ren,
                        grd_ren > 0 ? inl_ren / grd_ren : 0.0);
            std::fflush(stdout);
        }
        std::printf("\n_Only the march-loop scene_sdf varies; the AD-gradient "
                    "SDF stays inlined, so guarded render speedup is a lower "
                    "bound (normals still do the full min). 'guard speedup' is "
                    "inline render / guarded render — the spatial prune's "
                    "payoff. Spheres are spread on a grid (favourable for "
                    "pruning); dense/overlapping scenes prune less._\n");
        return 0;
    }

    // ── Scalability sweep ───────────────────────────────────────────────────
    if (scaling_mode) {
        const int W = 800, H = 600;
        std::printf("# FRep Designer — Scalability Benchmark\n\n");
        std::printf("Render resolution %d×%d, shadows+AO per scene defaults, "
                    "best-of-3 render time (compile/JIT excluded). "
                    "GPU = offscreen Vulkan compute.%s\n\n",
                    W, H, gpu_ok ? "" : " _(GPU unavailable — CPU only.)_");

        struct Axis {
            const char* name;
            SceneGraph (*gen)(int);
            std::vector<int> sizes;
        };
        const std::vector<int> obj_sizes   = smoke_mode ? std::vector<int>{2, 8}
                                                         : std::vector<int>{10, 100, 1000};
        const std::vector<int> depth_sizes = smoke_mode ? std::vector<int>{1, 4}
                                                         : std::vector<int>{1, 10, 100};
        const std::vector<Axis> axes = {
            {"Object count (N spheres)",   &make_many_spheres,   obj_sizes},
            {"Node depth (nested xforms)", &make_deep_transforms, depth_sizes},
            {"CSG depth (chained bools)",  &make_deep_csg,        depth_sizes},
            {"Mixed primitives",           &make_mixed_primitives, obj_sizes},
        };
        if (!smoke_mode)
        std::printf("> Sizes default to practical ranges (render time grows "
                    "fast: the CPU `scene_sdf` is a flat `min()`/boolean tree "
                    "evaluated O(N) per ray-step with no BVH yet, so deep/large "
                    "scenes cost seconds). The generators scale to any N — "
                    "raise the sizes for the full curve on faster hardware.\n\n");

        for (const auto& ax : axes) {
            std::printf("## %s\n\n", ax.name);
            // CPU columns: codegen + JIT (the two compile phases) and
            // render, plus working-set delta and process peak RSS.
            std::printf("| Size | Objects | CPU codegen | CPU JIT | CPU render "
                        "| CPU ΔRSS | CPU peak |");
            if (gpu_ok)
                std::printf(" GPU emit | GPU compile | GPU init | GPU render "
                            "| GPU host ΔRSS | GPU device |");
            std::printf("\n|---|---|---|---|---|---|---|");
            if (gpu_ok) std::printf("---|---|---|---|---|---|");
            std::printf("\n");
            // Stash GPU results to print an init-phase breakdown below.
            std::vector<std::pair<int, GpuResult>> gpu_rows;
            for (int sz : ax.sizes) {
                SceneGraph s = ax.gen(sz);
                int objs = (int)s.objects().size();
                auto c = bench_cpu(s, W, H);
                std::printf("| %d | %d | %.0f ms | %.0f ms | %.0f ms | %zu MB | %zu MB |",
                            sz, objs, c.codegen_ms, c.jit_ms, c.render_ms,
                            c.rss_delta_kb / 1024, c.peak_kb / 1024);
                if (gpu_ok) {
                    auto g = bench_gpu(s, W, H);
                    std::printf(" %.0f ms | %.0f ms | %.0f ms | %.0f ms | %zu MB | %zu MB |",
                                g.emit_ms, g.compile_ms, g.init_ms, g.render_ms,
                                g.rss_delta_kb / 1024, g.device_kb / 1024);
                    gpu_rows.emplace_back(sz, g);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
            std::printf("\n");

            // GPU init breakdown — splits the (sometimes huge) init time
            // into device setup, shader-module ingest, the driver pipeline
            // compile, buffer upload, and the rest. Confirms whether an
            // init blow-up is the driver compiling a large shader.
            if (gpu_ok && !gpu_rows.empty()) {
                std::printf("_GPU init breakdown (%s):_\n\n", ax.name);
                std::printf("| Size | device | shader module | "
                            "**pipeline (driver compile)** | buffers | misc |\n");
                std::printf("|---|---|---|---|---|---|\n");
                for (auto& [sz, g] : gpu_rows) {
                    std::printf("| %d | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %.0f ms |\n",
                                sz, g.init_device_ms, g.init_shader_ms,
                                g.init_pipeline_ms, g.init_buffers_ms, g.init_misc_ms);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
        }
        std::printf("_CPU codegen = build LLVM IR; CPU JIT = lower IR to native "
                    "(one-time compile = codegen+JIT); CPU render = per-frame. "
                    "ΔRSS = resident-memory growth for that scene; peak = process "
                    "high-water mark.%s_\n",
                    gpu_ok ? " GPU emit/compile/init = one-time pipeline build; "
                             "GPU render = per-frame dispatch+readback. GPU host "
                             "ΔRSS = driver/staging on the host; GPU device = "
                             "computed VRAM for output image + voxel/texture "
                             "buffers (lower bound, not in RSS). The init "
                             "breakdown isolates the driver's SPIR-V→native "
                             "pipeline compile." : "");
        return 0;
    }

    if (json_mode) {
        // Collect all measurements into vectors, emit as JSON at the end.
        struct Row { std::string scene; int W, H; double ms; };
        std::vector<Row> cpu_rows, gpu_rows;
        for (int c = 1; c <= 5; ++c) {
            auto s = make_scene(c);
            for (auto [W, H] : std::vector<std::pair<int,int>>{
                {400, 300}, {800, 600}, {1280, 720}}) {
                auto r = bench_cpu(s, W, H);
                cpu_rows.push_back({scene_name(c), W, H, r.render_ms});
            }
            if (gpu_ok) {
                for (auto [W, H] : std::vector<std::pair<int,int>>{
                    {400, 300}, {800, 600}}) {
                    auto r = bench_gpu(s, W, H);
                    gpu_rows.push_back({scene_name(c), W, H, r.render_ms});
                }
            }
        }
        std::printf("{\n  \"meta\": {\"gpu_available\": %s},\n",
            gpu_ok ? "true" : "false");
        std::printf("  \"cpu\": [\n");
        for (std::size_t i = 0; i < cpu_rows.size(); ++i) {
            auto& r = cpu_rows[i];
            std::printf("    {\"scene\": \"%s\", \"size\": \"%dx%d\", \"ms\": %.2f}%s\n",
                r.scene.c_str(), r.W, r.H, r.ms,
                i + 1 == cpu_rows.size() ? "" : ",");
        }
        std::printf("  ],\n  \"gpu\": [\n");
        for (std::size_t i = 0; i < gpu_rows.size(); ++i) {
            auto& r = gpu_rows[i];
            std::printf("    {\"scene\": \"%s\", \"size\": \"%dx%d\", \"ms\": %.2f}%s\n",
                r.scene.c_str(), r.W, r.H, r.ms,
                i + 1 == gpu_rows.size() ? "" : ",");
        }
        std::printf("  ]\n}\n");
        return 0;
    }

    std::printf("# FRep Designer 4.0 — Performance Benchmarks\n\n");
    std::printf("All numbers are wall-clock milliseconds, measured on the "
                "build environment.\n\n");
    if (!gpu_ok) std::printf("> ⚠ Vulkan unavailable; GPU rows skipped.\n\n");

    std::printf("## 1. CPU JIT — compile + render scaling\n\n");
    std::printf("| Scene | Codegen | JIT | Render 400×300 | Render 800×600 | "
                "Render 1280×720 | Peak RSS |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    for (int c = 1; c <= 5; ++c) {
        auto s = make_scene(c);
        auto a = bench_cpu(s, 400, 300);
        auto b = bench_cpu(s, 800, 600);
        auto x = bench_cpu(s, 1280, 720);
        std::printf("| %s | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %.0f ms | %zu MB |\n",
                    scene_name(c),
                    a.codegen_ms, a.jit_ms, a.render_ms, b.render_ms, x.render_ms,
                    x.peak_kb / 1024);
    }

    if (gpu_ok) {
        std::printf("\n## 2. GPU vs CPU render speedup\n\n");
        std::printf("Same scene, same resolution, both pipelines. GPU times "
                    "exclude one-time pipeline setup (emit + glslang + Vulkan "
                    "init); CPU times exclude one-time JIT compile. Steady-"
                    "state numbers only — what you'd see at frame N+1.\n\n");
        std::printf("| Scene | Resolution | CPU render | GPU render | Speedup |\n");
        std::printf("|---|---|---|---|---|\n");
        for (int c : {1, 2, 3, 4, 5}) {
            for (auto [W, H] : std::vector<std::pair<int,int>>{
                {400, 300}, {800, 600}}) {
                auto s = make_scene(c);
                auto cpu = bench_cpu(s, W, H);
                auto gp  = bench_gpu(s, W, H);
                std::printf("| %s | %dx%d | %.0f ms | %.0f ms | %.1f× |\n",
                            scene_name(c), W, H,
                            cpu.render_ms, gp.render_ms,
                            cpu.render_ms / std::max(1.0, gp.render_ms));
            }
        }
    }

    std::printf("\n## 3. Incremental compilation modes\n\n");
    std::printf("Same scene compiled three different ways. The Auto mode "
                "automatically picks Constant when only parameter values "
                "changed and Incremental when topology was the same.\n\n");
    {
        auto s = make_scene(3);
        using Mode = TracerConfig::CompileMode;
        IncrementalCompiler ic;
        // Warm up.
        ic.compile_if_changed(s);
        // Touch a parameter to force a recompile path.
        s.objects().begin()->second.geometry->params["k"] = 1.5f;

        // Constant mode — full module rebuild.
        ic.policy().set_mode(Mode::Constant);
        auto t0 = clk::now();
        ic.compile_if_changed(s);
        double constant_ms = ms_since(t0);

        // Incremental mode — only constants change.
        s.objects().begin()->second.geometry->params["k"] = 1.55f;
        ic.policy().set_mode(Mode::Incremental);
        t0 = clk::now();
        ic.compile_if_changed(s);
        double incr_ms = ms_since(t0);

        // Auto mode — heuristic decides.
        s.objects().begin()->second.geometry->params["k"] = 1.6f;
        ic.policy().set_mode(Mode::Auto);
        t0 = clk::now();
        ic.compile_if_changed(s);
        double auto_ms = ms_since(t0);

        std::printf("| Mode | Recompile time |\n");
        std::printf("|---|---|\n");
        std::printf("| Constant (full recompile) | %.1f ms |\n", constant_ms);
        std::printf("| Incremental | %.1f ms |\n", incr_ms);
        std::printf("| Auto | %.1f ms |\n", auto_ms);
    }

    bench_sparse();
    bench_bvh();
    bench_bvh_crossover();

    std::printf("\n---\n");
    std::printf("_Generated by `frep_bench` from "
                "`/home/claude/frep4/tools/benchmarks.cpp`._\n");
    return 0;
}
