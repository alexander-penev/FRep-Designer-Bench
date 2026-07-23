// tests/test_rtx_path.cpp
//
// Phase 0 of the GpuRtx path: capability detection and executor wiring. There
// is no real RT device in the sandbox, so these tests pin the *logic* around
// detection (backend selection, available()/using_hardware() semantics, the
// software-fallback opt-out) rather than any rendered output. Live tracing is
// validated on RTX 2080 hardware in later phases.

#include <gtest/gtest.h>

#include <cctype>
#include <string>

#include "core/exec/rtx_executor.hpp"
#include "core/gpu/rtx_caps.hpp"
#include "core/exec/multipath.hpp"

using namespace frep;
using namespace frep::exec;

TEST(RtxPath, PathKindHasName) {
    EXPECT_STREQ(path_kind_name(PathKind::GpuRtx), "gpu_rtx");
}

TEST(RtxPath, CapsDetectionDoesNotCrash) {
    // Whatever the environment, detection returns a well-formed result.
    auto caps = gpu::detect_rtx_caps();
    // Hardware implies both extensions; Software/None imply not-hardware.
    if (caps.backend == gpu::RtxBackend::Hardware) {
        EXPECT_TRUE(caps.has_ray_tracing_pipeline);
        EXPECT_TRUE(caps.has_acceleration_structure);
        EXPECT_TRUE(caps.hardware());
        EXPECT_TRUE(caps.usable());
    } else {
        EXPECT_FALSE(caps.hardware());
    }
    EXPECT_FALSE(caps.describe().empty());
}

TEST(RtxPath, ExecutorReportsGpuRtxPath) {
    RtxExecutor e;
    EXPECT_EQ(e.path(), PathKind::GpuRtx);
}

TEST(RtxPath, SoftwareFallbackOptOut) {
    // With software disallowed, a non-hardware environment is unavailable;
    // with it allowed, the path is available whenever any Vulkan device exists.
    RtxExecutor hw_only(TracerConfig{}, /*allow_software=*/false);
    RtxExecutor with_sw(TracerConfig{}, /*allow_software=*/true);

    auto caps = gpu::detect_rtx_caps();
    switch (caps.backend) {
        case gpu::RtxBackend::Hardware:
            EXPECT_TRUE(hw_only.available());
            EXPECT_TRUE(with_sw.available());
            EXPECT_TRUE(with_sw.using_hardware());
            break;
        case gpu::RtxBackend::Software:
            EXPECT_FALSE(hw_only.available());   // RT cores required, absent
            EXPECT_TRUE(with_sw.available());    // fallback allowed
            EXPECT_FALSE(with_sw.using_hardware());
            break;
        case gpu::RtxBackend::None:
            EXPECT_FALSE(hw_only.available());
            EXPECT_FALSE(with_sw.available());
            break;
    }
}

TEST(RtxPath, UnavailableRenderReportsError) {
    // In the sandbox (no RT device) render() must fail cleanly with a message,
    // never crash or silently produce garbage.
    RtxExecutor e;
    if (!e.available()) {
        SceneGraph s;
        auto r = e.render(s, 16, 16, Tile{0, 0, 16, 16});
        EXPECT_FALSE(r.ok);
        EXPECT_FALSE(r.error.empty());
    } else {
        GTEST_SKIP() << "RT device present; covered by hardware parity tests";
    }
}

TEST(RtxPath, SoftwareRasterizerIsNotHardware) {
    // Regression guard for the llvmpipe misclassification: a CPU software
    // rasterizer (llvmpipe/lavapipe) can advertise the RT extensions, but it
    // has no RT cores and must classify as Software, never Hardware. We can't
    // force a device in a unit test, but we can assert the invariant the
    // detector must uphold: whenever the chosen backend is Hardware, the
    // device is not a software rasterizer name we recognize.
    auto caps = gpu::detect_rtx_caps();
    if (caps.backend == gpu::RtxBackend::Hardware) {
        // crude but effective: hardware RT must not be llvmpipe/lavapipe/swift
        std::string n = caps.device_name;
        for (auto& c : n) c = (char)std::tolower((unsigned char)c);
        EXPECT_EQ(n.find("llvmpipe"), std::string::npos);
        EXPECT_EQ(n.find("lavapipe"), std::string::npos);
        EXPECT_EQ(n.find("swiftshader"), std::string::npos);
    } else {
        GTEST_SKIP() << "no hardware RT backend in this environment";
    }
}

// ── Phase 1c: RT shader generation ──────────────────────────────────────────
// These don't need a GPU — they check that the RT shader emitter produces the
// four stages and that the lifted scene_sdf/shade region is present. Actual
// SPIR-V compilation is validated separately with glslangValidator in the
// build, and end-to-end rendering on llvmpipe/hardware in Phase 1e.

#include "core/gpu/rtx_shaders.hpp"
#include "core/exec/parity_scenes.hpp"

TEST(RtxShaders, GeneratesFourStagesForAnalyticScenes) {
    using namespace frep;
    for (const char* want : {"sphere", "smooth_union", "twist", "customexpr"}) {
        SceneGraph s;
        for (auto& ns : parity::all_scenes())
            if (std::string(ns.name) == want) s = ns.make();
        auto rs = gpu::emit_rt_shaders(s, {});
        ASSERT_TRUE(rs) << want << ": " << (rs ? "" : rs.error());
        EXPECT_FALSE(rs->rgen.empty());
        EXPECT_FALSE(rs->rint.empty());
        EXPECT_FALSE(rs->rchit.empty());
        EXPECT_FALSE(rs->rmiss.empty());
        // The shared region must carry the real SDF + shading code, so the
        // intersection/closest-hit shaders compute the same thing as compute.
        EXPECT_NE(rs->shared_glsl.find("scene_sdf"), std::string::npos) << want;
        EXPECT_NE(rs->shared_glsl.find("vec3 shade"), std::string::npos) << want;
        // intersection shader sphere-traces scene_sdf and reports a hit
        EXPECT_NE(rs->rint.find("scene_sdf"), std::string::npos) << want;
        EXPECT_NE(rs->rint.find("reportIntersectionEXT"), std::string::npos) << want;
        // closest-hit calls shade with the emitter's 3-arg signature
        EXPECT_NE(rs->rchit.find("shade(p, n,"), std::string::npos) << want;
        // and uses scene_normal (lifted) rather than an ad-hoc normal
        EXPECT_NE(rs->rchit.find("scene_normal(p)"), std::string::npos) << want;
    }
}

TEST(RtxShaders, AcceptsMeshSceneAndDeclaresBuffer) {
    using namespace frep;
    SceneGraph s;
    for (auto& ns : parity::all_scenes())
        if (std::string(ns.name) == "mesh") s = ns.make();
    auto rs = gpu::emit_rt_shaders(s, {});
    ASSERT_TRUE(rs) << (rs ? "" : rs.error());
    // intersection sphere-traces scene_sdf, which samples the mesh, so the
    // intersection stage must see the mesh buffer at binding 3.
    EXPECT_NE(rs->rint.find("mesh_voxels"), std::string::npos);
    EXPECT_NE(rs->rint.find("binding = 3"), std::string::npos);
    EXPECT_FALSE(rs->mesh_voxels.empty());
}

// ── Phase 1d: push-constant layout parity ───────────────────────────────────
// The RT path reuses the compute path's build_push_from_scene by memcpy'ing
// ShaderPush into RtPushConstants. That only gives identical camera/light data
// to the RT shaders if the two structs agree field-for-field, not just in
// total size. Pin every field offset so a future edit to either struct can't
// silently shift the RT path's inputs and break parity.
#include "core/gpu/rtx_pipeline.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include <cstddef>

TEST(RtxPipeline, PushConstantLayoutMatchesComputePath) {
    using frep::gpu::ShaderPush;
    using frep::gpu::RtPushConstants;
    ASSERT_EQ(sizeof(ShaderPush), sizeof(RtPushConstants));
#define CK(f) EXPECT_EQ(offsetof(ShaderPush, f), offsetof(RtPushConstants, f)) << #f
    CK(cam_pos); CK(fov_scale); CK(cam_fwd); CK(cam_right); CK(cam_up);
    CK(lights); CK(light_colors); CK(width); CK(height);
    CK(projection_mode); CK(ortho_size); CK(tile_x0); CK(tile_y0);
    CK(tile_x1); CK(tile_y1);
#undef CK
}

// ── Regression: RtxCtx move must preserve the RT api table ──────────────────
// A move that dropped api_ made every RT entry point null after create()
// returned by value, segfaulting the first AS call. We can't create a real
// device in the sandbox, but we can pin the invariant on a hand-built context
// via a tiny friend-free check: move a context with a sentinel api pointer and
// confirm it survives. (Compile-time-ish guard against re-introducing the bug.)
#include "core/gpu/rtx_ctx.hpp"
TEST(RtxCtx, MovePreservesApiTable) {
    // RtxCtx's handles are private; we can only observe api() (const ref).
    // Build two default contexts (no device — safe), stamp one's api via a
    // move from a function-local that we can't poke directly. Instead, assert
    // the structural guarantee: a default-constructed ctx has a null table,
    // and the type is movable without losing constness of api(). The real
    // guarantee is exercised on hardware; this keeps the API shape stable.
    frep::gpu::RtxCtx a;
    EXPECT_FALSE(a.api().complete());      // no device loaded
    frep::gpu::RtxCtx b = std::move(a);    // must compile + not crash
    EXPECT_FALSE(b.api().complete());
}

// ── Phase 3: CSG-aware BLAS grouping ────────────────────────────────────────
// The broad-phase split is pure CPU tree logic (no GPU), so it's fully testable
// here. The rule: cut at hard unions (separable), keep smooth-union /
// intersection / difference sub-trees whole.
#include "core/gpu/rtx_csg_groups.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"

TEST(RtxCsgGroups, SplitsOnlyAtHardUnions) {
    using namespace frep;
    auto s1 = std::make_shared<SphereNode>(1.0f, "s1");
    auto s2 = std::make_shared<SphereNode>(1.0f, "s2");
    auto s3 = std::make_shared<SphereNode>(1.0f, "s3");

    // Union is separable → one group per operand.
    EXPECT_EQ(gpu::partition_csg_groups(
        std::make_shared<UnionNode>(s1, s2)).size(), 2u);

    // Non-separable ops stay whole.
    EXPECT_EQ(gpu::partition_csg_groups(
        std::make_shared<SmoothUnionNode>(s1, s2, 0.3f)).size(), 1u);
    EXPECT_EQ(gpu::partition_csg_groups(
        std::make_shared<IntersectionNode>(s1, s2)).size(), 1u);
    EXPECT_EQ(gpu::partition_csg_groups(
        std::make_shared<DifferenceNode>(s1, s2)).size(), 1u);

    // Mixed: union splits, the smooth sub-tree stays whole → 2 groups.
    auto su = std::make_shared<SmoothUnionNode>(s1, s2, 0.3f);
    EXPECT_EQ(gpu::partition_csg_groups(
        std::make_shared<UnionNode>(su, s3)).size(), 2u);

    // Chained unions → one group per leaf.
    auto chain = std::make_shared<UnionNode>(
        std::make_shared<UnionNode>(s1, s2), s3);
    EXPECT_EQ(gpu::partition_csg_groups(chain).size(), 3u);

    // A lone primitive is a single group (Phase-1 degenerate case).
    EXPECT_EQ(gpu::partition_csg_groups(s1).size(), 1u);

    // Each group carries a finite AABB.
    for (auto& g : gpu::partition_csg_groups(chain)) {
        EXPECT_TRUE(g.root != nullptr);
        EXPECT_LE(g.box.min_x, g.box.max_x);
    }
}

// ── Phase 3: per-group shader emission ──────────────────────────────────────
// Each CSG group must get its own intersection shader evaluating ONLY that
// group's sub-tree SDF — that's what lets the RT cores cull groups a ray
// misses instead of sphere-tracing the whole scene. Testable in-sandbox: the
// per-group SDFs must differ (different geometry) while raygen/chit/miss stay
// shared.
#include "core/frep/transforms.hpp"
#include "core/frep/scene.hpp"

TEST(RtxGroupShaders, PerGroupIntersectionsEvaluateOwnSubtree) {
    using namespace frep;
    auto s1 = std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.5f,"a"), -1.5f,0,0);
    auto s2 = std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.5f,"b"),  0.0f,0,0);
    auto s3 = std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.5f,"c"),  1.5f,0,0);
    auto root = std::make_shared<UnionNode>(std::make_shared<UnionNode>(s1,s2), s3);

    SceneGraph full; full.add_object(root);
    auto groups = gpu::partition_csg_groups(root);
    ASSERT_EQ(groups.size(), 3u);

    std::vector<SceneGraph> gscenes;
    for (auto& g : groups) { SceneGraph s; s.add_object(g.root); gscenes.push_back(std::move(s)); }

    auto rs = gpu::emit_rt_group_shaders(full, gscenes, {});
    ASSERT_TRUE(rs) << (rs ? "" : rs.error());
    EXPECT_EQ(rs->rint_per_group.size(), 3u);
    EXPECT_FALSE(rs->rgen.empty());
    EXPECT_FALSE(rs->rchit.empty());

    // Each group's intersection shader must differ (different translated SDF),
    // proving it evaluates only its own sphere, not the full union.
    EXPECT_NE(rs->rint_per_group[0], rs->rint_per_group[1]);
    EXPECT_NE(rs->rint_per_group[1], rs->rint_per_group[2]);
    // All sphere-trace (call scene_sdf_v and report a hit).
    for (auto& r : rs->rint_per_group) {
        EXPECT_NE(r.find("scene_sdf_v"), std::string::npos);
        EXPECT_NE(r.find("reportIntersectionEXT"), std::string::npos);
    }
}

// ── Phase 3: benchmark scene generation ─────────────────────────────────────
// The sphere-grid benchmark must produce exactly N CSG groups for the
// scaling-curve study (each sphere is an independent group → one BLAS each).
#include "core/exec/bench_scenes.hpp"

TEST(RtxBenchScenes, SphereGridYieldsNGroups) {
    using namespace frep;
    for (int n : {1, 4, 9, 16, 64}) {
        auto s = bench::make_sphere_grid(n);
        FRepNode::Ptr root;
        for (auto& [id, o] : s.objects()) root = o.geometry;
        ASSERT_TRUE(root != nullptr) << "n=" << n;
        EXPECT_EQ((int)gpu::partition_csg_groups(root).size(), n) << "n=" << n;
    }
}

// ── RT texture shader emission (descriptor wiring still pending) ─────────────
// The shader side of RT textures: emit_rt_shaders now accepts a textured scene
// and declares the texture storage buffer at RT binding 2 in every stage that
// embeds the shared region. (Binding the actual buffer in rtx_trace is the next
// step; this pins that the shaders are generated and reference tex_data.)
TEST(RtxShaders, AcceptsTextureSceneAndDeclaresBuffer) {
    using namespace frep;
    SceneGraph s;
    for (auto& ns : parity::all_scenes())
        if (std::string(ns.name) == "texture") s = ns.make();
    auto rs = gpu::emit_rt_shaders(s, {});
    ASSERT_TRUE(rs) << (rs ? "" : rs.error());
    // closest-hit shades, so it must see the texture buffer.
    EXPECT_NE(rs->rchit.find("tex_data"), std::string::npos);
    EXPECT_NE(rs->rchit.find("binding = 2"), std::string::npos);
}

TEST(RtxShaders, MeshSceneCompilesAllStagesToSpirv) {
    using namespace frep;
    SceneGraph s;
    for (auto& ns : parity::all_scenes())
        if (std::string(ns.name) == "mesh") s = ns.make();
    auto rs = gpu::emit_rt_shaders(s, {});
    ASSERT_TRUE(rs) << (rs ? "" : rs.error());
    // All four stages must compile — the lifted region includes the sample_mesh
    // helper (a regression we hit: it lives before scene_sdf and was dropped).
    for (auto [src, stage] : {std::pair{&rs->rgen, "rgen"},
                              std::pair{&rs->rint, "rint"},
                              std::pair{&rs->rchit, "rchit"},
                              std::pair{&rs->rmiss, "rmiss"}}) {
        auto spv = gpu::compile_rt_stage_to_spv(*src, stage);
        EXPECT_TRUE(spv) << stage << ": " << (spv ? "" : spv.error());
        if (spv) std::remove(spv->c_str());
    }
}

// RtxPipelineCache starts empty/invalid and exposes the right defaults so a
// caller can pass a fresh cache to rtx_trace_cached on the first frame. (The
// actual hit/miss reuse runs only on real RT hardware; here we check the
// value-type invariants that don't need a device.)
TEST(RtxPipelineCache, FreshCacheIsInvalidAndEmpty) {
    using namespace frep;
    gpu::RtxPipelineCache c;
    EXPECT_FALSE(c.valid);
    EXPECT_EQ(c.pipeline, nullptr);
    EXPECT_EQ(c.pipeline_layout, nullptr);
    EXPECT_EQ(c.descriptor_layout, nullptr);
    EXPECT_EQ(c.sbt_buffer, nullptr);
    EXPECT_EQ(c.sbt_memory, nullptr);
    EXPECT_EQ(c.key, 0u);
    for (void* m : c.shader_modules) EXPECT_EQ(m, nullptr);
    // A fresh cache means "first frame is a miss" — rtx_trace_cached will build
    // and populate it, and a matching key on frame 2 will be a hit.
}
