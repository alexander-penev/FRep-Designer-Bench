// tests/test_gpu_patterns_mesh.cpp
//
// Tests for two GLSL-emitter extensions:
//   1. Procedural patterns (Checker/Stripes/GradientY/Noise) in scene_albedo
//   2. MeshSDFNode → storage-buffer-backed trilinear sampling in the shader

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>

using namespace frep;

namespace {

bool glslang_available() { return !gpu::find_glslang().empty(); }

gpu::ShaderPush sane_push(int w, int h) {
    float cam[3]   = {0, 1, 4};
    float tgt[3]   = {0, 0, 0};
    float light[3] = {4, 6, 4};
    return frep::gpu::build_push_simple(cam, tgt, light, w, h);
}

} // anon

// ── Pattern emission tests (no Vulkan needed) ──────────────────────────────

TEST(GpuPatterns, CheckerEmitsMix) {
    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::Checker;
    m.albedo  = {1, 0, 0};
    m.albedo2 = {0, 1, 0};
    m.pattern_scale = 5.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("mix("), std::string::npos);
    EXPECT_NE(r->source.find("floor("), std::string::npos);  // checker
}

TEST(GpuPatterns, StripesEmitsFloorY) {
    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::Stripes;
    m.pattern_scale = 8.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("floor(8"), std::string::npos);
}

TEST(GpuPatterns, GradientEmitsClamp) {
    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::GradientY;
    m.pattern_scale = 1.5f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("clamp("), std::string::npos);
}

TEST(GpuPatterns, NoiseEmitsHashConstants) {
    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::Noise;
    m.pattern_scale = 10.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("0x9E3779B9"), std::string::npos);
}

// ── End-to-end pattern render ──────────────────────────────────────────────

TEST(GpuPatterns, CheckerRenderProducesTwoColours) {
    if (!glslang_available())             GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())     GTEST_SKIP() << "no Vulkan";

    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::Checker;
    m.albedo  = {1, 0, 0};
    m.albedo2 = {0, 1, 0};
    m.pattern_scale = 5.0f;
    s.add_object(std::make_shared<SphereNode>(1.5f, "ball"), m);

    auto e = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(e.has_value());
    auto spv = gpu::compile_glsl_to_spv(e->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 128, H = 128;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());

    // Count red-dominant vs green-dominant pixels — both should be visible.
    int reds = 0, greens = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 50 && r > g + 30 && r > b + 30) ++reds;
        if (g > 50 && g > r + 30 && g > b + 30) ++greens;
    }
    EXPECT_GT(reds, 50);
    EXPECT_GT(greens, 50);
}

// ── MeshSDF GPU support ────────────────────────────────────────────────────

TEST(GpuMeshSdf, EmitProducesStorageBuffer) {
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams p; p.rx=p.ry=p.rz=10;
    auto m = mesh::extract_iso_mesh(ref, p);

    SceneGraph s;
    s.add_object(std::make_shared<MeshSDFNode>(m, 16, "msph"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->mesh_count, 0);
    EXPECT_GT(r->mesh_voxels.size(), 0u);
    EXPECT_NE(r->source.find("MeshData"), std::string::npos);
    EXPECT_NE(r->source.find("sample_mesh_0"), std::string::npos);
}

TEST(GpuMeshSdf, DeduplicatesVoxelsAcrossBodies) {
    // The SDF, albedo, and PBR bodies each emit every node, so a MeshSDFNode is
    // visited multiple times. The voxel grid must be uploaded ONCE (one slot,
    // reused), not duplicated per visit — earlier this doubled mesh_voxels and
    // produced mesh_count == 2 for a single mesh.
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams p; p.rx = p.ry = p.rz = 10;
    auto m = mesh::extract_iso_mesh(ref, p);

    const int RES = 16;
    SceneGraph s;
    s.add_object(std::make_shared<MeshSDFNode>(m, RES, "msph"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    // Exactly one mesh slot, one grid copy (RES^3 floats), one sample function.
    EXPECT_EQ(r->mesh_count, 1);
    EXPECT_EQ(r->mesh_voxels.size(), (std::size_t)RES * RES * RES);
    EXPECT_NE(r->source.find("float sample_mesh_0"), std::string::npos);
    EXPECT_EQ(r->source.find("sample_mesh_1"), std::string::npos);  // no 2nd slot
}

TEST(GpuMeshSdf, NoMeshNoBuffer) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->mesh_count, 0);
    EXPECT_TRUE(r->mesh_voxels.empty());
    EXPECT_EQ(r->source.find("MeshData"), std::string::npos);
}

TEST(GpuMeshSdf, RendersOnGpu) {
    if (!glslang_available())             GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())     GTEST_SKIP() << "no Vulkan";

    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams p; p.rx=p.ry=p.rz=10;
    auto m = mesh::extract_iso_mesh(ref, p);

    SceneGraph s;
    Material mat{{0.9f, 0.5f, 0.3f}};
    s.add_object(std::make_shared<MeshSDFNode>(m, 16, "msph"), mat);

    auto e = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(e.has_value());
    auto spv = gpu::compile_glsl_to_spv(e->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv, e->mesh_voxels);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 96, H = 96;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());

    // The voxelized sphere should be visibly red (orange dominant).
    int red_dom = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        // PBR shading: full-bright albedo (0.9, 0.4, 0.3) tops out around
        // r=60..80 due to energy-conserving Cook-Torrance. So we just
        // check red-dominance, not absolute brightness.
        if (r > 30 && r > b + 10 && r > g + 10) ++red_dom;
    }
    EXPECT_GT(red_dom, 80);
}
