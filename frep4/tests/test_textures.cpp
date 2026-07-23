// tests/test_textures.cpp
//
// Tests for image texture mapping: BMP loader round-trip + GLSL emission
// + end-to-end Vulkan render of a textured SDF.

#include "core/io/bmp_loader.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
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

io::Image make_checker_image(int W, int H, int cell) {
    io::Image img;
    img.width = W; img.height = H;
    img.rgba.assign(static_cast<std::size_t>(W) * H * 4, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            std::size_t i = (y * W + x) * 4;
            bool on = ((x / cell) + (y / cell)) & 1;
            img.rgba[i + 0] = on ? 240 : 30;
            img.rgba[i + 1] = on ? 80  : 200;
            img.rgba[i + 2] = on ? 60  : 220;
            img.rgba[i + 3] = 255;
        }
    return img;
}

gpu::ShaderPush sane_push(int w, int h) {
    float cam[3]   = {0, 1, 4};
    float tgt[3]   = {0, 0, 0};
    float light[3] = {4, 6, 4};
    return frep::gpu::build_push_simple(cam, tgt, light, w, h);
}

} // anon

// ── BMP loader round-trip ─────────────────────────────────────────────────

TEST(Textures, BmpRoundTrip) {
    auto img = make_checker_image(8, 8, 2);
    ASSERT_TRUE(io::save_bmp(img, "/tmp/gtest_tex.bmp"));
    auto loaded = io::load_bmp("/tmp/gtest_tex.bmp");
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.width,  img.width);
    EXPECT_EQ(loaded.height, img.height);
    // Pixels should match exactly (no compression).
    int mismatches = 0;
    for (std::size_t i = 0; i < img.rgba.size(); i += 4) {
        for (int k = 0; k < 3; ++k) {
            if (img.rgba[i + k] != loaded.rgba[i + k]) ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

TEST(Textures, BmpLoadMissingFile) {
    auto img = io::load_bmp("/does/not/exist.bmp");
    EXPECT_TRUE(img.empty());
}

// ── Emit-only tests (no Vulkan) ───────────────────────────────────────────

TEST(GpuTextures, EmitDeclaresBuffer) {
    auto img = make_checker_image(16, 16, 4);
    SceneGraph s;
    Material m;
    m.pattern        = Material::Pattern::Texture;
    m.texture_rgba   = img.rgba;
    m.texture_width  = img.width;
    m.texture_height = img.height;
    m.pattern_scale  = 1.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->texture_count, 1);
    EXPECT_GT(r->texture_pixels.size(), 0u);
    EXPECT_NE(r->source.find("TextureData"), std::string::npos);
    EXPECT_NE(r->source.find("triplanar_sample_0"), std::string::npos);
}

TEST(GpuTextures, NoTextureNoBuffer) {
    SceneGraph s;
    Material m;  // default Solid
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->texture_count, 0);
    EXPECT_TRUE(r->texture_pixels.empty());
    EXPECT_EQ(r->source.find("TextureData"), std::string::npos);
}

// ── End-to-end Vulkan render ──────────────────────────────────────────────

TEST(GpuTextures, RendersOnGpu) {
    if (!glslang_available())          GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())  GTEST_SKIP() << "no Vulkan";

    // Big-cell checker — guaranteed to show both colors on a 1-radius sphere.
    auto img = make_checker_image(32, 32, 8);
    SceneGraph s;
    Material m;
    m.pattern        = Material::Pattern::Texture;
    m.texture_rgba   = img.rgba;
    m.texture_width  = img.width;
    m.texture_height = img.height;
    m.pattern_scale  = 1.5f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);

    auto e = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(e.has_value());
    auto spv = gpu::compile_glsl_to_spv(e->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv, e->mesh_voxels, e->texture_pixels);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 96, H = 96;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());

    // Texture has colors (240,80,60) and (30,200,220). With PBR shading
    // energy conservation drops absolute output by 3-4×, so the test
    // just checks color dominance instead of absolute brightness.
    int reds = 0, blue_greens = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 30 && r > g + 10 && r > b + 10) ++reds;
        if (b > 30 && b > r + 10) ++blue_greens;
    }
    EXPECT_GT(reds, 20)        << "expected red checker squares visible";
    EXPECT_GT(blue_greens, 20) << "expected blue-green checker squares visible";
}
