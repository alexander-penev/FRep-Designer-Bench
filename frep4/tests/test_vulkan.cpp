// tests/test_vulkan.cpp
//
// Tests for the Vulkan GPU compute path. Designed to skip gracefully on
// systems without a Vulkan implementation (and on shared CI runners where
// validation layers may be missing). Each test that needs a live device
// calls GTEST_SKIP() when VulkanCtx::available() returns false.

#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <string>

using namespace frep::gpu;

namespace {

// Resolve the shader file path relative to the build's known layout.
std::string find_shader_path() {
    namespace fs = std::filesystem;
    for (const char* p : {
        "build/shaders/sphere_trace.spv",
        "shaders/sphere_trace.spv",
        "../build/shaders/sphere_trace.spv",
    }) {
        if (fs::exists(p)) return p;
    }
    return {};
}

// Build a sane fixed push struct for a small render.
ShaderPush make_push(int w, int h) {
    float cam[3]   = {0, 1, 4};
    float tgt[3]   = {0, 0, 0};
    float light[3] = {4, 6, 4};
    return frep::gpu::build_push_simple(cam, tgt, light, w, h);
}

} // anon

// ── Availability probe never crashes ────────────────────────────────────────

TEST(Vulkan, Availability) {
    // Just calling it should not throw or crash, regardless of result.
    (void)VulkanCtx::available();
    SUCCEED();
}

// ── Context creation when Vulkan is present ─────────────────────────────────

TEST(Vulkan, ContextCreates) {
    if (!VulkanCtx::available()) GTEST_SKIP() << "no Vulkan device";
    auto spv = find_shader_path();
    ASSERT_FALSE(spv.empty()) << "sphere_trace.spv not found";

    auto ctx_or = VulkanCtx::create(spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();
    auto& ctx = **ctx_or;
    EXPECT_FALSE(ctx.stats().device_name.empty());
    EXPECT_GT(ctx.stats().init_ms, 0.0);
}

// ── End-to-end: render an image and check it isn't blank ───────────────────

TEST(Vulkan, RendersNonBlankImage) {
    if (!VulkanCtx::available()) GTEST_SKIP() << "no Vulkan device";
    auto spv = find_shader_path();
    ASSERT_FALSE(spv.empty());

    auto ctx_or = VulkanCtx::create(spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();
    auto& ctx = **ctx_or;

    int W = 80, H = 60;
    std::vector<std::uint8_t> pixels;
    auto r = ctx.render(make_push(W, H), pixels);
    ASSERT_TRUE(r.has_value()) << r.error();
    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(W * H * 4));

    // Histogram check: not all pixels the same colour; sky + sphere + floor
    // should give us at least three distinct red levels.
    int r_low = 0, r_mid = 0, r_high = 0;
    for (int i = 0; i < W * H; ++i) {
        int r8 = pixels[i*4 + 0];
        if      (r8 < 100) ++r_low;
        else if (r8 < 200) ++r_mid;
        else               ++r_high;
    }
    EXPECT_GT(r_low,  20) << "expected dark pixels (shadow)";
    EXPECT_GT(r_high, 20) << "expected bright pixels (lit sphere/sky)";
}

// ── Sphere should be visibly red in the centre area ────────────────────────

TEST(Vulkan, SphereAlbedoVisible) {
    if (!VulkanCtx::available()) GTEST_SKIP() << "no Vulkan device";
    auto spv = find_shader_path();
    ASSERT_FALSE(spv.empty());
    auto ctx_or = VulkanCtx::create(spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();
    auto& ctx = **ctx_or;

    int W = 128, H = 96;
    std::vector<std::uint8_t> pixels;
    ASSERT_TRUE(ctx.render(make_push(W, H), pixels).has_value());

    // Count pixels where red dominates over blue (i.e. the sphere region).
    int red_dom = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = pixels[i*4 + 0], g = pixels[i*4 + 1], b = pixels[i*4 + 2];
        if (r > 100 && r > b + 30 && r > g + 30) ++red_dom;
    }
    EXPECT_GT(red_dom, 200)
        << "expected the red sphere to occupy a few hundred pixels";
}

// ── Resizing between renders should not crash ──────────────────────────────

TEST(Vulkan, ReusesContextAcrossSizes) {
    if (!VulkanCtx::available()) GTEST_SKIP() << "no Vulkan device";
    auto spv = find_shader_path();
    ASSERT_FALSE(spv.empty());
    auto ctx_or = VulkanCtx::create(spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();
    auto& ctx = **ctx_or;

    std::vector<std::uint8_t> px;
    for (auto [w, h] : std::initializer_list<std::pair<int,int>>{
            {32, 32}, {64, 48}, {32, 32}}) {
        EXPECT_TRUE(ctx.render(make_push(w, h), px).has_value());
        EXPECT_EQ(px.size(), static_cast<std::size_t>(w * h * 4));
    }
}
