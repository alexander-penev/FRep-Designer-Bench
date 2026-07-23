// tests/test_glsl_emitter.cpp
//
// Unit tests for the FRepNode → GLSL translator. Each test verifies that
// the emitted shader text:
//   (a) parses cleanly through glslangValidator
//   (b) produces a non-blank GPU render
// Tests skip when Vulkan / glslang are unavailable.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
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

bool glslang_available() {
    return !gpu::find_glslang().empty();
}

gpu::ShaderPush sane_push(int w, int h) {
    float cam[3]   = {0, 1, 4};
    float tgt[3]   = {0, 0, 0};
    float light[3] = {4, 6, 4};
    return frep::gpu::build_push_simple(cam, tgt, light, w, h);
}

int lit_pixels(const std::vector<std::uint8_t>& px, int W, int H) {
    int c = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 25 && (r > b * 0.85 || g > b * 0.85)) ++c;
    }
    return c;
}

} // anon

// ── Emit-only tests (no Vulkan needed) ─────────────────────────────────────

TEST(GlslEmitter, EmitsForBasicScene) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_EQ(r->object_count, 1);
    EXPECT_NE(r->source.find("#version 450"), std::string::npos);
    EXPECT_NE(r->source.find("void main()"), std::string::npos);
    EXPECT_NE(r->source.find("scene_sdf"), std::string::npos);
}

TEST(GlslEmitter, EmitsForCsg) {
    SceneGraph s;
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b");
    s.add_object(std::make_shared<DifferenceNode>(a, b, "diff"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("max("), std::string::npos);  // Difference uses max
}

TEST(GlslEmitter, EmitsForDeformations) {
    SceneGraph s;
    auto box = std::make_shared<BoxNode>(0.5f, 1.0f, 0.5f, "b");
    s.add_object(std::make_shared<TwistYNode>(box, 1.5f, "tw"));
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_NE(r->source.find("cos("), std::string::npos);
    EXPECT_NE(r->source.find("sin("), std::string::npos);
}

// Analytic normals via dual-number AD. A scene of dual-supported nodes
// (sphere, box, plane, CSG, transforms, smooth-union) must emit the Dual
// machinery + scene_sdf_grad and use it for the normal — matching the CPU's
// forward-mode AD instead of central differences. Must still compile.
TEST(GlslEmitter, AnalyticNormalForDualScene) {
    SceneGraph s;
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<SphereNode>(1.0f, "b");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.3f, "blob"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    const std::string& src = emitted->source;

    EXPECT_NE(src.find("struct Dual"), std::string::npos);
    EXPECT_NE(src.find("Dual scene_sdf_grad"), std::string::npos);
    // The normal must use the analytic gradient, not central differences.
    EXPECT_NE(src.find("scene_sdf_grad(p).g"), std::string::npos);
    EXPECT_EQ(src.find("scene_sdf_v(p + vec3(h,0,0))"), std::string::npos)
        << "central-difference normal leaked into a dual-capable scene";

    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";
    auto spv = gpu::compile_glsl_to_spv(src);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

// A node with no dual emitter (e.g. BendXY) must make the whole scene fall
// back to central-difference normals — analytic AD is abandoned globally so
// correctness is preserved. Must still compile.
TEST(GlslEmitter, FallsBackToCentralDiffWithoutDual) {
    SceneGraph s;
    auto box = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "bx");
    s.add_object(std::make_shared<BendXYNode>(box, 0.5f, "bend"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    const std::string& src = emitted->source;

    EXPECT_NE(src.find("scene_sdf_v(p + vec3(h,0,0))"), std::string::npos)
        << "expected central-difference fallback for a non-dual node";
    EXPECT_EQ(src.find("Dual scene_sdf_grad"), std::string::npos);

    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";
    auto spv = gpu::compile_glsl_to_spv(src);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

TEST(GlslEmitter, EmptySceneStillWellFormed) {
    SceneGraph s;
    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->object_count, 0);
    EXPECT_NE(r->source.find("void main()"), std::string::npos);
}

// ── Compile tests — require glslang ────────────────────────────────────────

TEST(GlslEmitter, CompilesViaGlslang) {
    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";

    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    auto spv = gpu::compile_glsl_to_spv(emitted->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

// The temporal-denoise accumulation path changes the storage image from
// `writeonly` to read-write (so the shader can imageLoad the previous
// frame) and adds an accum_blend mix at the final store. Verify the
// emitted GLSL contains that machinery and still compiles — a guard
// against the read-write image or the blend expression regressing into
// something glslang rejects (e.g. imageLoad on a writeonly image).
TEST(GlslEmitter, EmitsAccumulationBlendAndCompiles) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    const std::string& src = emitted->source;

    // The output image must NOT be writeonly (imageLoad needs read access).
    EXPECT_EQ(src.find("writeonly image2D out_image"), std::string::npos)
        << "out_image is still writeonly — imageLoad for accumulation "
           "would be invalid.";
    // The push constant and the blend must be present.
    EXPECT_NE(src.find("accum_blend"), std::string::npos);
    EXPECT_NE(src.find("imageLoad(out_image"), std::string::npos);

    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";
    auto spv = gpu::compile_glsl_to_spv(src);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

// The primary and reflection raymarch loops must step by a fraction of
// the reported SDF distance (safety_factor), not the full distance.
// CSG scenes are not true Euclidean SDFs, so a full step overshoots
// near union/intersection seams and grazing silhouettes, producing dark
// speckle along edges. This mirrors the CPU JIT march and keeps the two
// renderers in agreement. Guards against a regression back to `t += d`.
TEST(GlslEmitter, RaymarchUsesSafetyFactor) {
    SceneGraph s;
    // A CSG union — the case where full-distance stepping misbehaves.
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "b");
    s.add_object(std::make_shared<UnionNode>(a, b, "u"));

    TracerConfig cfg;
    cfg.max_bounces = 1;   // also emit the reflection march
    auto emitted = gpu::GlslEmitter::emit(s, cfg);
    ASSERT_TRUE(emitted.has_value());
    const std::string& src = emitted->source;

    // No bare full-distance step should remain in either march loop.
    EXPECT_EQ(src.find("t += d;"),  std::string::npos)
        << "primary raymarch still uses a full-distance step (t += d)";
    EXPECT_EQ(src.find("rt += d;"), std::string::npos)
        << "reflection raymarch still uses a full-distance step (rt += d)";
    // The safety-factor-scaled step should be present instead.
    EXPECT_NE(src.find("t += d *"),  std::string::npos);

    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";
    auto spv = gpu::compile_glsl_to_spv(src);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

// Grazing-ray rescue: a silhouette ray can exhaust max_steps just above
// epsilon, and treating that as a miss paints a bright sky-coloured
// fringe along object outlines against dark backgrounds. The emitter
// must track the last sampled distance and reclassify a near-surface
// stop as a hit. Guards against the rescue being dropped.
TEST(GlslEmitter, EmitsGrazingRescue) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    const std::string& src = emitted->source;

    // The last-distance tracker and the near-surface reclassification
    // must both be present.
    EXPECT_NE(src.find("last_d"), std::string::npos)
        << "grazing rescue dropped: no last_d tracking";
    EXPECT_NE(src.find("!hit"), std::string::npos)
        << "grazing rescue dropped: no post-loop hit reclassification";

    if (!glslang_available()) GTEST_SKIP() << "glslangValidator not on PATH";
    auto spv = gpu::compile_glsl_to_spv(src);
    ASSERT_TRUE(spv.has_value()) << spv.error();
}

// Adaptive raymarch step: a single primitive is a true SDF and should
// emit a full step (no 0.85 factor); CSG / multi-object scenes must emit
// the reduced step. Guards the perf optimisation against regressing.
TEST(GlslEmitter, AdaptiveSafetyFactor) {
    {
        SceneGraph s;
        s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
        auto e = gpu::GlslEmitter::emit(s);
        ASSERT_TRUE(e.has_value());
        EXPECT_EQ(e->source.find("* 0.85"), std::string::npos)
            << "single primitive should march at full step";
    }
    {
        SceneGraph s;
        auto a = std::make_shared<SphereNode>(1.0f, "a");
        auto b = std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "b");
        s.add_object(std::make_shared<UnionNode>(a, b, "u"));
        auto e = gpu::GlslEmitter::emit(s);
        ASSERT_TRUE(e.has_value());
        EXPECT_NE(e->source.find("* 0.85"), std::string::npos)
            << "CSG scene must march at the reduced safety step";
    }
    {
        SceneGraph s;
        s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
        s.add_object(std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "b"));
        auto e = gpu::GlslEmitter::emit(s);
        ASSERT_TRUE(e.has_value());
        EXPECT_NE(e->source.find("* 0.85"), std::string::npos)
            << "multi-object scene must march at the reduced safety step";
    }
}

TEST(GlslEmitter, RendersBasicSceneOnGpu) {
    if (!glslang_available())                GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())        GTEST_SKIP() << "no Vulkan";

    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    auto spv = gpu::compile_glsl_to_spv(emitted->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 128, H = 96;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());
    EXPECT_EQ(px.size(), static_cast<std::size_t>(W * H * 4));
    EXPECT_GT(lit_pixels(px, W, H), 100);
}

TEST(GlslEmitter, RendersCsgOnGpu) {
    if (!glslang_available())                GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())        GTEST_SKIP() << "no Vulkan";

    SceneGraph s;
    auto a = std::make_shared<SphereNode>(0.95f, "a");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b"),
        0.55f, 0.45f, 0.45f, "bt");
    s.add_object(std::make_shared<DifferenceNode>(a, b, "diff"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    auto spv = gpu::compile_glsl_to_spv(emitted->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 96, H = 72;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());
    EXPECT_GT(lit_pixels(px, W, H), 50);
}

TEST(GlslEmitter, RendersDeformationsOnGpu) {
    if (!glslang_available())                GTEST_SKIP() << "no glslang";
    if (!gpu::VulkanCtx::available())        GTEST_SKIP() << "no Vulkan";

    SceneGraph s;
    auto box = std::make_shared<BoxNode>(0.4f, 1.0f, 0.4f, "b");
    s.add_object(std::make_shared<TwistYNode>(box, 1.5f, "tw"));

    auto emitted = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(emitted.has_value());
    auto spv = gpu::compile_glsl_to_spv(emitted->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx_or = gpu::VulkanCtx::create(*spv);
    ASSERT_TRUE(ctx_or.has_value()) << ctx_or.error();

    int W = 96, H = 96;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx_or).render(sane_push(W, H), px).has_value());
    EXPECT_GT(lit_pixels(px, W, H), 50);
}
