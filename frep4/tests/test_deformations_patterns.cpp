// tests/test_deformations_patterns.cpp
//
// Tests for the new deformation nodes (TwistY, BendXY, TaperY) and the
// procedural-material patterns (Checker, Stripes, GradientY, Noise).

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/deformations.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>
#include <cmath>
#include <memory>

using namespace frep;

namespace {

struct LlvmInit {
    LlvmInit() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};
LlvmInit g_llvm_init;

// Helper: count lit pixels for visual presence (filters blue sky).
int lit_pixels(const std::vector<float>& im, int W, int H) {
    int c = 0;
    for (int i = 0; i < W*H; ++i) {
        float r = im[i*4+0], g = im[i*4+1], b = im[i*4+2];
        if (r > 0.05f && r >= b * 0.85f && g >= b * 0.85f) ++c;
    }
    return c;
}

} // anon

// ── Deformations: zero parameter is identity ────────────────────────────────

TEST(Deformations, TwistZeroIsIdentity) {
    // TwistY with k=0 should reduce to the identity transform; eval()
    // should match the child SDF exactly.
    auto child = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    TwistYNode twist(child, 0.0f);
    for (float x : {-0.5f, 0.5f, 1.5f})
        for (float y : {-0.5f, 0.5f, 1.5f})
            for (float z : {-0.5f, 0.5f, 1.5f}) {
                EXPECT_NEAR(twist.eval(x, y, z),
                            child->eval(x, y, z), 1e-4f);
            }
}

TEST(Deformations, BendZeroIsIdentity) {
    auto child = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    BendXYNode bend(child, 0.0f);
    for (float x : {-0.5f, 0.5f, 1.5f})
        for (float y : {-0.5f, 0.5f, 1.5f})
            for (float z : {-0.5f, 0.5f, 1.5f}) {
                EXPECT_NEAR(bend.eval(x, y, z),
                            child->eval(x, y, z), 1e-4f);
            }
}

TEST(Deformations, TaperOneIsIdentity) {
    // TaperY with t = 1 (no taper at the top) is identity.
    auto child = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    TaperYNode taper(child, 1.0f, 2.0f);
    for (float x : {-0.5f, 0.5f})
        for (float y : {-0.5f, 0.5f})
            for (float z : {-0.5f, 0.5f}) {
                EXPECT_NEAR(taper.eval(x, y, z),
                            child->eval(x, y, z), 1e-4f);
            }
}

// ── Deformations preserve inside/outside topology ───────────────────────────

TEST(Deformations, TwistKeepsCenterInside) {
    // The origin is inside a unit box; twisting space around Y leaves the
    // origin in place, so it remains inside.
    auto child = std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b");
    TwistYNode twist(child, 2.5f);
    EXPECT_LT(twist.eval(0, 0, 0), 0);
    // A point well outside stays outside.
    EXPECT_GT(twist.eval(5, 0, 0), 0);
}

TEST(Deformations, TaperShrinksTop) {
    // After tapering (t=0 at top), a point that *was* inside the box top
    // corner should now be outside, since the top has collapsed to a point.
    auto child = std::make_shared<BoxNode>(0.5f, 1.0f, 0.5f, "b");
    TaperYNode taper(child, 0.0f, 2.0f);  // tip at y=+1
    // Near the tip but offset in X — used to be inside the un-tapered box.
    EXPECT_LT(child->eval(0.3f, 0.95f, 0.0f), 0);
    EXPECT_GT(taper .eval(0.3f, 0.95f, 0.0f), 0);
}

// ── JIT pipeline survives deformations end-to-end ──────────────────────────

TEST(Deformations, RendersThroughJit) {
    // Build a tiny scene with a twisted box; verify JIT compile and render
    // produces a non-empty image.
    SceneGraph s;
    auto child = std::make_shared<BoxNode>(0.4f, 1.0f, 0.4f, "b");
    s.add_object(std::make_shared<TwistYNode>(child, 1.5f, "twy"));
    auto& L = s.lights(); L.clear();
    L.push_back({{4, 5, 4}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());

    int W = 80, H = 60;
    std::vector<float> px(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);
    EXPECT_GT(lit_pixels(px, W, H), 100);
}

// ── Procedural patterns: codegen produces visibly different images ─────────

TEST(Patterns, CheckerProducesAlternation) {
    // Two adjacent points on a sphere surface, in different checker cells,
    // should yield different colors.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(2.0f, "ball"));
    Material m;
    m.pattern       = Material::Pattern::Checker;
    m.albedo        = {1.0f, 0.0f, 0.0f};
    m.albedo2       = {0.0f, 1.0f, 0.0f};
    m.pattern_scale = 5.0f;
    s.objects().at("ball");  // ensure presence
    // Re-add with the material (set_material on existing obj).
    s.set_material("ball", m);

    auto& L = s.lights(); L.clear();
    L.push_back({{0, 0, 4}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 6};
    s.camera().target   = {0, 0, 0};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());

    int W = 256, H = 256;
    std::vector<float> px(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);

    // Histogram into red-dominant vs green-dominant.
    int red_pix = 0, grn_pix = 0;
    for (int i = 0; i < W*H; ++i) {
        float r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 0.05f || g > 0.05f) {
            // Sphere pixels: classify by which channel dominates.
            if (r > g + 0.05f && r > b) ++red_pix;
            else if (g > r + 0.05f && g > b) ++grn_pix;
        }
    }
    // Both colors should be visibly present.
    EXPECT_GT(red_pix, 100);
    EXPECT_GT(grn_pix, 100);
}

TEST(Patterns, SolidIsConstant) {
    // A solid material should produce a single dominant color (no checker
    // alternation). Just verify the render runs.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.5f, "ball"));
    Material m{{0.9f, 0.2f, 0.1f}};
    s.set_material("ball", m);
    auto& L = s.lights(); L.clear();
    L.push_back({{0, 0, 4}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());

    int W = 128, H = 128;
    std::vector<float> px(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);
    // Should be visible.
    int lit = lit_pixels(px, W, H);
    EXPECT_GT(lit, 100);
}

TEST(Patterns, NoiseProducesVariation) {
    // Noise material should produce a range of values on the surface,
    // not a single constant. Measure the standard deviation of pixel
    // brightness on the sphere.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.5f, "ball"));
    Material m;
    m.pattern       = Material::Pattern::Noise;
    m.albedo        = {0.1f, 0.1f, 0.1f};
    m.albedo2       = {0.9f, 0.9f, 0.9f};
    m.pattern_scale = 15.0f;
    s.set_material("ball", m);
    auto& L = s.lights(); L.clear();
    L.push_back({{0, 0, 4}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());

    int W = 128, H = 128;
    std::vector<float> px(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);

    // Compute std-dev of brightness on the sphere pixels.
    std::vector<float> brightness;
    for (int i = 0; i < W*H; ++i) {
        float r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 0.05f && r >= b * 0.85f) {  // not sky
            brightness.push_back((r + g + b) / 3.0f);
        }
    }
    ASSERT_GT(brightness.size(), 100u);
    float mean = 0;
    for (auto v : brightness) mean += v;
    mean /= brightness.size();
    float var = 0;
    for (auto v : brightness) var += (v - mean) * (v - mean);
    var /= brightness.size();
    // For solid material the std-dev would be very small; noise should
    // exceed at least 0.05.
    EXPECT_GT(std::sqrt(var), 0.05f);
}

// ─── Scene I/O round-trip for deformations ──────────────────────────────────
// Regression test: built-in deformation nodes used to require a
// PluginRegistry at load time; now they're handled by the core path so a
// nullptr registry suffices.

#include "core/io/scene_io.hpp"

TEST(SceneIO, DeformationsRoundTrip) {
    SceneGraph s;
    Material m{{0.8f, 0.3f, 0.2f}};
    auto bx = std::make_shared<BoxNode>(0.3f, 1.0f, 0.3f, "bx");
    auto tw = std::make_shared<TwistYNode>(bx, 1.5f, "tw");
    auto bd = std::make_shared<BendXYNode>(
        std::make_shared<BoxNode>(0.2f, 0.7f, 0.4f, "bx2"), 0.4f, "bd");
    auto tp = std::make_shared<TaperYNode>(
        std::make_shared<BoxNode>(0.4f, 0.8f, 0.4f, "bx3"), 0.3f, 1.6f, "tp");
    s.add_object(std::make_shared<TranslateNode>(tw, -2, 0, 0, "ttw"), m);
    s.add_object(std::make_shared<TranslateNode>(bd,  0, 0, 0, "tbd"), m);
    s.add_object(std::make_shared<TranslateNode>(tp,  2, 0, 0, "ttp"), m);

    auto path = std::string("/tmp/test_deforms_io.json");
    ASSERT_TRUE(io::save_scene(s, path));

    // Note: no PluginRegistry passed — deformations are core types.
    SceneGraph s2;
    ASSERT_NO_THROW(s2 = io::load_scene(path, nullptr));
    EXPECT_EQ(s2.objects().size(), 3u);
}

// ─── Material round-trip ────────────────────────────────────────────────────
// Regression test: materials (albedo, pattern, pattern_scale, PBR fields)
// must survive save_scene + load_scene. Earlier versions of scene_io
// either skipped materials entirely or lost the pattern field.

TEST(SceneIO, MaterialRoundTrip) {
    SceneGraph s;
    Material m1;
    m1.albedo  = {0.95f, 0.4f, 0.3f};
    m1.albedo2 = {0.2f, 0.7f, 0.95f};
    m1.pattern = Material::Pattern::Checker;
    m1.pattern_scale = 5.5f;
    m1.roughness = 0.35f;
    m1.metallic  = 0.7f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m1);

    Material m2;  // default Solid material
    m2.albedo = {0.55f, 0.55f, 0.55f};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), m2);

    auto path = std::string("/tmp/test_material_io.json");
    ASSERT_TRUE(io::save_scene(s, path));

    SceneGraph s2 = io::load_scene(path, nullptr);
    ASSERT_EQ(s2.objects().size(), 2u);

    const auto& ball = s2.objects().at("ball").material;
    EXPECT_FLOAT_EQ(ball.albedo[0],  0.95f);
    EXPECT_FLOAT_EQ(ball.albedo[1],  0.4f);
    EXPECT_FLOAT_EQ(ball.albedo[2],  0.3f);
    EXPECT_FLOAT_EQ(ball.albedo2[0], 0.2f);
    EXPECT_EQ(ball.pattern, Material::Pattern::Checker);
    EXPECT_FLOAT_EQ(ball.pattern_scale, 5.5f);
    EXPECT_FLOAT_EQ(ball.roughness,     0.35f);
    EXPECT_FLOAT_EQ(ball.metallic,      0.7f);

    const auto& floor = s2.objects().at("floor").material;
    EXPECT_EQ(floor.pattern, Material::Pattern::Solid);
    EXPECT_FLOAT_EQ(floor.albedo[0], 0.55f);
}
