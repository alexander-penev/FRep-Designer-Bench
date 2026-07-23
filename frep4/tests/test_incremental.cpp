// tests/test_incremental.cpp
//
// Tests for Per-parameter incremental compilation. Verifies:
//   - The SceneCodegen records all primitive/transform parameters in the
//     binding table when incremental_params=true.
//   - In Constant mode the table is empty (no slots allocated).
//   - An Incremental-compiled render with the buffer seeded to defaults
//     matches a Constant-mode render bit-exactly (within sky-filter tolerance).
//   - Updating the buffer changes the output without recompiling.
//   - The Auto policy switches to Incremental after the threshold.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/incremental_params.hpp"
#include "core/compiler/compile_mode_policy.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>
#include <memory>
#include <thread>

using namespace frep;

namespace {

struct LlvmInit {
    LlvmInit() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};
LlvmInit g_llvm_init;

// Helper: render a scene and return the lit-pixel count (filters blue sky).
int lit_pixels(const std::vector<float>& im, int W, int H) {
    int c = 0;
    for (int i = 0; i < W*H; ++i) {
        float r = im[i*4+0], g = im[i*4+1], b = im[i*4+2];
        if (r > 0.05f && r >= b * 0.85f && g >= b * 0.85f) ++c;
    }
    return c;
}

} // namespace

// ── Binding table ───────────────────────────────────────────────────────────

TEST(Incremental, ConstantModeHasNoBindings) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;  // default — Constant mode
    cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    EXPECT_TRUE(cg.param_bindings().empty());
    EXPECT_EQ(cg.param_slot_count(), 0);
}

TEST(Incremental, IncrementalModeBindsAllPrimitiveParams) {
    SceneGraph s;
    // 1 sphere (1 param), 1 box (3 params), 1 translate (3 params) wrapping
    // the box. That's 7 unique (node_id, param_name) slots.
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "cube"),
        2.0f, 0, 0, "cube_t"));

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;
    cfg.enable_shadows = false; cfg.enable_ao = false;
    cfg.incremental_params = true;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);

    EXPECT_EQ(cg.param_slot_count(), 7);

    // Each binding has a distinct slot in [0, 7).
    std::set<int> slots;
    for (const auto& b : cg.param_bindings()) {
        EXPECT_GE(b.slot, 0);
        EXPECT_LT(b.slot, 7);
        slots.insert(b.slot);
    }
    EXPECT_EQ(slots.size(), 7u);

    // The defaults should match the scene values.
    auto find = [&](const std::string& id, const std::string& p) -> float {
        for (const auto& b : cg.param_bindings())
            if (b.node_id == id && b.param_name == p) return b.default_value;
        return -999.0f;
    };
    EXPECT_FLOAT_EQ(find("ball", "r"),  1.0f);
    EXPECT_FLOAT_EQ(find("cube", "hx"), 0.5f);
    EXPECT_FLOAT_EQ(find("cube_t", "tx"), 2.0f);
}

// ── Equivalence: Incremental + defaults  ≡  Constant ────────────────────────

TEST(Incremental, MatchesConstantWithSeededDefaults) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.8f, "ball"));
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 5, 5}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    int W = 100, H = 75;
    RenderParams rp; rp.width = W; rp.height = H;

    auto run = [&](bool inc, std::vector<float>& out) {
        auto ctx = std::make_unique<llvm::LLVMContext>();
        TracerConfig cfg;
        cfg.enable_shadows = false; cfg.enable_ao = false;
        cfg.incremental_params = inc;
        SceneCodegen cg(*ctx, cfg);
        cg.emit_render_tile(s);
        IncrementalParams ip(cg);
        auto mod = cg.take_module();
        JitEngine jit;
        auto fn_or = jit.load(std::move(mod), std::move(ctx));
        ASSERT_TRUE(fn_or.has_value());
        out.assign(W*H*4, 0.0f);
        TileScheduler::render(*fn_or, out.data(), s.camera(), rp,
                              inc ? ip.buffer() : nullptr);
    };

    std::vector<float> img_const, img_inc;
    run(false, img_const);
    run(true,  img_inc);

    int lc = lit_pixels(img_const, W, H);
    int li = lit_pixels(img_inc,   W, H);
    EXPECT_GT(lc, 100);            // there should be visible pixels
    EXPECT_EQ(lc, li);             // and they should match exactly
}

// ── Edit the buffer, see a different image, no recompile ───────────────────

TEST(Incremental, EditBufferWithoutRecompileChangesOutput) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 5, 5}, {1, 1, 1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;
    cfg.enable_shadows = false; cfg.enable_ao = false;
    cfg.incremental_params = true;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    IncrementalParams ip(cg);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());
    auto fn = *fn_or;

    int W = 100, H = 75;
    RenderParams rp; rp.width = W; rp.height = H;
    std::vector<float> img_big(W*H*4), img_small(W*H*4);

    // Render with default r=1.0
    TileScheduler::render(fn, img_big.data(), s.camera(), rp, ip.buffer());
    int lit_big = lit_pixels(img_big, W, H);

    // Shrink the sphere — buffer edit only, no recompile.
    ASSERT_TRUE(ip.set("ball", "r", 0.4f));
    TileScheduler::render(fn, img_small.data(), s.camera(), rp, ip.buffer());
    int lit_small = lit_pixels(img_small, W, H);

    // The smaller sphere must cover visibly fewer pixels.
    EXPECT_LT(lit_small, lit_big / 2);
}

// ── IncrementalParams API ───────────────────────────────────────────────────

TEST(IncrementalParams, GetSetHasRoundTrip) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.5f, "ball"));
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.incremental_params = true;
    cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    IncrementalParams ip(cg);

    EXPECT_TRUE(ip.has("ball", "r"));
    EXPECT_FLOAT_EQ(ip.get("ball", "r"), 1.5f);

    EXPECT_TRUE(ip.set("ball", "r", 0.3f));
    EXPECT_FLOAT_EQ(ip.get("ball", "r"), 0.3f);

    // Non-existent parameter — set should report failure and not crash.
    EXPECT_FALSE(ip.set("nope", "x", 1.0f));
    EXPECT_FALSE(ip.has("nope", "x"));
}

// ── Auto policy ─────────────────────────────────────────────────────────────

TEST(CompileModePolicy, AutoSwitchesAfterThreshold) {
    CompileModePolicy p(TracerConfig::CompileMode::Auto);
    p.set_threshold(3);
    EXPECT_FALSE(p.use_incremental());  // start in Constant

    p.note_recompile();
    p.note_recompile();
    EXPECT_FALSE(p.use_incremental());  // 2 < 3

    p.note_recompile();
    EXPECT_TRUE(p.use_incremental());   // 3 triggers the latch
    p.note_recompile();
    EXPECT_TRUE(p.use_incremental());   // stays latched
}

TEST(CompileModePolicy, ConstantNeverSwitches) {
    CompileModePolicy p(TracerConfig::CompileMode::Constant);
    for (int i = 0; i < 100; ++i) p.note_recompile();
    EXPECT_FALSE(p.use_incremental());
}

TEST(CompileModePolicy, IncrementalAlwaysOn) {
    CompileModePolicy p(TracerConfig::CompileMode::Incremental);
    EXPECT_TRUE(p.use_incremental());
    p.note_recompile();
    EXPECT_TRUE(p.use_incremental());
}

TEST(CompileModePolicy, ResetClearsLatch) {
    CompileModePolicy p(TracerConfig::CompileMode::Auto);
    p.set_threshold(2);
    p.note_recompile();
    p.note_recompile();
    EXPECT_TRUE(p.use_incremental());

    p.reset_auto_state();
    EXPECT_FALSE(p.use_incremental());
    p.note_recompile();
    EXPECT_FALSE(p.use_incremental());
    p.note_recompile();
    EXPECT_TRUE(p.use_incremental());
}
