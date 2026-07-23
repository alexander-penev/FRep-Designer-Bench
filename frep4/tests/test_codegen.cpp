// tests/test_codegen.cpp
//
// End-to-end: FRepNode → LLVM IR → JIT → numeric result.
// Google Test, C++26 (we use templated lambdas, deducing this where it fits).

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/picker.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/tracer/tile_scheduler.hpp"
#include "tests/test_support.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>

#include <cmath>
#include <memory>
#include <vector>

using namespace frep;

namespace {

// LLVM global initialization — once per executable.
class LLVMInitEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};
[[maybe_unused]] auto* g_llvm_env =
    ::testing::AddGlobalTestEnvironment(new LLVMInitEnvironment);

using test::SdfFn;
using test::jit_pool;

// Compiles FRepNode → JIT → function pointer.
SdfFn jit_sdf(const FRepNode& root) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx);
    cg.emit_scene_sdf(root);

    auto& mod = *cg.module();
    auto* sdf_fn = mod.getFunction("scene_sdf");
    if (!sdf_fn) return nullptr;

    // A wrapper named "render_tile" for compatibility with JitEngine::load lookup.
    auto* fty = llvm::FunctionType::get(
        llvm::Type::getFloatTy(*ctx),
        {llvm::Type::getFloatTy(*ctx),
         llvm::Type::getFloatTy(*ctx),
         llvm::Type::getFloatTy(*ctx)}, false);
    auto* wrapper = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "render_tile", &mod);
    auto* bb = llvm::BasicBlock::Create(*ctx, "entry", wrapper);
    llvm::IRBuilder<> b(bb);
    auto it = wrapper->arg_begin();
    auto* x = &*it++;
    auto* y = &*it++;
    auto* z = &*it++;
    b.CreateRet(b.CreateCall(sdf_fn, {x, y, z}));

    auto mod_ptr = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto& jit = *jit_pool().back();
    auto fn_or = jit.load(std::move(mod_ptr), std::move(ctx));
    if (!fn_or) return nullptr;
    return reinterpret_cast<SdfFn>(*fn_or);
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// JIT tests
// ═════════════════════════════════════════════════════════════════════════════

// Compiles FRepNode → JIT at a given optimisation level → function ptr.
// Used to verify that lowering the opt level (the lever for taming JIT
// compile time on large scenes) doesn't change the numerical result.
static SdfFn jit_sdf_opt(const FRepNode& root, llvm::OptimizationLevel lvl) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx);
    cg.emit_scene_sdf(root);
    auto& mod = *cg.module();
    auto* sdf_fn = mod.getFunction("scene_sdf");
    if (!sdf_fn) return nullptr;
    auto* fty = llvm::FunctionType::get(
        llvm::Type::getFloatTy(*ctx),
        {llvm::Type::getFloatTy(*ctx), llvm::Type::getFloatTy(*ctx),
         llvm::Type::getFloatTy(*ctx)}, false);
    auto* wrapper = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "render_tile", &mod);
    auto* bb = llvm::BasicBlock::Create(*ctx, "entry", wrapper);
    llvm::IRBuilder<> b(bb);
    auto it = wrapper->arg_begin();
    auto* x = &*it++; auto* y = &*it++; auto* z = &*it++;
    b.CreateRet(b.CreateCall(sdf_fn, {x, y, z}));
    auto mod_ptr = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto& jit = *jit_pool().back();
    jit.set_opt_level(lvl);
    auto fn_or = jit.load(std::move(mod_ptr), std::move(ctx));
    if (!fn_or) return nullptr;
    return reinterpret_cast<SdfFn>(*fn_or);
}

// All optimisation levels must produce the same SDF result — only compile
// time and codegen quality change, never the math. Guards the opt-level
// lever (JitEngine::set_opt_level) used to control compile cost.
TEST(JitCodegen, OptLevelsAgree) {
    auto u = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b"), 1.0f, 0, 0, "t"),
        "u");
    const llvm::OptimizationLevel lvls[] = {
        llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
        llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3,
    };
    const std::array<std::array<float, 3>, 4> pts = {{
        {0, 0, 0}, {2, 0, 0}, {1.5f, 0.5f, 0}, {-1, 0.3f, 0.2f}}};
    std::array<float, 4> ref{};
    bool have_ref = false;
    for (auto lvl : lvls) {
        SdfFn fn = jit_sdf_opt(*u, lvl);
        ASSERT_NE(fn, nullptr);
        for (std::size_t i = 0; i < pts.size(); ++i) {
            float v = fn(pts[i][0], pts[i][1], pts[i][2]);
            if (!have_ref) ref[i] = v;
            else EXPECT_NEAR(v, ref[i], 1e-4f)
                << "opt level disagreed at point " << i;
        }
        have_ref = true;
    }
}

// The per-object-function split (emit_scene_sdf_split) must compute the
// same distance field as the inlined emit_scene_sdf — it only changes how
// the IR is structured (N small functions + min-fold vs one inlined tree),
// never the result. Guards the function-split path explored as the fix for
// large-scene compile time.
TEST(JitCodegen, SplitSceneSdfMatchesInline) {
    auto make_objs = [] {
        std::vector<FRepNode::Ptr> v;
        v.push_back(std::make_shared<SphereNode>(1.0f, "a"));
        v.push_back(std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b"), 1.5f, 0, 0, "tb"));
        v.push_back(std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.7f, "c"), -1.0f, 0.5f, 0, "tc"));
        return v;
    };

    // Inline scene_sdf.
    auto ctx1 = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg1(*ctx1);
    {
        auto objs = make_objs();
        // union_all is internal; build a manual left-fold union for inline.
        FRepNode::Ptr root = objs[0];
        for (std::size_t i = 1; i < objs.size(); ++i)
            root = std::make_shared<UnionNode>(root, objs[i], "u" + std::to_string(i));
        cg1.emit_scene_sdf(*root);
    }
    SdfFn inl = nullptr;
    {
        auto& mod = *cg1.module();
        auto* fty = llvm::FunctionType::get(llvm::Type::getFloatTy(*ctx1),
            {llvm::Type::getFloatTy(*ctx1), llvm::Type::getFloatTy(*ctx1),
             llvm::Type::getFloatTy(*ctx1)}, false);
        auto* w = llvm::Function::Create(fty, llvm::Function::ExternalLinkage,
                                         "render_tile", &mod);
        auto* bb = llvm::BasicBlock::Create(*ctx1, "e", w);
        llvm::IRBuilder<> b(bb);
        auto it = w->arg_begin();
        auto* x = &*it++; auto* y = &*it++; auto* z = &*it++;
        auto* np = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(*ctx1));
        b.CreateRet(b.CreateCall(mod.getFunction("scene_sdf"), {x, y, z, np}));
        auto m = cg1.take_module();
        jit_pool().emplace_back(std::make_unique<JitEngine>());
        auto fn = jit_pool().back()->load(std::move(m), std::move(ctx1));
        ASSERT_TRUE(fn.has_value());
        inl = reinterpret_cast<SdfFn>(*fn);
    }

    // Split scene_sdf.
    auto ctx2 = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg2(*ctx2);
    cg2.emit_scene_sdf_split(make_objs());
    SdfFn spl = nullptr;
    {
        auto& mod = *cg2.module();
        auto* fty = llvm::FunctionType::get(llvm::Type::getFloatTy(*ctx2),
            {llvm::Type::getFloatTy(*ctx2), llvm::Type::getFloatTy(*ctx2),
             llvm::Type::getFloatTy(*ctx2)}, false);
        auto* w = llvm::Function::Create(fty, llvm::Function::ExternalLinkage,
                                         "render_tile", &mod);
        auto* bb = llvm::BasicBlock::Create(*ctx2, "e", w);
        llvm::IRBuilder<> b(bb);
        auto it = w->arg_begin();
        auto* x = &*it++; auto* y = &*it++; auto* z = &*it++;
        auto* np = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(*ctx2));
        b.CreateRet(b.CreateCall(mod.getFunction("scene_sdf"), {x, y, z, np}));
        auto m = cg2.take_module();
        jit_pool().emplace_back(std::make_unique<JitEngine>());
        auto fn = jit_pool().back()->load(std::move(m), std::move(ctx2));
        ASSERT_TRUE(fn.has_value());
        spl = reinterpret_cast<SdfFn>(*fn);
    }

    for (auto [x, y, z] : std::vector<std::array<float, 3>>{
            {{0, 0, 0}}, {{2, 0, 0}}, {{1.5f, 0.3f, 0}},
            {{-1, 0.5f, 0}}, {{0.5f, -0.5f, 0.5f}}, {{3, 3, 3}}}) {
        EXPECT_NEAR(inl(x, y, z), spl(x, y, z), 1e-4f)
            << "split disagreed with inline at (" << x << "," << y << "," << z << ")";
    }
}

// The build-time guarded SDF (emit_scene_sdf_guarded) must also match the
// inlined min() — the AABB guard only skips objects that can't be nearer,
// so the result is identical. Covers spread + overlapping + a far point.
TEST(JitCodegen, GuardedSceneSdfMatchesInline) {
    auto make_objs = [] {
        std::vector<FRepNode::Ptr> v;
        v.push_back(std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.8f, "a"), -2.0f, 0, 0, "ta"));
        v.push_back(std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b"), 2.0f, 0, 0, "tb"));
        v.push_back(std::make_shared<SphereNode>(0.6f, "c"));   // at origin
        return v;
    };

    auto jit_named = [&](SceneCodegen& cg, std::unique_ptr<llvm::LLVMContext> ctx) -> SdfFn {
        auto& mod = *cg.module();
        auto* fty = llvm::FunctionType::get(llvm::Type::getFloatTy(*ctx),
            {llvm::Type::getFloatTy(*ctx), llvm::Type::getFloatTy(*ctx),
             llvm::Type::getFloatTy(*ctx)}, false);
        auto* w = llvm::Function::Create(fty, llvm::Function::ExternalLinkage,
                                         "render_tile", &mod);
        auto* bb = llvm::BasicBlock::Create(*ctx, "e", w);
        llvm::IRBuilder<> b(bb);
        auto it = w->arg_begin();
        auto* x = &*it++; auto* y = &*it++; auto* z = &*it++;
        auto* np = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(*ctx));
        b.CreateRet(b.CreateCall(mod.getFunction("scene_sdf"), {x, y, z, np}));
        auto m = cg.take_module();
        jit_pool().emplace_back(std::make_unique<JitEngine>());
        auto fn = jit_pool().back()->load(std::move(m), std::move(ctx));
        return fn.has_value() ? reinterpret_cast<SdfFn>(*fn) : nullptr;
    };

    // Inline (manual left-fold union).
    auto ctx1 = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg1(*ctx1);
    {
        auto objs = make_objs();
        FRepNode::Ptr root = objs[0];
        for (std::size_t i = 1; i < objs.size(); ++i)
            root = std::make_shared<UnionNode>(root, objs[i], "u" + std::to_string(i));
        cg1.emit_scene_sdf(*root);
    }
    SdfFn inl = jit_named(cg1, std::move(ctx1));
    ASSERT_NE(inl, nullptr);

    // Guarded.
    auto ctx2 = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg2(*ctx2);
    cg2.emit_scene_sdf_guarded(make_objs());
    SdfFn grd = jit_named(cg2, std::move(ctx2));
    ASSERT_NE(grd, nullptr);

    for (auto [x, y, z] : std::vector<std::array<float, 3>>{
            {{0, 0, 0}}, {{2, 0, 0}}, {{-2, 0, 0}}, {{1.5f, 0.3f, 0}},
            {{-1, 0.5f, 0}}, {{10, 10, 10}}, {{0.5f, -0.5f, 0.5f}}}) {
        EXPECT_NEAR(inl(x, y, z), grd(x, y, z), 1e-4f)
            << "guarded disagreed with inline at (" << x << "," << y << "," << z << ")";
    }
}

TEST(JitCodegen, SphereTrueSdf) {
    SphereNode s(2.0f, "s");
    auto fn = jit_sdf(s);
    ASSERT_NE(fn, nullptr);

    // f(x,y,z) = sqrt(x²+y²+z²) - r
    EXPECT_NEAR(fn(0, 0, 0), -2.0f, 1e-4f);            // center, -r
    EXPECT_NEAR(fn(2, 0, 0),  0.0f, 1e-4f);            // surface
    EXPECT_NEAR(fn(0, 2, 0),  0.0f, 1e-4f);
    EXPECT_NEAR(fn(0, 0, 2),  0.0f, 1e-4f);
    EXPECT_NEAR(fn(3, 0, 0),  1.0f, 1e-4f);            // outside
    EXPECT_NEAR(fn(2, 2, 2),  std::sqrt(12.0f) - 2.0f, 1e-4f);
}

TEST(JitCodegen, BoxSdf) {
    BoxNode b(1.0f, 1.0f, 1.0f, "b");
    auto fn = jit_sdf(b);
    ASSERT_NE(fn, nullptr);

    EXPECT_NEAR(fn(0, 0, 0), -1.0f, 1e-4f);  // center
    EXPECT_NEAR(fn(1, 0, 0),  0.0f, 1e-4f);  // face
    EXPECT_NEAR(fn(2, 0, 0),  1.0f, 1e-4f);  // outside
}

TEST(JitCodegen, TranslateMovesGeometry) {
    auto s = std::make_shared<SphereNode>(1.0f, "s");
    TranslateNode t(s, 3.0f, 0.0f, 0.0f, "t");
    auto fn = jit_sdf(t);
    ASSERT_NE(fn, nullptr);

    EXPECT_NEAR(fn(3, 0, 0), -1.0f, 1e-4f);  // new center
    EXPECT_NEAR(fn(4, 0, 0),  0.0f, 1e-4f);  // new surface
}

TEST(JitCodegen, UnionTakesMin) {
    // F-Rep* convention: Union = min(f1, f2)
    auto a = std::make_shared<SphereNode>(1.0f, "a");                // r=1 at (0,0,0)
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "b"), 5.0f, 0.0f, 0.0f);  // r=1 at (5,0,0)
    UnionNode u(a, b);
    auto fn = jit_sdf(u);
    ASSERT_NE(fn, nullptr);

    EXPECT_NEAR(fn(0, 0, 0),    -1.0f, 1e-4f);  // inside a
    EXPECT_NEAR(fn(5, 0, 0),    -1.0f, 1e-4f);  // inside b
    EXPECT_NEAR(fn(2.5f, 0, 0),  1.5f, 1e-4f);  // between the two → positive
}

TEST(JitCodegen, IntersectionTakesMax) {
    // Sphere(r=2) ∩ Box(1,1,1) → rounded cube
    auto sph = std::make_shared<SphereNode>(2.0f, "s");
    auto box = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    IntersectionNode i(sph, box);
    auto fn = jit_sdf(i);
    ASSERT_NE(fn, nullptr);

    EXPECT_NEAR(fn(0, 0, 0), -1.0f, 1e-4f);  // in both → max(-2,-1) = -1
    EXPECT_NEAR(fn(1, 0, 0),  0.0f, 1e-4f);  // on the box wall
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward-mode AD — the normal must match the analytic solution.
// ─────────────────────────────────────────────────────────────────────────────
namespace {
// JIT-compiles scene_sdf_grad → returns a signature for direct invocation.
// scene_sdf_grad(xv,xd, yv,yd, zv,zd, *out_dot) → float val
using GradFn = float(*)(float,float, float,float, float,float, float*);

GradFn jit_sdf_grad(const FRepNode& root) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx);
    cg.emit_scene_sdf_grad(root);

    auto& mod = *cg.module();
    auto* g = mod.getFunction("scene_sdf_grad");
    if (!g) return nullptr;

    // A "render_tile" wrapper for JitEngine::load lookup.
    auto* f32 = llvm::Type::getFloatTy(*ctx);
    auto* ptr = llvm::PointerType::getUnqual(*ctx);
    auto* fty = llvm::FunctionType::get(f32,
        {f32,f32,f32,f32,f32,f32,ptr}, false);
    auto* w = llvm::Function::Create(fty,
        llvm::Function::ExternalLinkage, "render_tile", &mod);
    auto* bb = llvm::BasicBlock::Create(*ctx, "entry", w);
    llvm::IRBuilder<> b(bb);
    std::vector<llvm::Value*> args;
    for (auto& a : w->args()) args.push_back(&a);
    b.CreateRet(b.CreateCall(g, args));

    auto mp = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto r = jit_pool().back()->load(std::move(mp), std::move(ctx));
    if (!r) return nullptr;
    return reinterpret_cast<GradFn>(*r);
}
} // namespace

TEST(ForwardAD, SphereNormalMatchesAnalytic) {
    // f = sqrt(x²+y²+z²) - r;  ∇f = p / |p|
    auto s = std::make_shared<SphereNode>(2.0f, "s");
    auto gf = jit_sdf_grad(*s);
    ASSERT_NE(gf, nullptr);

    float dot = 0.0f;
    // At (2,0,0): df/dx = 1, df/dy = 0, df/dz = 0
    gf(2,1, 0,0, 0,0, &dot);  EXPECT_NEAR(dot, 1.0f, 1e-4f);
    gf(2,0, 0,1, 0,0, &dot);  EXPECT_NEAR(dot, 0.0f, 1e-4f);
    gf(2,0, 0,0, 0,1, &dot);  EXPECT_NEAR(dot, 0.0f, 1e-4f);

    // At (0,3,0): df/dy = 1
    gf(0,0, 3,1, 0,0, &dot);  EXPECT_NEAR(dot, 1.0f, 1e-4f);

    // At (1,1,1): df/dx = 1/sqrt(3) ~ 0.5774
    gf(1,1, 1,0, 1,0, &dot);
    EXPECT_NEAR(dot, 1.0f / std::sqrt(3.0f), 1e-4f);
}

TEST(ForwardAD, TranslatedSphereNormal) {
    // Translate(Sphere(1), +3 x).  The normal at (4,0,0) → (1,0,0)
    auto s = std::make_shared<SphereNode>(1.0f, "s");
    auto t = std::make_shared<TranslateNode>(s, 3.0f, 0.0f, 0.0f, "t");
    auto gf = jit_sdf_grad(*t);
    ASSERT_NE(gf, nullptr);

    float dot = 0.0f;
    gf(4,1, 0,0, 0,0, &dot);  EXPECT_NEAR(dot, 1.0f, 1e-4f);
    gf(4,0, 0,1, 0,0, &dot);  EXPECT_NEAR(dot, 0.0f, 1e-4f);
}

TEST(ForwardAD, UnionNormalPicksClosest) {
    // Union(Sphere@origin r1, Sphere@(5,0,0) r1)
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "b"), 5.0f, 0.0f, 0.0f);
    UnionNode u(a, b);
    auto gf = jit_sdf_grad(u);
    ASSERT_NE(gf, nullptr);

    float dot = 0.0f;
    // At (2,0,0) — closer to sphere a, the normal points +x
    gf(2,1, 0,0, 0,0, &dot);  EXPECT_NEAR(dot, 1.0f, 1e-4f);
    // At (3,0,0) — closer to sphere b, the normal points -x (toward b's center)
    gf(3,1, 0,0, 0,0, &dot);  EXPECT_NEAR(dot, -1.0f, 1e-4f);
}

// ─── Ray-cast selection (ScenePicker) ─────────────────────────────────────────

TEST(ScenePick, HitsExpectedObject) {
    // Three spheres on the X axis: left (-3), center (0), right (+3).
    SceneGraph scene;
    scene.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "L"), -3.0f, 0.0f, 0.0f, "tL"));
    scene.add_object(std::make_shared<SphereNode>(1.0f, "C"));
    scene.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "R"), 3.0f, 0.0f, 0.0f, "tR"));

    ScenePicker picker;
    auto rb = picker.rebuild(scene);
    ASSERT_TRUE(rb.has_value()) << (rb ? "" : rb.error());
    EXPECT_EQ(picker.object_count(), 3u);

    // Camera looks at the origin from +Z. The center pixel → the center sphere.
    Camera cam;
    cam.position = {0.0f, 0.0f, 10.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.fov_deg  = 60.0f;

    auto hit_center = picker.pick_pixel(cam, 200, 150, 400, 300);
    ASSERT_TRUE(hit_center.has_value());
    EXPECT_EQ(*hit_center, "C") << "center hit: " << *hit_center;
}

TEST(ScenePick, CenterPixelHitsCenterSphere) {
    SceneGraph scene;
    scene.add_object(std::make_shared<SphereNode>(1.5f, "ball"));

    ScenePicker picker;
    ASSERT_TRUE(picker.rebuild(scene).has_value());

    Camera cam;
    cam.position = {0.0f, 0.0f, 8.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.fov_deg  = 50.0f;

    // The center pixel must hit the sphere.
    auto hit = picker.pick_pixel(cam, 256, 256, 512, 512);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, "ball");
}

TEST(ScenePick, CornerPixelMissesLoneSphere) {
    // A small sphere in the center; a corner pixel must miss it.
    SceneGraph scene;
    scene.add_object(std::make_shared<SphereNode>(0.3f, "tiny"));

    ScenePicker picker;
    ASSERT_TRUE(picker.rebuild(scene).has_value());

    Camera cam;
    cam.position = {0.0f, 0.0f, 8.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.fov_deg  = 60.0f;

    auto miss = picker.pick_pixel(cam, 0, 0, 512, 512);
    EXPECT_FALSE(miss.has_value());
}

TEST(ScenePick, EmptySceneReturnsNullopt) {
    SceneGraph scene;
    ScenePicker picker;
    ASSERT_TRUE(picker.rebuild(scene).has_value());
    EXPECT_EQ(picker.object_count(), 0u);

    Camera cam;
    cam.position = {0, 0, 5};
    cam.target   = {0, 0, 0};
    cam.up       = {0, 1, 0};
    auto r = picker.pick_pixel(cam, 100, 100, 200, 200);
    EXPECT_FALSE(r.has_value());
}

TEST(ScenePick, LeftPixelHitsLeftObject) {
    // Two spheres: left (-2.5) and right (+2.5).
    SceneGraph scene;
    scene.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "Lg"), -2.5f, 0.0f, 0.0f, "left"));
    scene.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "Rg"), 2.5f, 0.0f, 0.0f, "right"));

    ScenePicker picker;
    ASSERT_TRUE(picker.rebuild(scene).has_value());

    Camera cam;
    cam.position = {0.0f, 0.0f, 10.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};
    cam.up       = {0.0f, 1.0f, 0.0f};
    cam.fov_deg  = 60.0f;

    // A pixel left of center → the left sphere.
    auto left_hit = picker.pick_pixel(cam, 128, 256, 512, 512);
    ASSERT_TRUE(left_hit.has_value());
    EXPECT_EQ(*left_hit, "left");

    // A pixel to the right → the right sphere.
    auto right_hit = picker.pick_pixel(cam, 384, 256, 512, 512);
    ASSERT_TRUE(right_hit.has_value());
    EXPECT_EQ(*right_hit, "right");
}

// ─── Shading models — both compile and produce a non-trivial image ──────────
// We don't check exact pixel values (those depend on the BRDF math), but we
// do verify that:
//   - both models emit a verifiable LLVM module
//   - rendering produces non-zero output on a hit
//   - the energy-conserving Cook-Torrance model produces a darker mean
//     than Blinn-Phong for the same scene (since CT lacks the Phong-style
//     ambient boost)
static float mean_brightness(const SceneGraph& s,
                             TracerConfig::ShadingModel model) {
    TracerConfig cfg; cfg.shading_model = model;
    cfg.enable_shadows = false;  // keep the test fast and deterministic
    cfg.enable_ao      = false;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto fn_or = jit_pool().back()->load(std::move(mod), std::move(ctx));
    EXPECT_TRUE(fn_or.has_value());
    auto fn = *fn_or;

    int W = 128, H = 96;
    std::vector<float> px(W*H*4, 0.0f);
    RenderParams rp; rp.width = W; rp.height = H;
    TileScheduler::render(fn, px.data(), s.camera(), rp);

    double sum = 0;
    for (int i = 0; i < W*H; ++i)
        sum += (px[i*4 + 0] + px[i*4 + 1] + px[i*4 + 2]) / 3.0;
    return static_cast<float>(sum / (W * H));
}

TEST(Shading, BlinnPhongRendersScene) {
    SceneGraph s;
    Material m; m.albedo = {0.8f, 0.3f, 0.3f}; m.roughness = 0.3f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto& L = s.lights(); L.clear();
    L.push_back({{5,5,5}, {1,1,1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto b = mean_brightness(s, TracerConfig::ShadingModel::BlinnPhong);
    EXPECT_GT(b, 0.05f);     // some lit pixels
    EXPECT_LT(b, 1.0f);      // not saturated everywhere
}

TEST(Shading, CookTorranceRendersScene) {
    SceneGraph s;
    Material m; m.albedo = {0.8f, 0.3f, 0.3f}; m.roughness = 0.3f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto& L = s.lights(); L.clear();
    L.push_back({{5,5,5}, {1,1,1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto b = mean_brightness(s, TracerConfig::ShadingModel::CookTorrance);
    EXPECT_GT(b, 0.01f);
    EXPECT_LT(b, 1.0f);
}

TEST(Shading, CookTorranceIsEnergyConservingVsPhong) {
    // Same scene, same light — energy-conserving CT should be NO BRIGHTER
    // than Phong (which adds an unphysical ambient term). The exact ratio
    // depends on roughness; here we just check direction.
    SceneGraph s;
    Material m; m.albedo = {0.8f, 0.3f, 0.3f}; m.roughness = 0.5f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    auto& L = s.lights(); L.clear();
    L.push_back({{5,5,5}, {1,1,1}, 1.0f});
    s.camera().position = {0, 0, 4};
    s.camera().target   = {0, 0, 0};

    auto phong = mean_brightness(s, TracerConfig::ShadingModel::BlinnPhong);
    auto ct    = mean_brightness(s, TracerConfig::ShadingModel::CookTorrance);
    EXPECT_LT(ct, phong);
}

// ─── BVH-accelerated picker early-exit ──────────────────────────────────────

TEST(ScenePick, FarMissReturnsNullopt) {
    // 10 small spheres clustered around origin. A ray pointing away from
    // them should miss with the ray-AABB early-exit kicking in.
    SceneGraph s;
    for (int i = 0; i < 10; ++i) {
        s.add_object(std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.4f, "s"),
            (i - 5) * 0.3f, 0, 0,
            "s" + std::to_string(i)));
    }
    Camera cam;
    cam.position = {0.0f, 0.0f, 10.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};

    ScenePicker picker;
    auto rb = picker.rebuild(s);
    ASSERT_TRUE(rb.has_value()) << rb.error();

    // Top-right corner of a 100x100 image should not hit anything — the
    // ray points up and right toward the sky.
    auto hit_corner = picker.pick_pixel(cam, 99, 0, 100, 100);
    EXPECT_FALSE(hit_corner.has_value());
}

TEST(ScenePick, EarlyExitDoesNotMissNearbyHit) {
    // Sanity: with the early-exit active, central pixels still hit the
    // object correctly. (Regression guard for the AABB margin.)
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.8f, "ball"));
    Camera cam;
    cam.position = {0.0f, 0.0f, 4.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};

    ScenePicker picker;
    auto rb = picker.rebuild(s);
    ASSERT_TRUE(rb.has_value()) << rb.error();

    auto hit = picker.pick_pixel(cam, 50, 50, 100, 100);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, "ball");
}

TEST(ScenePick, EarlyExitDisabledForInfinitePlane) {
    // If the scene contains a Plane (infinite AABB), the early-exit code
    // path is skipped entirely. Verify the regular tracing still works.
    SceneGraph s;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    s.add_object(std::make_shared<SphereNode>(0.5f, "ball"));
    Camera cam;
    cam.position = {0.0f, 0.0f, 4.0f};
    cam.target   = {0.0f, 0.0f, 0.0f};

    ScenePicker picker;
    auto rb = picker.rebuild(s);
    ASSERT_TRUE(rb.has_value()) << rb.error();

    // Center pixel hits the ball.
    auto hit = picker.pick_pixel(cam, 50, 50, 100, 100);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, "ball");

    // A ray aimed downward should hit the floor.
    cam.target = {0.0f, -10.0f, 0.0f};
    auto floor_hit = picker.pick_pixel(cam, 50, 50, 100, 100);
    ASSERT_TRUE(floor_hit.has_value());
    EXPECT_EQ(*floor_hit, "floor");
}

// ─── Camera projection modes ────────────────────────────────────────────────

TEST(Camera, OrthographicProducesNonEmptyImage) {
    // Sanity: rendering with ortho projection produces non-zero output and
    // does not crash. Compare pixel coverage with perspective; the two
    // images differ but both have lit pixels.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    auto& L = s.lights(); L.clear();
    L.push_back({{5,5,5}, {1,1,1}, 1.0f});

    s.camera().position = {0.0f, 0.0f, 4.0f};
    s.camera().target   = {0.0f, 0.0f, 0.0f};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto fn_or = jit_pool().back()->load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());
    auto fn = *fn_or;

    int W = 128, H = 96;
    std::vector<float> px_persp(W*H*4), px_ortho(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;

    s.camera().projection = Camera::Projection::Perspective;
    s.camera().fov_deg    = 60.0f;
    TileScheduler::render(fn, px_persp.data(), s.camera(), rp);

    s.camera().projection = Camera::Projection::Orthographic;
    s.camera().ortho_size = 3.0f;
    TileScheduler::render(fn, px_ortho.data(), s.camera(), rp);

    auto lit_pixels = [&](const std::vector<float>& im) {
        // Count only sphere pixels — filter out the blue sky background.
        int count = 0;
        for (int i = 0; i < W*H; ++i) {
            float r = im[i*4 + 0], g = im[i*4 + 1], b = im[i*4 + 2];
            if (r > 0.05f && r >= b * 0.85f && g >= b * 0.85f) ++count;
        }
        return count;
    };
    int lp = lit_pixels(px_persp);
    int lo = lit_pixels(px_ortho);
    EXPECT_GT(lp, 100);
    EXPECT_GT(lo, 100);
    // The two should not be identical pixel-for-pixel.
    int diff = 0;
    for (int i = 0; i < W*H*4; ++i)
        if (std::abs(px_persp[i] - px_ortho[i]) > 0.01f) ++diff;
    EXPECT_GT(diff, W*H / 4);  // at least a quarter of the channels differ
}

TEST(Camera, OrthographicShowsEqualSizedSpheres) {
    // Two same-radius spheres at different Z. In perspective, the farther
    // sphere occupies fewer pixels; in orthographic, the two pixel counts
    // are equal regardless of Z.
    SceneGraph s;
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "a"), -1.5f, 0,  2.0f, "ta"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "b"),  1.5f, 0, -4.0f, "tb"));
    auto& L = s.lights(); L.clear();
    L.push_back({{5,5,5}, {1,1,1}, 1.0f});
    s.camera().position = {0.0f, 0.0f, 8.0f};
    s.camera().target   = {0.0f, 0.0f, 0.0f};

    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto fn_or = jit_pool().back()->load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());
    auto fn = *fn_or;

    int W = 300, H = 200;
    std::vector<float> px(W*H*4);
    RenderParams rp; rp.width = W; rp.height = H;

    // Helper: count lit pixels in left and right halves. We filter out sky
    // pixels (which are mostly blue: high B, lower R/G) by requiring the
    // red channel to be the dominant or nearly-equal — true for the
    // lit white sphere, false for the blue sky.
    auto count_halves = [&]() {
        int left = 0, right = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float r = px[(y*W+x)*4 + 0];
                float g = px[(y*W+x)*4 + 1];
                float bch = px[(y*W+x)*4 + 2];
                // Sphere is unlit-grey -> sphere shaded -> ratios near 1.
                // Sky has b >> r. Require r >= b * 0.85 to count.
                //
                // The lower bound was 0.05 when CPU codegen applied
                // sqrt() gamma encoding; without that gamma (dropped
                // in v4.0.5 for parity with the GPU paths), shadowed
                // sphere pixels can dip to ~0.01, so we relax the
                // brightness floor to keep the heuristic counting
                // them as sphere coverage rather than background.
                if (r > 0.01f && r >= bch * 0.85f && g >= bch * 0.85f) {
                    if (x < W/2) ++left; else ++right;
                }
            }
        return std::pair{left, right};
    };

    // Perspective: the near sphere (left side of image, z=+2) appears
    // larger than the far sphere (z=-4).
    s.camera().projection = Camera::Projection::Perspective;
    s.camera().fov_deg    = 45.0f;
    std::fill(px.begin(), px.end(), 0.0f);
    TileScheduler::render(fn, px.data(), s.camera(), rp);
    auto [pl, pr] = count_halves();
    EXPECT_GT(pl, pr)
        << "Perspective: near sphere (left) should cover more pixels than far. "
        << "Got pl=" << pl << " pr=" << pr;

    // Orthographic: roughly balanced; with the same world-space radius
    // each sphere covers the same image area regardless of Z.
    s.camera().projection = Camera::Projection::Orthographic;
    s.camera().ortho_size = 5.0f;
    std::fill(px.begin(), px.end(), 0.0f);
    TileScheduler::render(fn, px.data(), s.camera(), rp);
    auto [ol, orr] = count_halves();
    int ortho_max = std::max(ol, orr);
    int ortho_min = std::min(ol, orr);
    EXPECT_LT(ortho_max - ortho_min, ortho_max / 4)
        << "Orthographic: coverage should be balanced. "
        << "Got ol=" << ol << " or=" << orr;
}
