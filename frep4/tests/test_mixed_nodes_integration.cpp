// tests/test_mixed_nodes_integration.cpp
//
// Integration tests for the corner cases of mixing built-in nodes,
// CustomExpr nodes, plugin nodes, and deformations. These are the
// kind of scenes a real user assembles in the GUI, and the kind of
// interactions that are silently broken if no test exercises them.
//
// Specifically:
//   H1. Deformation wrapping a CustomExpr — Twist(CustomExpr(...))
//       on both CPU eval and GLSL emit. The deformation substitutes
//       x/y/z inside its child; the CustomExpr just emits the string.
//       Failure mode: substitution doesn't reach into the CustomExpr.
//
//   H2. Deformation wrapping a plugin node — Twist(Capsule(...)).
//       Similar substitution requirement; the plugin's emit_glsl
//       receives child_exprs but not transformed coordinates.
//
//   H3. CSG of three different node-kind families:
//       Union(SphereNode, CustomExprNode, TestPluginNode). Tests that
//       the GLSL emitter's per-object dispatch handles mixed kinds
//       correctly.
//
//   H4. Mesh export (marching cubes) of a CustomExpr inside a
//       Twist deformation. Exercises the eval() interpreter through
//       a nested chain.
//
// We use a TestCapsuleNode inline (same as test_plugin_glsl.cpp) so
// these tests don't depend on the dynamic plugin being loadable.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/gpu/glsl_emitter.hpp"

#include <llvm/IR/Intrinsics.h>

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using namespace frep;

namespace {

// Inline plugin-style node — emits its own GLSL via the emit_glsl
// virtual, exercising the GLSL emitter's default-branch dispatch.
class TestCapsuleNode final : public FRepNode {
public:
    TestCapsuleNode(float h, float r, std::string nid = "tcap") {
        kind = NodeKind::Plugin;
        id   = std::move(nid);
        params["height"] = h;
        params["radius"] = r;
    }
    const char* type_name() const noexcept override { return "TestCapsule"; }
    float eval(float x, float y, float z) const override {
        float h = params.at("height"), r = params.at("radius");
        float y_off = y - std::clamp(y, -h, h);
        return std::sqrt(x*x + y_off*y_off + z*z) - r;
    }
    llvm::Value* codegen(CgCtx& c, llvm::Value* x,
                         llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        float h = params.at("height"), r = params.at("radius");
        auto ny = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, y, c.fc(-h));
        auto cy = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, ny, c.fc(h));
        auto y_off = b.CreateFSub(y, cy);
        auto x2 = b.CreateFMul(x, x);
        auto y2 = b.CreateFMul(y_off, y_off);
        auto z2 = b.CreateFMul(z, z);
        auto sum = b.CreateFAdd(b.CreateFAdd(x2, y2), z2);
        auto len = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, sum);
        return b.CreateFSub(len, c.fc(r), "tcap");
    }
    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("height")) * 31
             ^ std::hash<float>{}(params.at("radius"))
             ^ 0xDEADBEEFull;
    }
    bool emit_glsl(std::ostream& out,
                   const std::vector<std::string>& /*child_exprs*/,
                   const std::string& /*var_prefix*/) const override {
        float h = params.at("height"), r = params.at("radius");
        out << "(length(vec3(x, y - clamp(y, " << (-h) << ", " << h
            << "), z)) - " << r << ")";
        return true;
    }
};

} // anon

// ─────────────────────────────────────────────────────────────────────────────
// H1: Deformation around CustomExpr
// ─────────────────────────────────────────────────────────────────────────────
TEST(MixedNodes, TwistAroundCustomExprEvalsCorrectly) {
    // A Twist substitutes x/y/z with rotated coordinates inside the
    // child SDF. If the CustomExpr child eval() ignores the
    // substituted values, the deformation would have no effect.
    auto expr = std::make_shared<CustomExprNode>(
        "sqrt(x*x + y*y + z*z) - 1.0", "sphere");
    auto twist = std::make_shared<TwistYNode>(expr, 1.0f, "tw");

    // At p = (1, 0, 0), the sphere SDF is 0 (on the surface). With
    // Twist active at y=0, sin(0)=0 → no rotation → same value.
    EXPECT_NEAR(twist->eval(1.0f, 0, 0), 0.0f, 1e-4);

    // At p = (1, π/2, 0), with k=1.0 the rotation angle is π/2.
    // After rotation: xr = 0, zr = -1, y unchanged. Rotated point's
    // |p|² = 0 + (π/2)² + 1, so the sphere SDF returns sqrt(π²/4 + 1)
    // - 1. Then TwistY divides by its Lipschitz factor sqrt(1+(k·r)²)
    // = √2 (r = 1 here), so the final value ≈ 0.862 / √2 ≈ 0.610.
    float v = twist->eval(1.0f, static_cast<float>(M_PI_2), 0);
    float raw = std::sqrt(static_cast<float>(M_PI_2 * M_PI_2) + 1.0f) - 1.0f;
    float lip = std::sqrt(1.0f + 1.0f);  // k=1, r=1
    float expected = raw / lip;
    EXPECT_NEAR(v, expected, 1e-3) << "Twist applied to CustomExpr";
}

TEST(MixedNodes, TwistAroundCustomExprEmitsValidGlsl) {
    SceneGraph s;
    auto expr = std::make_shared<CustomExprNode>(
        "sqrt(x*x + y*y + z*z) - 1.0", "sphere");
    auto twist = std::make_shared<TwistYNode>(expr, 1.0f, "tw");
    Material m{{0.9f, 0.4f, 0.3f}};
    s.add_object(twist, m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << (r ? "" : r.error());
    // Twist emits a cos/sin rotation block; the CustomExpr emits the
    // sqrt(...) substring. Both must appear in the shader source.
    EXPECT_NE(r->source.find("cos("),  std::string::npos);
    EXPECT_NE(r->source.find("sin("),  std::string::npos);
    EXPECT_NE(r->source.find("sqrt("), std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// H2: Deformation around plugin node
// ─────────────────────────────────────────────────────────────────────────────
TEST(MixedNodes, TwistAroundPluginEvalsCorrectly) {
    // Same substitution semantics as for CustomExpr — the plugin's
    // eval() must see the deformed coordinates.
    auto cap = std::make_shared<TestCapsuleNode>(0.5f, 0.2f);
    auto twist = std::make_shared<TwistYNode>(cap, 0.5f, "tw");
    // On axis at the centre — r=0 so Lipschitz factor = 1.
    // distance = -radius = -0.2
    EXPECT_NEAR(twist->eval(0, 0, 0), -0.2f, 1e-5);
    // Well outside on axis at (2,0,0) — r = 2, lip = sqrt(1 + (0.5*2)²)
    // = √2. Raw capsule SDF at (2,0,0) = sqrt(4) - 0.2 = 1.8.
    // Final = 1.8 / √2 ≈ 1.273.
    float raw = std::sqrt(4.0f) - 0.2f;
    float lip = std::sqrt(1.0f + 1.0f);
    EXPECT_NEAR(twist->eval(2.0f, 0, 0), raw / lip, 1e-4);
}

TEST(MixedNodes, TwistAroundPluginEmitsValidGlsl) {
    SceneGraph s;
    auto cap = std::make_shared<TestCapsuleNode>(0.6f, 0.25f);
    auto twist = std::make_shared<TwistYNode>(cap, 0.5f, "tw");
    Material m{{0.9f, 0.4f, 0.3f}};
    s.add_object(twist, m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << (r ? "" : r.error());
    // Both Twist's rotation and the plugin's length(vec3(...)) must appear.
    EXPECT_NE(r->source.find("length(vec3"), std::string::npos);
    EXPECT_NE(r->source.find("cos("),        std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// H3: CSG of mixed node-kind families
// ─────────────────────────────────────────────────────────────────────────────
TEST(MixedNodes, UnionOfBuiltInExprAndPlugin_EmitsGlsl) {
    // Union(SphereNode, CustomExprNode, TestCapsuleNode) — every
    // major dispatch case in one tree.
    SceneGraph s;
    auto sph = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.3f, "s"),
        -1.0f, 0, 0, "st");
    auto expr = std::make_shared<TranslateNode>(
        std::make_shared<CustomExprNode>(
            "x*x + y*y + z*z - 0.09", "e"),
        0, 0, 0, "et");
    auto cap = std::make_shared<TranslateNode>(
        std::make_shared<TestCapsuleNode>(0.3f, 0.15f),
        1.0f, 0, 0, "ct");
    auto u1 = std::make_shared<UnionNode>(sph, expr, "u1");
    auto u2 = std::make_shared<UnionNode>(u1, cap, "u2");
    Material m{{0.7f, 0.5f, 0.8f}};
    s.add_object(u2, m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << (r ? "" : r.error());
    // The shader source should mention all three SDF kinds.
    EXPECT_NE(r->source.find("min("),  std::string::npos);  // Union → min()
    // Sphere translates to `length(...) - r` in the emitter.
    // CustomExpr inlines its expression literally.
    // Plugin emits `length(vec3(x, y - clamp...`.
    auto has_sphere = r->source.find("length(") != std::string::npos;
    auto has_expr   = r->source.find("0.09")    != std::string::npos;
    auto has_clamp  = r->source.find("clamp(y") != std::string::npos;
    EXPECT_TRUE(has_sphere) << "missing built-in sphere SDF";
    EXPECT_TRUE(has_expr)   << "missing CustomExpr literal 0.09";
    EXPECT_TRUE(has_clamp)  << "missing plugin capsule length(...clamp)";
}

TEST(MixedNodes, UnionOfBuiltInExprAndPlugin_EvalsCorrectly) {
    // Each child has its own SDF; Union returns min of all three.
    auto sph = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.5f, "s"),
        -1.0f, 0, 0, "st");
    auto expr = std::make_shared<TranslateNode>(
        std::make_shared<CustomExprNode>(
            "sqrt(x*x + y*y + z*z) - 0.5", "e"),
        0, 0, 0, "et");
    auto cap = std::make_shared<TranslateNode>(
        std::make_shared<TestCapsuleNode>(0.5f, 0.25f),
        1.0f, 0, 0, "ct");
    auto u = std::make_shared<UnionNode>(
        std::make_shared<UnionNode>(sph, expr, "u1"),
        cap, "u2");

    // At origin, expr SDF = sqrt(0)-0.5 = -0.5 (deepest).
    // Sphere is 1 unit away → sqrt(1)-0.5 = 0.5.
    // Capsule is 1 unit away → ~0.75 from surface.
    EXPECT_NEAR(u->eval(0, 0, 0), -0.5f, 1e-4);
    // At (-1, 0, 0), sphere SDF = -0.5, others positive. Union picks
    // the sphere.
    EXPECT_NEAR(u->eval(-1.0f, 0, 0), -0.5f, 1e-4);
}

// ─────────────────────────────────────────────────────────────────────────────
// H4: Marching-cubes export of CustomExpr wrapped in deformation
// ─────────────────────────────────────────────────────────────────────────────
TEST(MixedNodes, MarchingCubesOnTwistedCustomExpr) {
    // Without working eval() at every level (CustomExpr + Twist), this
    // would throw at the first SDF sample. Verifies the eval-path
    // through nested wrappers.
    SceneGraph s;
    auto expr = std::make_shared<CustomExprNode>(
        "sqrt(x*x + y*y + z*z) - 0.8", "sphere");
    auto twist = std::make_shared<TwistYNode>(expr, 1.5f, "tw");
    s.add_object(twist);

    mesh::MarchingCubesParams p;
    p.rx = p.ry = p.rz = 24;
    p.auto_bounds = false;
    p.bmin[0] = p.bmin[1] = p.bmin[2] = -1.2f;
    p.bmax[0] = p.bmax[1] = p.bmax[2] =  1.2f;

    auto m = mesh::extract_iso_mesh(s, p);
    EXPECT_GT(m.vertices.size(), 50u);
    EXPECT_GT(m.indices.size(),  50u);
}
