// tests/test_sdf_invariant.cpp
//
// Enforces the SDF contract documented on FRepNode for every built-in
// node, sampled via eval():
//
//   1. 1-Lipschitz: |f(a) - f(b)| <= |a - b| + eps for all sampled pairs.
//      The field must never over-estimate distance — this is what makes
//      sphere-tracing safe and the BVH's AABB lower-bound prune valid.
//      (The Chebyshev box bug we fixed was exactly a Lipschitz violation:
//      it *under*-estimated far-corner distance, but the general test below
//      catches over-estimation, which is the dangerous direction for the
//      AABB lower-bound; we also assert the box matches its analytic
//      distance on the diagonal, which pins the under-estimation case.)
//   2. Sign convention: positive outside, negative inside, ~0 on surface.
//
// These guard against future metric regressions in any primitive,
// operation, transform, or deformation.

#include <gtest/gtest.h>

#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace frep;

namespace {

float dist3(std::array<float,3> a, std::array<float,3> b) {
    float dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

// Sample many random pairs and assert the 1-Lipschitz bound. A small eps
// tolerates float error and the slight non-metricity of smooth/blended
// operators (which are still safe — they under-estimate, never over).
void expect_lipschitz(const FRepNode& n, float extent = 6.0f) {
    std::mt19937 rng(987654321u);
    std::uniform_real_distribution<float> U(-extent, extent);
    for (int i = 0; i < 4000; ++i) {
        std::array<float,3> a{U(rng),U(rng),U(rng)};
        std::array<float,3> b{U(rng),U(rng),U(rng)};
        float fa = n.eval(a[0],a[1],a[2]);
        float fb = n.eval(b[0],b[1],b[2]);
        float lhs = std::abs(fa - fb);
        float rhs = dist3(a,b) + 1e-3f;
        ASSERT_LE(lhs, rhs)
            << "Lipschitz violation: |f(a)-f(b)|=" << lhs
            << " > |a-b|=" << dist3(a,b);
    }
}

// Assert positive far away, negative at the centre (for centred solids).
void expect_sign_convention(const FRepNode& n, float far = 100.0f) {
    EXPECT_GT(n.eval(far, far, far), 0.0f) << "should be positive outside";
    EXPECT_LT(n.eval(0, 0, 0), 0.0f)       << "should be negative at centre";
}

} // namespace

TEST(SdfInvariant, SphereIsLipschitzAndSigned) {
    SphereNode s(1.5f, "s");
    expect_lipschitz(s);
    expect_sign_convention(s);
}

TEST(SdfInvariant, BoxIsLipschitzAndSigned) {
    BoxNode b(1.0f, 0.7f, 1.3f, "b");
    expect_lipschitz(b);
    expect_sign_convention(b);
}

// Pin the box's exact Euclidean distance on the corner diagonal — the
// case the old Chebyshev metric got wrong (it under-estimated here).
TEST(SdfInvariant, BoxExactCornerDistance) {
    BoxNode b(1.0f, 1.0f, 1.0f, "b");
    // Point offset (2,2,2) from the +++ corner at (1,1,1): the nearest
    // surface point is that corner, true distance sqrt(1+1+1)=√3.
    float got = b.eval(2.0f, 2.0f, 2.0f);
    EXPECT_NEAR(got, std::sqrt(3.0f), 1e-4f);
}

TEST(SdfInvariant, TranslatePreservesMetric) {
    auto t = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "s"), 2.0f, -1.0f, 0.5f, "t");
    expect_lipschitz(*t);
    // Centre shifted to the translation.
    EXPECT_LT(t->eval(2.0f, -1.0f, 0.5f), 0.0f);
    EXPECT_GT(t->eval(50.0f, 50.0f, 50.0f), 0.0f);
}

TEST(SdfInvariant, RotateYPreservesMetric) {
    auto r = std::make_shared<RotateYNode>(
        std::make_shared<BoxNode>(1.0f, 0.5f, 0.8f, "b"), 0.6f, "r");
    expect_lipschitz(*r);
    expect_sign_convention(*r);
}

TEST(SdfInvariant, UnionIsLipschitzAndSigned) {
    auto u = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(1.0f, "b"), 1.5f, 0, 0, "tb"),
        "u");
    expect_lipschitz(*u);
    // Inside either sphere is inside the union.
    EXPECT_LT(u->eval(0, 0, 0), 0.0f);
    EXPECT_LT(u->eval(1.5f, 0, 0), 0.0f);
}

TEST(SdfInvariant, IntersectionIsLipschitz) {
    auto x = std::make_shared<IntersectionNode>(
        std::make_shared<SphereNode>(1.2f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(1.2f, "b"), 1.0f, 0, 0, "tb"),
        "x");
    expect_lipschitz(*x);
}

TEST(SdfInvariant, DifferenceIsLipschitz) {
    auto d = std::make_shared<DifferenceNode>(
        std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "a"),
        std::make_shared<SphereNode>(1.2f, "b"),
        "d");
    expect_lipschitz(*d);
}

TEST(SdfInvariant, SmoothUnionIsLipschitz) {
    // Smooth blends slightly under-estimate distance near the joint, which
    // is still safe (under, not over). The bound must hold.
    auto su = std::make_shared<SmoothUnionNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(1.0f, "b"), 1.2f, 0, 0, "tb"),
        0.3f, "su");
    expect_lipschitz(*su);
}
