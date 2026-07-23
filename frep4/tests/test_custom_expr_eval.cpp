// tests/test_custom_expr_eval.cpp
//
// Tests the CPU interpreter added to CustomExprNode::eval(). Without
// this path the picker, BVH, and marching-cubes mesh extraction throw
// at the first eval() call on a CustomExpr scene.

#include "core/frep/scene.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/mesh/marching_cubes.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using namespace frep;

TEST(CustomExprEval, ArithmeticBasics) {
    CustomExprNode n("x*x + y*y + z*z - 1.0", "sphere");
    EXPECT_NEAR(n.eval(0, 0, 0), -1.0f,    1e-6);
    EXPECT_NEAR(n.eval(1, 0, 0),  0.0f,    1e-6);
    EXPECT_NEAR(n.eval(0.5f, 0.5f, 0.5f), 0.75f - 1.0f, 1e-6);
}

TEST(CustomExprEval, MathFunctions) {
    CustomExprNode n("sin(x) + cos(y) + sqrt(z) - 1.0", "f");
    float v = n.eval(0.0f, 0.0f, 4.0f);
    // sin(0) + cos(0) + sqrt(4) - 1 = 0 + 1 + 2 - 1 = 2
    EXPECT_NEAR(v, 2.0f, 1e-5);
}

TEST(CustomExprEval, BinaryFunctions) {
    CustomExprNode npow("pow(x, 3)", "p");
    EXPECT_NEAR(npow.eval(2, 0, 0), 8.0f, 1e-5);
    CustomExprNode nmin("min(x, y)", "mn");
    EXPECT_NEAR(nmin.eval(3, 7, 0), 3.0f, 1e-6);
    CustomExprNode nmax("max(x, y)", "mx");
    EXPECT_NEAR(nmax.eval(3, 7, 0), 7.0f, 1e-6);
}

TEST(CustomExprEval, UnaryMinusAndPrecedence) {
    CustomExprNode n("-x + 2*y - x*y", "f");
    // f(2, 3) = -2 + 6 - 6 = -2
    EXPECT_NEAR(n.eval(2, 3, 0), -2.0f, 1e-6);
    CustomExprNode n2("(x + y) * (x - y)", "diff_of_sq");
    // (3+2)*(3-2) = 5
    EXPECT_NEAR(n2.eval(3, 2, 0), 5.0f, 1e-6);
}

TEST(CustomExprEval, GyroidSurface) {
    CustomExprNode n("sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)", "gy");
    // At origin all sin terms are 0, so f(0,0,0) = 0.
    EXPECT_NEAR(n.eval(0, 0, 0), 0.0f, 1e-6);
    // At (π/2, 0, 0): sin(π/2)cos(0) + sin(0)cos(0) + sin(0)cos(π/2) = 1 + 0 + 0 = 1
    EXPECT_NEAR(n.eval(M_PI_2, 0, 0), 1.0f, 1e-5);
}

TEST(CustomExprEval, MarchingCubesWorks) {
    // Without an eval() interpreter, this would throw on the first
    // SDF sample. With it, the sphere mesh-extracts cleanly.
    SceneGraph s;
    s.add_object(std::make_shared<CustomExprNode>(
        "sqrt(x*x + y*y + z*z) - 1.0", "sphere"));
    mesh::MarchingCubesParams p;
    p.rx = p.ry = p.rz = 24;
    p.auto_bounds = false;
    p.bmin[0]=p.bmin[1]=p.bmin[2] = -1.3f;
    p.bmax[0]=p.bmax[1]=p.bmax[2] =  1.3f;
    auto m = mesh::extract_iso_mesh(s, p);
    EXPECT_GT(m.vertices.size(), 100u);
    EXPECT_GT(m.indices.size(), 100u);
    // Sphere should have closed surface — Euler should be roughly V - E + F ≈ 2.
    // For an extracted mesh that's ~true; we just check non-trivial size.
}

TEST(CustomExprEval, UnknownIdentifierThrows) {
    CustomExprNode n("foo(x)", "f");
    EXPECT_THROW(n.eval(0, 0, 0), std::runtime_error);
}

TEST(CustomExprEval, UnknownVariableThrows) {
    CustomExprNode n("w + x", "f");  // 'w' isn't defined
    EXPECT_THROW(n.eval(0, 0, 0), std::runtime_error);
}
