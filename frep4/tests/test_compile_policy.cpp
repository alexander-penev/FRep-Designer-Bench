// tests/test_compile_policy.cpp
//
// Tests for CompilePolicy — the per-parameter constant-vs-runtime decision.
// These check the policy logic in isolation (placement decisions); the
// codegen integration is exercised by the incremental-params tests.

#include <gtest/gtest.h>
#include "core/compiler/compile_policy.hpp"

using namespace frep;

TEST(CompilePolicy, AllConstantPlacesEverythingConstant) {
    AllConstantPolicy p;
    EXPECT_EQ(p.decide("n", "r", ParamClass::Geometry), ParamPlacement::Constant);
    EXPECT_EQ(p.decide("n", "albedo", ParamClass::Material), ParamPlacement::Constant);
    EXPECT_EQ(p.decide("n", "ssaa", ParamClass::Render), ParamPlacement::Constant);
}

TEST(CompilePolicy, InteractiveMakesGeometryMaterialDeformRuntime) {
    auto p = ByParamClassPolicy::interactive();
    EXPECT_EQ(p.decide("n", "r",  ParamClass::Geometry), ParamPlacement::Runtime);
    EXPECT_EQ(p.decide("n", "alb", ParamClass::Material), ParamPlacement::Runtime);
    EXPECT_EQ(p.decide("n", "k",  ParamClass::Deform),   ParamPlacement::Runtime);
}

TEST(CompilePolicy, InteractiveKeepsRenderAndOtherConstant) {
    // Render/observer settings change rarely → stay constant for speed.
    auto p = ByParamClassPolicy::interactive();
    EXPECT_EQ(p.decide("n", "ssaa", ParamClass::Render), ParamPlacement::Constant);
    EXPECT_EQ(p.decide("n", "misc", ParamClass::Other),  ParamPlacement::Constant);
}

TEST(CompilePolicy, ByParamClassRespectsCustomSet) {
    // Only geometry runtime; material now constant.
    ByParamClassPolicy p({ParamClass::Geometry});
    EXPECT_EQ(p.decide("n", "r",   ParamClass::Geometry), ParamPlacement::Runtime);
    EXPECT_EQ(p.decide("n", "alb", ParamClass::Material), ParamPlacement::Constant);
}
