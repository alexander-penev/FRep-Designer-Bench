// tests/test_transform_gizmo.cpp
//
// Tests for the per-object translation gizmo: SceneGraph::set_translation /
// get_translation and the SetTransformCommand undo wrapper.
//
// The translation is represented as an implicit TranslateNode wrapping
// the object's geometry root, so these tests also verify the wrap /
// unwrap / in-place-update behaviour that keeps the tree clean.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/undo/undo_stack.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace frep;

namespace {
SceneGraph one_sphere() {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), Material{});
    return s;
}
}  // namespace

TEST(TransformGizmo, DefaultTranslationIsZero) {
    auto s = one_sphere();
    auto t = s.get_translation("ball");
    EXPECT_FLOAT_EQ(t[0], 0.0f);
    EXPECT_FLOAT_EQ(t[1], 0.0f);
    EXPECT_FLOAT_EQ(t[2], 0.0f);
}

TEST(TransformGizmo, SetThenGetRoundTrips) {
    auto s = one_sphere();
    s.set_translation("ball", {1.5f, -2.0f, 3.25f});
    auto t = s.get_translation("ball");
    EXPECT_FLOAT_EQ(t[0],  1.5f);
    EXPECT_FLOAT_EQ(t[1], -2.0f);
    EXPECT_FLOAT_EQ(t[2],  3.25f);
}

TEST(TransformGizmo, WrapsGeometryInTranslateNode) {
    auto s = one_sphere();
    // Before: root is the Sphere itself.
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Sphere");
    s.set_translation("ball", {1.0f, 0.0f, 0.0f});
    // After: root is a Translate wrapping the Sphere, keeping the id.
    const auto& g = s.objects().at("ball").geometry;
    EXPECT_EQ(std::string(g->type_name()), "Translate");
    EXPECT_EQ(g->id, "ball");
    ASSERT_EQ(g->children.size(), 1u);
    EXPECT_EQ(std::string(g->children[0]->type_name()), "Sphere");
}

TEST(TransformGizmo, RepeatedSetsDoNotNest) {
    auto s = one_sphere();
    s.set_translation("ball", {1.0f, 0.0f, 0.0f});
    s.set_translation("ball", {2.0f, 0.0f, 0.0f});
    s.set_translation("ball", {3.0f, 0.0f, 0.0f});
    // Still a single Translate, not Translate(Translate(Translate(...))).
    const auto& g = s.objects().at("ball").geometry;
    EXPECT_EQ(std::string(g->type_name()), "Translate");
    ASSERT_EQ(g->children.size(), 1u);
    EXPECT_EQ(std::string(g->children[0]->type_name()), "Sphere");
    EXPECT_FLOAT_EQ(g->params.at("tx"), 3.0f);
}

TEST(TransformGizmo, ZeroOffsetUnwraps) {
    auto s = one_sphere();
    s.set_translation("ball", {5.0f, 0.0f, 0.0f});
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Translate");
    // Setting back to zero should unwrap to the bare Sphere.
    s.set_translation("ball", {0.0f, 0.0f, 0.0f});
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Sphere");
}

TEST(TransformGizmo, UndoRestoresPreviousOffset) {
    auto s = one_sphere();
    s.set_translation("ball", {1.0f, 1.0f, 1.0f});

    undo::UndoStack stack;
    stack.push_apply(std::make_unique<undo::SetTransformCommand>(
        s, "ball", std::array<float, 3>{4.0f, 5.0f, 6.0f}));

    auto after = s.get_translation("ball");
    EXPECT_FLOAT_EQ(after[0], 4.0f);
    EXPECT_FLOAT_EQ(after[1], 5.0f);
    EXPECT_FLOAT_EQ(after[2], 6.0f);

    stack.undo();
    auto restored = s.get_translation("ball");
    EXPECT_FLOAT_EQ(restored[0], 1.0f);
    EXPECT_FLOAT_EQ(restored[1], 1.0f);
    EXPECT_FLOAT_EQ(restored[2], 1.0f);

    stack.redo();
    auto redone = s.get_translation("ball");
    EXPECT_FLOAT_EQ(redone[0], 4.0f);
    EXPECT_FLOAT_EQ(redone[1], 5.0f);
    EXPECT_FLOAT_EQ(redone[2], 6.0f);
}

TEST(TransformGizmo, TranslationAffectsEvalPosition) {
    // The whole point: translating an object moves where its surface is.
    // Sample the SDF before and after translating a unit sphere.
    auto s = one_sphere();
    const auto& g0 = s.objects().at("ball").geometry;
    // Unit sphere at origin: distance at (2,0,0) is 1.0 (2 - radius).
    EXPECT_NEAR(g0->eval(2.0f, 0.0f, 0.0f), 1.0f, 1e-5f);

    s.set_translation("ball", {2.0f, 0.0f, 0.0f});
    const auto& g1 = s.objects().at("ball").geometry;
    // Sphere now centred at (2,0,0): distance at (2,0,0) is -1 (inside),
    // and the surface point (4,0,0) should be ~0.
    EXPECT_NEAR(g1->eval(4.0f, 0.0f, 0.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(g1->eval(2.0f, 0.0f, 0.0f), -1.0f, 1e-5f);
}

// ── Rotation / scale gizmo (Stage 4) ──────────────────────────────────────

TEST(TransformGizmo, ScaleSetGetRoundTrips) {
    auto s = one_sphere();
    EXPECT_FLOAT_EQ(s.get_scale("ball"), 1.0f);   // identity by default
    s.set_scale("ball", 2.5f);
    EXPECT_FLOAT_EQ(s.get_scale("ball"), 2.5f);
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Scale");
}

TEST(TransformGizmo, RotationSetGetRoundTrips) {
    auto s = one_sphere();
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 0.0f);
    s.set_rotation_y("ball", 1.0f);               // 1 radian
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 1.0f);
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "RotateY");
}

TEST(TransformGizmo, IdentityScaleAndRotationUnwrap) {
    auto s = one_sphere();
    s.set_scale("ball", 3.0f);
    s.set_scale("ball", 1.0f);                    // identity → unwrap
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Sphere");
    s.set_rotation_y("ball", 0.5f);
    s.set_rotation_y("ball", 0.0f);               // identity → unwrap
    EXPECT_EQ(std::string(s.objects().at("ball").geometry->type_name()),
              "Sphere");
}

TEST(TransformGizmo, CanonicalTRSOrderAndNoNesting) {
    auto s = one_sphere();
    // Apply all three; expect Translate(RotateY(Scale(Sphere))).
    s.set_scale("ball", 2.0f);
    s.set_rotation_y("ball", 0.7f);
    s.set_translation("ball", {1.0f, 0.0f, 0.0f});

    const auto* g = s.objects().at("ball").geometry.get();
    ASSERT_EQ(std::string(g->type_name()), "Translate");
    ASSERT_EQ(g->children.size(), 1u);
    const auto* r = g->children[0].get();
    ASSERT_EQ(std::string(r->type_name()), "RotateY");
    ASSERT_EQ(r->children.size(), 1u);
    const auto* sc = r->children[0].get();
    ASSERT_EQ(std::string(sc->type_name()), "Scale");
    ASSERT_EQ(sc->children.size(), 1u);
    EXPECT_EQ(std::string(sc->children[0]->type_name()), "Sphere");

    // Re-editing each value must update in place, not add a second
    // wrapper of the same type.
    s.set_scale("ball", 4.0f);
    s.set_rotation_y("ball", 0.2f);
    s.set_translation("ball", {5.0f, 0.0f, 0.0f});
    EXPECT_FLOAT_EQ(s.get_scale("ball"), 4.0f);
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 0.2f);
    EXPECT_FLOAT_EQ(s.get_translation("ball")[0], 5.0f);
    // Depth is still exactly 3 wrappers.
    const auto* g2 = s.objects().at("ball").geometry.get();
    EXPECT_EQ(std::string(g2->type_name()), "Translate");
    EXPECT_EQ(std::string(g2->children[0]->type_name()), "RotateY");
    EXPECT_EQ(std::string(g2->children[0]->children[0]->type_name()), "Scale");
}

TEST(TransformGizmo, ScaleAffectsEvalDistance) {
    auto s = one_sphere();
    // Unit sphere: surface at radius 1. Scale ×2 → surface at radius 2.
    s.set_scale("ball", 2.0f);
    const auto& g = s.objects().at("ball").geometry;
    EXPECT_NEAR(g->eval(2.0f, 0.0f, 0.0f), 0.0f, 1e-4f);
}
