// tests/test_undo.cpp
//
// Tests for the undo/redo stack and its commands.

#include <array>
#include "core/undo/undo_stack.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace frep;
using namespace frep::undo;

// ─── UndoStack basics ───────────────────────────────────────────────────────

TEST(UndoStack, EmptyByDefault) {
    UndoStack u;
    EXPECT_FALSE(u.can_undo());
    EXPECT_FALSE(u.can_redo());
    EXPECT_EQ(u.size(), 0);
    EXPECT_EQ(u.cursor(), 0);
}

TEST(UndoStack, PushApplyAndUndo) {
    SceneGraph s;
    UndoStack u;
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "ball", std::make_shared<SphereNode>(1.0f, "ball"), Material{}));
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_TRUE(u.can_undo());
    EXPECT_FALSE(u.can_redo());

    u.undo();
    EXPECT_EQ(s.objects().size(), 0u);
    EXPECT_FALSE(u.can_undo());
    EXPECT_TRUE(u.can_redo());

    u.redo();
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_TRUE(u.can_undo());
}

TEST(UndoStack, NewCommandTruncatesRedoHistory) {
    SceneGraph s;
    UndoStack u;
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "a", std::make_shared<SphereNode>(1.0f, "a"), Material{}));
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "b", std::make_shared<SphereNode>(1.0f, "b"), Material{}));
    u.undo();
    u.undo();
    // Two commands available for redo.
    EXPECT_EQ(u.size(), 2);
    EXPECT_TRUE(u.can_redo());

    // Push a new command — discards both pending redos.
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "c", std::make_shared<SphereNode>(1.0f, "c"), Material{}));
    EXPECT_EQ(u.size(), 1);
    EXPECT_FALSE(u.can_redo());
    EXPECT_EQ(s.objects().count("c"), 1u);
    EXPECT_EQ(s.objects().count("a"), 0u);
}

TEST(UndoStack, ChangeObserverFires) {
    SceneGraph s;
    UndoStack u;
    int call_count = 0;
    u.set_change_observer([&]{ ++call_count; });
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "ball", std::make_shared<SphereNode>(1.0f, "ball"), Material{}));
    u.undo();
    u.redo();
    EXPECT_GE(call_count, 3);
}

// ─── Per-command behaviour ──────────────────────────────────────────────────

TEST(UndoCommand, RemoveObjectRestoresExactSnapshot) {
    SceneGraph s;
    Material m; m.albedo = {0.9f, 0.2f, 0.4f}; m.roughness = 0.3f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), m);
    s.set_visibility("ball", false);  // edit visibility before remove

    UndoStack u;
    u.push_apply(std::make_unique<RemoveObjectCommand>(s, "ball"));
    EXPECT_EQ(s.objects().count("ball"), 0u);

    u.undo();
    ASSERT_EQ(s.objects().count("ball"), 1u);
    const auto& restored = s.objects().at("ball");
    EXPECT_FLOAT_EQ(restored.material.albedo[0], 0.9f);
    EXPECT_FLOAT_EQ(restored.material.roughness, 0.3f);
    EXPECT_FALSE(restored.visible);  // visibility round-tripped
}

TEST(UndoCommand, SetMaterialUndoRestoresOld) {
    SceneGraph s;
    Material old_m; old_m.albedo = {0.1f, 0.2f, 0.3f};
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), old_m);

    UndoStack u;
    Material new_m; new_m.albedo = {0.9f, 0.8f, 0.7f}; new_m.metallic = 1.0f;
    u.push_apply(std::make_unique<SetMaterialCommand>(s, "ball", new_m));

    EXPECT_FLOAT_EQ(s.objects().at("ball").material.metallic, 1.0f);
    EXPECT_FLOAT_EQ(s.objects().at("ball").material.albedo[0], 0.9f);

    u.undo();
    EXPECT_FLOAT_EQ(s.objects().at("ball").material.metallic, 0.0f);
    EXPECT_FLOAT_EQ(s.objects().at("ball").material.albedo[0], 0.1f);

    u.redo();
    EXPECT_FLOAT_EQ(s.objects().at("ball").material.albedo[0], 0.9f);
}

// One coalesced command reverts a material edit broadcast across a
// multi-selection — every affected object, not just the primary.
TEST(UndoCommand, SetMaterialsCommandRevertsAllObjects) {
    SceneGraph s;
    Material a; a.albedo = {0.1f, 0.1f, 0.1f};
    Material b; b.albedo = {0.2f, 0.2f, 0.2f};
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"), a);
    s.add_object(std::make_shared<SphereNode>(1.0f, "b"), b);

    // Broadcast a new roughness to both, with per-object old/new pairs.
    Material a2 = a; a2.roughness = 0.9f;
    Material b2 = b; b2.roughness = 0.9f;
    std::vector<SetMaterialsCommand::Entry> entries = {
        {"a", a, a2}, {"b", b, b2},
    };

    UndoStack u;
    u.push_apply(std::make_unique<SetMaterialsCommand>(s, entries));
    EXPECT_FLOAT_EQ(s.objects().at("a").material.roughness, 0.9f);
    EXPECT_FLOAT_EQ(s.objects().at("b").material.roughness, 0.9f);

    // One undo restores BOTH objects' baselines.
    u.undo();
    EXPECT_FLOAT_EQ(s.objects().at("a").material.roughness, 0.5f);
    EXPECT_FLOAT_EQ(s.objects().at("b").material.roughness, 0.5f);
    EXPECT_FLOAT_EQ(s.objects().at("a").material.albedo[0], 0.1f);
    EXPECT_FLOAT_EQ(s.objects().at("b").material.albedo[0], 0.2f);

    // One redo re-applies to both.
    u.redo();
    EXPECT_FLOAT_EQ(s.objects().at("a").material.roughness, 0.9f);
    EXPECT_FLOAT_EQ(s.objects().at("b").material.roughness, 0.9f);
}

// Material value-equality: only genuinely changed objects should be
// recorded by the editor's coalescing logic.
TEST(Material, EqualityComparesAllFields) {
    Material m;
    Material n = m;
    EXPECT_TRUE(m == n);
    n.roughness = 0.123f;
    EXPECT_TRUE(m != n);
    n = m; n.albedo2 = {0.4f, 0.5f, 0.6f};
    EXPECT_TRUE(m != n);
    n = m; n.pattern = Material::Pattern::Checker;
    EXPECT_TRUE(m != n);
}

TEST(UndoCommand, SetVisibilityRoundTrips) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    EXPECT_TRUE(s.objects().at("ball").visible);

    UndoStack u;
    u.push_apply(std::make_unique<SetVisibilityCommand>(s, "ball", false));
    EXPECT_FALSE(s.objects().at("ball").visible);
    u.undo();
    EXPECT_TRUE(s.objects().at("ball").visible);
}

TEST(UndoCommand, AddLightUndoRemovesIt) {
    SceneGraph s;
    s.lights().clear();
    UndoStack u;

    PointLight L; L.pos = {3, 4, 5}; L.color = {0.5f, 0.6f, 0.7f}; L.intensity = 0.9f;
    u.push_apply(std::make_unique<AddLightCommand>(s, L));
    ASSERT_EQ(s.lights().size(), 1u);
    EXPECT_FLOAT_EQ(s.lights()[0].intensity, 0.9f);

    u.undo();
    EXPECT_TRUE(s.lights().empty());

    u.redo();
    ASSERT_EQ(s.lights().size(), 1u);
    EXPECT_FLOAT_EQ(s.lights()[0].color[1], 0.6f);
}

TEST(UndoCommand, RemoveLightRestoresAtSameIndex) {
    SceneGraph s;
    auto& Ls = s.lights(); Ls.clear();
    Ls.push_back(PointLight{{1, 0, 0}, {1, 0, 0}, 1.0f});  // red
    Ls.push_back(PointLight{{0, 1, 0}, {0, 1, 0}, 1.0f});  // green
    Ls.push_back(PointLight{{0, 0, 1}, {0, 0, 1}, 1.0f});  // blue

    UndoStack u;
    u.push_apply(std::make_unique<RemoveLightCommand>(s, 1));  // remove green
    ASSERT_EQ(s.lights().size(), 2u);
    EXPECT_FLOAT_EQ(s.lights()[0].color[0], 1.0f);  // red at 0
    EXPECT_FLOAT_EQ(s.lights()[1].color[2], 1.0f);  // blue at 1

    u.undo();
    ASSERT_EQ(s.lights().size(), 3u);
    EXPECT_FLOAT_EQ(s.lights()[1].color[1], 1.0f);  // green back at 1
}

TEST(UndoCommand, SetLightFullyRoundTrips) {
    SceneGraph s;
    auto& Ls = s.lights(); Ls.clear();
    Ls.push_back(PointLight{{0, 0, 0}, {0.5f, 0.5f, 0.5f}, 0.5f});

    UndoStack u;
    PointLight new_val; new_val.pos = {5, 5, 5}; new_val.color = {1, 0.5f, 0};
    new_val.intensity = 0.8f;
    u.push_apply(std::make_unique<SetLightCommand>(s, 0, new_val));
    EXPECT_FLOAT_EQ(s.lights()[0].pos[0], 5.0f);
    EXPECT_FLOAT_EQ(s.lights()[0].color[1], 0.5f);

    u.undo();
    EXPECT_FLOAT_EQ(s.lights()[0].pos[0], 0.0f);
    EXPECT_FLOAT_EQ(s.lights()[0].color[1], 0.5f);  // (was already 0.5)
    EXPECT_FLOAT_EQ(s.lights()[0].intensity, 0.5f);
}

TEST(UndoStack, MultiStepUndoRedoMaintainsConsistency) {
    SceneGraph s;
    UndoStack u;

    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "a", std::make_shared<SphereNode>(1.0f, "a"), Material{}));
    u.push_apply(std::make_unique<AddObjectCommand>(
        s, "b", std::make_shared<SphereNode>(1.0f, "b"), Material{}));
    Material m; m.albedo = {0.9f, 0.1f, 0.1f};
    u.push_apply(std::make_unique<SetMaterialCommand>(s, "a", m));
    u.push_apply(std::make_unique<RemoveObjectCommand>(s, "b"));

    // State: a (red), b removed.
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_FLOAT_EQ(s.objects().at("a").material.albedo[0], 0.9f);

    // Walk back to the start.
    u.undo();  // restore b
    EXPECT_EQ(s.objects().size(), 2u);
    u.undo();  // revert a's material
    EXPECT_FLOAT_EQ(s.objects().at("a").material.albedo[0], 0.8f);
    u.undo();  // remove b
    EXPECT_EQ(s.objects().count("b"), 0u);
    u.undo();  // remove a
    EXPECT_EQ(s.objects().size(), 0u);
    EXPECT_FALSE(u.can_undo());

    // Walk forward again.
    u.redo(); u.redo(); u.redo(); u.redo();
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_FLOAT_EQ(s.objects().at("a").material.albedo[0], 0.9f);
}

// SetGeometryCommand is how node-graph edits (including primitive parameter
// changes like a sphere's radius or a box's dimensions) reach the scene
// undoably — sync_graph_to_scene wraps the rebuilt tree in one. Verify the
// geometry swap round-trips: editing a param produces a new geometry node, and
// undo restores the exact prior node.
TEST(UndoCommand, SetGeometryRoundTrips) {
    SceneGraph s;
    auto orig = std::make_shared<SphereNode>(1.0f, "ball");
    s.add_object(orig, Material{});
    EXPECT_EQ(s.objects().at("ball").geometry.get(), orig.get());

    UndoStack u;
    // Simulate a param edit: the graph rebuilds the node with a new radius.
    auto edited = std::make_shared<SphereNode>(2.5f, "ball");
    u.push_apply(std::make_unique<SetGeometryCommand>(s, "ball", edited));
    EXPECT_EQ(s.objects().at("ball").geometry.get(), edited.get());

    u.undo();
    EXPECT_EQ(s.objects().at("ball").geometry.get(), orig.get());
    u.redo();
    EXPECT_EQ(s.objects().at("ball").geometry.get(), edited.get());
}

TEST(UndoCommand, SetTransformRoundTrips) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), Material{});
    auto t0 = s.get_translation("ball");

    UndoStack u;
    std::array<float, 3> t1{3.0f, -1.0f, 2.0f};
    u.push_apply(std::make_unique<SetTransformCommand>(s, "ball", t1));
    EXPECT_FLOAT_EQ(s.get_translation("ball")[0], 3.0f);

    u.undo();
    EXPECT_FLOAT_EQ(s.get_translation("ball")[0], t0[0]);
    EXPECT_FLOAT_EQ(s.get_translation("ball")[1], t0[1]);
    u.redo();
    EXPECT_FLOAT_EQ(s.get_translation("ball")[0], 3.0f);
}

TEST(UndoCommand, SetRotationAndScaleRoundTrip) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), Material{});

    UndoStack u;
    u.push_apply(std::make_unique<SetRotationCommand>(s, "ball", 1.2f));
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 1.2f);
    u.push_apply(std::make_unique<SetScaleCommand>(s, "ball", 2.0f));
    EXPECT_FLOAT_EQ(s.get_scale("ball"), 2.0f);

    // Two-step undo reverts scale then rotation.
    u.undo();
    EXPECT_FLOAT_EQ(s.get_scale("ball"), 1.0f);
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 1.2f);
    u.undo();
    EXPECT_FLOAT_EQ(s.get_rotation_y("ball"), 0.0f);
}
