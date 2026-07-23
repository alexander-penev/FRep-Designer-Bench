// tests/test_scene_set_geometry.cpp
//
// Tests SceneGraph::set_geometry — the in-place geometry replacement
// added to fix the node-editor / scene desynchronization bug.
//
// Previously, the GUI's only way to apply a node graph change was to
// wipe the entire scene and add a single object — which destroyed any
// objects the user had created via the toolbar, inspector, or
// expression editor. set_geometry lets us replace just the active
// object's tree without touching the rest of the scene.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace frep;

TEST(SceneSetGeometry, ReplacesActiveObjectOnly) {
    SceneGraph s;
    Material red{{1, 0, 0}};
    Material green{{0, 1, 0}};
    Material blue{{0, 0, 1}};

    s.add_object(std::make_shared<SphereNode>(0.5f, "a"), red);
    s.add_object(std::make_shared<BoxNode>(0.3f, 0.3f, 0.3f, "b"), green);
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "c"), blue);

    EXPECT_EQ(s.objects().size(), 3u);

    // Replace the geometry of "b" — sphere instead of box.
    s.set_geometry("b", std::make_shared<SphereNode>(0.7f, "b"));

    EXPECT_EQ(s.objects().size(), 3u);  // still 3 objects
    // "a" untouched.
    EXPECT_EQ(s.objects().at("a").geometry->kind, NodeKind::Sphere);
    EXPECT_FLOAT_EQ(s.objects().at("a").geometry->params.at("r"), 0.5f);
    EXPECT_FLOAT_EQ(s.objects().at("a").material.albedo[0], 1.0f);
    // "b" got new geometry but kept its material.
    EXPECT_EQ(s.objects().at("b").geometry->kind, NodeKind::Sphere);
    EXPECT_FLOAT_EQ(s.objects().at("b").geometry->params.at("r"), 0.7f);
    EXPECT_FLOAT_EQ(s.objects().at("b").material.albedo[1], 1.0f);  // green
    // "c" untouched.
    EXPECT_EQ(s.objects().at("c").geometry->kind, NodeKind::Plane);
    EXPECT_FLOAT_EQ(s.objects().at("c").material.albedo[2], 1.0f);  // blue
}

TEST(SceneSetGeometry, MissingIdIsNoop) {
    // Calling set_geometry with an id that doesn't exist must not add
    // a new object or throw — just no-op silently. This matches the
    // semantics of set_material and set_visibility.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.5f, "real"));
    EXPECT_EQ(s.objects().size(), 1u);

    s.set_geometry("ghost", std::make_shared<BoxNode>(1, 1, 1, "ghost"));
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_FALSE(s.objects().count("ghost"));
}

TEST(SceneSetGeometry, MarksGeometryDirty) {
    // Replacing geometry must mark the scene dirty so the viewport
    // recompiles. Otherwise the on-screen render won't update after a
    // node graph edit.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.5f, "a"));
    s.clear_dirty();
    EXPECT_FALSE(s.geom_dirty());

    s.set_geometry("a", std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "a"));
    EXPECT_TRUE(s.geom_dirty());
}

TEST(SceneSetGeometry, CanReplaceWithTreeOfOtherShape) {
    // The replacement isn't restricted to the same node kind — the
    // node graph editor often replaces a primitive with a CSG tree.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.5f, "a"));

    auto tree = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(0.4f, "u_l"),
        std::make_shared<BoxNode>(0.3f, 0.3f, 0.3f, "u_r"),
        "a");
    s.set_geometry("a", tree);

    EXPECT_EQ(s.objects().at("a").geometry->kind, NodeKind::Union);
    EXPECT_EQ(s.objects().at("a").geometry->children.size(), 2u);
}
