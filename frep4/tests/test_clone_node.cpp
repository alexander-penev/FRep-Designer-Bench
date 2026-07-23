// tests/test_clone_node.cpp
//
// Tests for frep::io::clone_node — the deep-copy primitive behind the
// GUI's Duplicate / Copy-Paste commands. clone_node round-trips a
// subtree through the same JSON machinery as save/load, assigning
// fresh ids so the clone is independent of the original.
//
// What we assert:
//   - the clone is a distinct object (different pointer)
//   - the root id is exactly the requested new id
//   - the structure (kind, children count) matches
//   - parameters are preserved
//   - descendant ids are rewritten (no collision with the original)
//   - mutating the clone's tree doesn't touch the original

#include "core/io/scene_io.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace frep;

TEST(CloneNode, ClonesSinglePrimitive) {
    auto sphere = std::make_shared<SphereNode>(2.5f, "ball");
    auto clone  = io::clone_node(*sphere, "ball_copy");

    ASSERT_TRUE(clone != nullptr);
    EXPECT_NE(clone.get(), sphere.get());          // distinct object
    EXPECT_EQ(clone->id, "ball_copy");             // new root id
    EXPECT_EQ(std::string(clone->type_name()), "Sphere");
    // Radius parameter preserved.
    EXPECT_FLOAT_EQ(clone->params.at("r"), 2.5f);
}

TEST(CloneNode, ClonesCsgTreeWithChildren) {
    // Union of two primitives — exercises recursive child cloning and
    // descendant id rewriting.
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    auto u = std::make_shared<UnionNode>(a, b, "csg");

    auto clone = io::clone_node(*u, "csg_copy");
    ASSERT_TRUE(clone != nullptr);
    EXPECT_EQ(clone->id, "csg_copy");
    EXPECT_EQ(std::string(clone->type_name()), "Union");
    ASSERT_EQ(clone->children.size(), 2u);

    // Children are distinct objects from the originals.
    EXPECT_NE(clone->children[0].get(), a.get());
    EXPECT_NE(clone->children[1].get(), b.get());

    // Descendant ids were rewritten — they must NOT collide with the
    // original child ids (otherwise two objects in one scene would
    // share an id).
    EXPECT_NE(clone->children[0]->id, "a");
    EXPECT_NE(clone->children[1]->id, "b");

    // Types preserved in order.
    EXPECT_EQ(std::string(clone->children[0]->type_name()), "Sphere");
    EXPECT_EQ(std::string(clone->children[1]->type_name()), "Box");
}

TEST(CloneNode, CloneIsIndependentOfOriginal) {
    // Mutating the clone must not affect the source tree.
    auto sphere = std::make_shared<SphereNode>(1.0f, "orig");
    auto clone  = io::clone_node(*sphere, "orig_copy");
    ASSERT_TRUE(clone != nullptr);

    clone->params["r"] = 9.0f;
    EXPECT_FLOAT_EQ(sphere->params.at("r"), 1.0f)   // original untouched
        << "Mutating the clone leaked into the original — not a deep copy.";
}

TEST(CloneNode, ClonesNestedTransforms) {
    // Transform wrapping a primitive — a common shape (translate(sphere)).
    auto sphere = std::make_shared<SphereNode>(1.0f, "s");
    auto xform  = std::make_shared<TranslateNode>(sphere, 1.0f, 2.0f, 3.0f, "xf");

    auto clone = io::clone_node(*xform, "xf_copy");
    ASSERT_TRUE(clone != nullptr);
    EXPECT_EQ(clone->id, "xf_copy");
    EXPECT_EQ(std::string(clone->type_name()), "Translate");
    ASSERT_EQ(clone->children.size(), 1u);
    EXPECT_EQ(std::string(clone->children[0]->type_name()), "Sphere");
    // Translation params preserved.
    EXPECT_FLOAT_EQ(clone->params.at("tx"), 1.0f);
    EXPECT_FLOAT_EQ(clone->params.at("ty"), 2.0f);
    EXPECT_FLOAT_EQ(clone->params.at("tz"), 3.0f);
}
