// tests/test_nodes.cpp
//
// Tests for the FRepNode AST, AD, and structural hashes — Google Test.

#include "core/ad/forward_ad.hpp"
#include "core/compiler/bvh.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/transforms.hpp"
#include "core/io/json.hpp"
#include "core/io/scene_io.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace frep;

// ═════════════════════════════════════════════════════════════════════════════
// Forward-mode AD
// ═════════════════════════════════════════════════════════════════════════════
TEST(DualNumbers, Addition) {
    using D = ad::Dual<float>;
    auto r = D{3.0f, 1.0f} + D{4.0f, 0.0f};
    EXPECT_NEAR(r.val, 7.0f, 1e-5f);
    EXPECT_NEAR(r.dot, 1.0f, 1e-5f);
}

TEST(DualNumbers, MultiplicationProductRule) {
    using D = ad::Dual<float>;
    // d/dx (x * 4) = 4
    auto r = D{3.0f, 1.0f} * D{4.0f, 0.0f};
    EXPECT_NEAR(r.val, 12.0f, 1e-5f);
    EXPECT_NEAR(r.dot, 4.0f, 1e-5f);
}

TEST(DualNumbers, SqrtDerivative) {
    // d/dx sqrt(x) = 1/(2*sqrt(x)) -> at x=9, = 1/6
    auto r = ad::sqrt(ad::Dual<float>{9.0f, 1.0f});
    EXPECT_NEAR(r.val, 3.0f,       1e-4f);
    EXPECT_NEAR(r.dot, 1.0f / 6.0f, 1e-4f);
}

TEST(Gradient, SphereOutwardNormal) {
    // f(x,y,z) = x^2 + y^2 + z^2 - 4    ->   grad(f) / |grad(f)| = unit outward normal
    auto sdf = []<typename T>(T x, T y, T z) {
        using D = std::decay_t<T>;
        return x*x + y*y + z*z - D::cst(4.0f);
    };

    // At (2,0,0): normal = (1,0,0)
    auto n = ad::gradient(sdf, 2.0f, 0.0f, 0.0f);
    EXPECT_NEAR(n[0], 1.0f, 1e-4f);
    EXPECT_NEAR(n[1], 0.0f, 1e-4f);
    EXPECT_NEAR(n[2], 0.0f, 1e-4f);

    // At (0,0,2): normal = (0,0,1)
    auto n2 = ad::gradient(sdf, 0.0f, 0.0f, 2.0f);
    EXPECT_NEAR(n2[2], 1.0f, 1e-4f);
}

// ═════════════════════════════════════════════════════════════════════════════
// Structural hashes — for the incremental cache
// ═════════════════════════════════════════════════════════════════════════════
TEST(StructuralHash, SphereDependsOnRadius) {
    EXPECT_NE(SphereNode(1.0f, "a").structural_hash(),
              SphereNode(2.0f, "b").structural_hash());
}

TEST(StructuralHash, SphereSameRadiusSameHash) {
    EXPECT_EQ(SphereNode(1.5f, "a").structural_hash(),
              SphereNode(1.5f, "b").structural_hash());
}

TEST(StructuralHash, BoxIncludesAllDimensions) {
    EXPECT_NE(BoxNode(1, 1, 1).structural_hash(),
              BoxNode(1, 1, 2).structural_hash());
    EXPECT_NE(BoxNode(1, 1, 1).structural_hash(),
              BoxNode(1, 2, 1).structural_hash());
    EXPECT_NE(BoxNode(1, 1, 1).structural_hash(),
              BoxNode(2, 1, 1).structural_hash());
}

TEST(StructuralHash, UnionDiffersFromChildren) {
    auto s1 = std::make_shared<SphereNode>(1.0f, "s1");
    auto s2 = std::make_shared<SphereNode>(2.0f, "s2");
    UnionNode u(s1, s2);
    EXPECT_NE(u.structural_hash(), s1->structural_hash());
    EXPECT_NE(u.structural_hash(), s2->structural_hash());
}

TEST(StructuralHash, TransformIncludesChild) {
    auto s1 = std::make_shared<SphereNode>(1.0f, "s1");
    auto s2 = std::make_shared<SphereNode>(2.0f, "s2");
    EXPECT_NE(TranslateNode(s1, 1, 0, 0).structural_hash(),
              TranslateNode(s2, 1, 0, 0).structural_hash());
}

TEST(StructuralHash, TransformParamsMatter) {
    auto s = std::make_shared<SphereNode>(1.0f, "s");
    EXPECT_NE(TranslateNode(s, 1, 0, 0).structural_hash(),
              TranslateNode(s, 2, 0, 0).structural_hash());
}

// ═════════════════════════════════════════════════════════════════════════════
// SceneGraph
// ═════════════════════════════════════════════════════════════════════════════
TEST(SceneGraph, AddRemoveDirtyTracking) {
    SceneGraph s;
    EXPECT_TRUE(s.objects().empty());
    EXPECT_FALSE(s.geom_dirty());

    s.add_object(std::make_shared<SphereNode>(1.0f, "x"));
    EXPECT_EQ(s.objects().size(), 1u);
    EXPECT_TRUE(s.geom_dirty());

    s.clear_dirty();
    EXPECT_FALSE(s.geom_dirty());

    s.remove_object("x");
    EXPECT_TRUE(s.objects().empty());
    EXPECT_TRUE(s.geom_dirty());
}

TEST(SceneGraph, HashChangesPerObject) {
    SceneGraph s;
    auto h0 = s.scene_hash();
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    auto h1 = s.scene_hash();
    s.add_object(std::make_shared<SphereNode>(2.0f, "b"));
    auto h2 = s.scene_hash();
    EXPECT_NE(h0, h1);
    EXPECT_NE(h1, h2);
}

// ─── BVH ──────────────────────────────────────────────────────────────────────

TEST(AABB, SphereBox) {
    SphereNode s(2.0f, "s");
    auto box = s.aabb();
    EXPECT_FLOAT_EQ(box.min_x, -2.0f);
    EXPECT_FLOAT_EQ(box.max_x,  2.0f);
    EXPECT_FLOAT_EQ(box.min_y, -2.0f);
    EXPECT_FLOAT_EQ(box.max_z,  2.0f);
}

TEST(AABB, BoxDimensions) {
    BoxNode b(1.0f, 2.0f, 3.0f, "b");
    auto box = b.aabb();
    EXPECT_FLOAT_EQ(box.min_x, -1.0f);
    EXPECT_FLOAT_EQ(box.max_y,  2.0f);
    EXPECT_FLOAT_EQ(box.max_z,  3.0f);
}

TEST(AABB, PlaneInfinite) {
    PlaneNode p(0.0f, 1.0f, 0.0f, 0.0f, "p");
    auto box = p.aabb();
    EXPECT_GT(box.max_x, 1e8f);
    EXPECT_LT(box.min_x, -1e8f);
}

TEST(AABB, TranslateShiftsBox) {
    auto s = std::make_shared<SphereNode>(1.0f, "s");
    TranslateNode t(s, 5.0f, 0.0f, 0.0f, "t");
    auto box = t.aabb();
    EXPECT_FLOAT_EQ(box.min_x, 4.0f);
    EXPECT_FLOAT_EQ(box.max_x, 6.0f);
    EXPECT_FLOAT_EQ(box.min_y, -1.0f);
}

TEST(AABB, UnionMerges) {
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "b"), 10.0f, 0.0f, 0.0f, "tb");
    UnionNode u(a, b);
    auto box = u.aabb();
    EXPECT_FLOAT_EQ(box.min_x, -1.0f);
    EXPECT_FLOAT_EQ(box.max_x, 11.0f);
}

TEST(AABB, IntersectionShrinks) {
    auto big = std::make_shared<SphereNode>(3.0f, "big");
    auto sml = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "sml");
    IntersectionNode i(big, sml);
    auto box = i.aabb();
    // the intersection box is the smaller one
    EXPECT_FLOAT_EQ(box.min_x, -1.0f);
    EXPECT_FLOAT_EQ(box.max_x,  1.0f);
}

TEST(BVH, BuildsFromScene) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "b"), 5.0f, 0.0f, 0.0f, "tb"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "c"), -5.0f, 0.0f, 0.0f, "tc"));

    auto bvh = BVH::build(s);
    EXPECT_FALSE(bvh.empty());
    EXPECT_EQ(bvh.entries().size(), 3u);
    EXPECT_NE(bvh.root(), nullptr);
    // 3 objects -> depth at least 2 (root + two layers)
    EXPECT_GE(bvh.depth(), 2);
}

TEST(BVH, EmptySceneSafe) {
    SceneGraph s;
    auto bvh = BVH::build(s);
    EXPECT_TRUE(bvh.empty());
    EXPECT_EQ(bvh.root(), nullptr);
}

TEST(BVH, LeafCarriesAlbedo) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"),
                 Material{{0.9f, 0.1f, 0.2f}});
    auto bvh = BVH::build(s);
    ASSERT_EQ(bvh.entries().size(), 1u);
    EXPECT_FLOAT_EQ(bvh.entries()[0].albedo[0], 0.9f);
    EXPECT_FLOAT_EQ(bvh.entries()[0].albedo[1], 0.1f);
}

// ─── Scene serialization ──────────────────────────────────────────────────────

TEST(SceneIO, RoundTripSimple) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.5f, "sph1"),
                 Material{{0.9f, 0.2f, 0.1f}});
    s.add_object(std::make_shared<BoxNode>(0.5f, 0.7f, 0.9f, "box1"),
                 Material{{0.1f, 0.8f, 0.3f}});
    s.camera().fov_deg = 42.0f;
    s.camera().position = {1.0f, 2.0f, 3.0f};

    auto text = io::serialize_scene(s);
    auto s2   = io::deserialize_scene(text);

    EXPECT_EQ(s2.objects().size(), 2u);
    EXPECT_FLOAT_EQ(s2.camera().fov_deg, 42.0f);
    EXPECT_FLOAT_EQ(s2.camera().position[0], 1.0f);
    EXPECT_FLOAT_EQ(s2.camera().position[2], 3.0f);

    ASSERT_TRUE(s2.objects().count("sph1"));
    auto& sph = s2.objects().at("sph1");
    EXPECT_EQ(std::string(sph.geometry->type_name()), "Sphere");
    EXPECT_FLOAT_EQ(sph.geometry->params.at("r"), 1.5f);
    EXPECT_FLOAT_EQ(sph.material.albedo[0], 0.9f);
}

TEST(SceneIO, RoundTripNestedTree) {
    SceneGraph s;
    // Translate(Union(Sphere, Box))
    auto sph = std::make_shared<SphereNode>(1.0f, "s");
    auto box = std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b");
    auto un  = std::make_shared<UnionNode>(sph, box, "u");
    auto tr  = std::make_shared<TranslateNode>(un, 2.0f, 0.0f, -1.0f, "t");
    s.add_object(tr, Material{{0.5f, 0.5f, 0.5f}});

    auto text = io::serialize_scene(s);
    auto s2   = io::deserialize_scene(text);

    ASSERT_TRUE(s2.objects().count("t"));
    auto root = s2.objects().at("t").geometry;
    EXPECT_EQ(std::string(root->type_name()), "Translate");
    EXPECT_FLOAT_EQ(root->params.at("tx"), 2.0f);
    ASSERT_EQ(root->children.size(), 1u);

    auto child = root->children[0];
    EXPECT_EQ(std::string(child->type_name()), "Union");
    ASSERT_EQ(child->children.size(), 2u);
    EXPECT_EQ(std::string(child->children[0]->type_name()), "Sphere");
    EXPECT_EQ(std::string(child->children[1]->type_name()), "Box");
}

TEST(SceneIO, RoundTripCustomExpr) {
    SceneGraph s;
    s.add_object(
        std::make_shared<CustomExprNode>("x*x + y*y + z*z - 1.0", "ce"),
        Material{{1.0f, 1.0f, 0.0f}});

    auto text = io::serialize_scene(s);
    auto s2   = io::deserialize_scene(text);

    ASSERT_TRUE(s2.objects().count("ce"));
    auto node = s2.objects().at("ce").geometry;
    EXPECT_EQ(std::string(node->type_name()), "CustomExpr");
    auto& ce = static_cast<const CustomExprNode&>(*node);
    EXPECT_EQ(ce.expression(), "x*x + y*y + z*z - 1.0");
}

TEST(SceneIO, VisibilityPreserved) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "vis"));
    s.add_object(std::make_shared<SphereNode>(1.0f, "hidden"));
    s.set_visibility("hidden", false);

    auto s2 = io::deserialize_scene(io::serialize_scene(s));
    EXPECT_TRUE(s2.objects().at("vis").visible);
    EXPECT_FALSE(s2.objects().at("hidden").visible);
}

TEST(SceneIO, MalformedJsonThrows) {
    EXPECT_THROW(io::deserialize_scene("{ not valid json"), std::runtime_error);
    EXPECT_THROW(io::deserialize_scene("{\"objects\": [{\"geometry\":"
                 "{\"type\":\"UnknownType\"}}]}"), std::runtime_error);
}

TEST(JsonValue, BasicTypes) {
    using namespace frep::json;
    auto v = Value::parse(R"({"a": 1, "b": "text", "c": [1,2,3], "d": true})");
    EXPECT_TRUE(v.is_object());
    EXPECT_EQ(v["a"].as_int(), 1);
    EXPECT_EQ(v["b"].as_string(), "text");
    EXPECT_EQ(v["c"].as_array().size(), 3u);
    EXPECT_TRUE(v["d"].as_bool());
}

// ─── structure_hash — ignores parameter values ────────────────────────────────

TEST(StructureHash, SphereSameRegardlessOfRadius) {
    EXPECT_EQ(SphereNode(1.0f, "a").structure_hash(),
              SphereNode(2.0f, "b").structure_hash());
}

TEST(StructureHash, BoxSameRegardlessOfDimensions) {
    EXPECT_EQ(BoxNode(1, 1, 1).structure_hash(),
              BoxNode(2, 3, 4).structure_hash());
}

TEST(StructureHash, SphereVsBoxDiffer) {
    EXPECT_NE(SphereNode(1.0f, "a").structure_hash(),
              BoxNode(1, 1, 1).structure_hash());
}

TEST(StructureHash, UnionStructureMatchesAcrossParams) {
    auto u1 = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<BoxNode>(1, 1, 1, "b"));
    auto u2 = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(5.0f, "a"),
        std::make_shared<BoxNode>(3, 2, 7, "b"));
    EXPECT_EQ(u1->structure_hash(), u2->structure_hash());
    // But structural_hash (which includes values) must differ.
    EXPECT_NE(u1->structural_hash(), u2->structural_hash());
}

TEST(StructureHash, NestedTransformDiffers) {
    auto base = std::make_shared<SphereNode>(1.0f, "s");
    auto plain    = base;
    auto wrapped  = std::make_shared<TranslateNode>(base, 1, 0, 0);
    EXPECT_NE(plain->structure_hash(), wrapped->structure_hash());
}

TEST(StructureHash, CustomExprDependsOnText) {
    CustomExprNode a("x*x + y*y", "a");
    CustomExprNode b("x + y",     "b");
    CustomExprNode c("x*x + y*y", "c");
    EXPECT_NE(a.structure_hash(), b.structure_hash());
    EXPECT_EQ(a.structure_hash(), c.structure_hash());
}

TEST(StructureHash, SceneGraphAggregates) {
    SceneGraph s1, s2;
    s1.add_object(std::make_shared<SphereNode>(1.0f, "x"));
    s2.add_object(std::make_shared<SphereNode>(7.5f, "x"));  // same id, diff r
    // Different values → different scene_hash, but same structure_hash.
    EXPECT_NE(s1.scene_hash(),     s2.scene_hash());
    EXPECT_EQ(s1.structure_hash(), s2.structure_hash());
}
