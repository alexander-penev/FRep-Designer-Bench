// tests/test_scene_bvh.cpp
//
// The scene BVH must return exactly the brute-force min() distance — it's
// an acceleration structure, never an approximation. These tests check
// that equivalence across primitive types, spread and overlapping
// layouts, unbounded objects (planes), and edge cases (empty, single).

#include <gtest/gtest.h>

#include "core/accel/bvh.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"

#include <cmath>
#include <random>

using namespace frep;

namespace {

// Brute-force reference: min over every visible object's eval().
float brute_distance(const SceneGraph& s, float x, float y, float z) {
    float d = 1e30f;
    for (const auto& [id, obj] : s.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        d = std::min(d, obj.geometry->eval(x, y, z));
    }
    return d;
}

// Assert BVH == brute force over a cloud of random query points.
void expect_matches_brute(const SceneGraph& s, float extent = 20.0f) {
    accel::Bvh bvh;
    bvh.build(s);
    std::mt19937 rng(20260601);
    std::uniform_real_distribution<float> U(-extent, extent);
    for (int i = 0; i < 5000; ++i) {
        float x = U(rng), y = U(rng), z = U(rng);
        float a = bvh.distance(x, y, z);
        float b = brute_distance(s, x, y, z);
        ASSERT_NEAR(a, b, 1e-3f)
            << "BVH != brute at (" << x << "," << y << "," << z << ")";
    }
}

SceneGraph grid_spheres(int n, float spacing = 3.0f) {
    SceneGraph s;
    int side = std::max(1, (int)std::ceil(std::cbrt((double)n)));
    float off = (side - 1) * spacing * 0.5f;
    int m = 0;
    for (int i = 0; i < side && m < n; ++i)
      for (int j = 0; j < side && m < n; ++j)
        for (int k = 0; k < side && m < n; ++k, ++m)
            s.add_object(std::make_shared<TranslateNode>(
                std::make_shared<SphereNode>(0.5f, "s" + std::to_string(m)),
                i*spacing-off, j*spacing-off, k*spacing-off,
                "t" + std::to_string(m)));
    return s;
}

} // namespace

TEST(SceneBvh, EmptySceneIsFarAway) {
    SceneGraph s;
    accel::Bvh bvh;
    bvh.build(s);
    EXPECT_TRUE(bvh.empty());
    EXPECT_GT(bvh.distance(0, 0, 0), 1e29f);
}

TEST(SceneBvh, SingleObjectMatchesEval) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    accel::Bvh bvh;
    bvh.build(s);
    EXPECT_NEAR(bvh.distance(2, 0, 0), 1.0f, 1e-4f);
    EXPECT_NEAR(bvh.distance(0, 0, 0), -1.0f, 1e-4f);
}

TEST(SceneBvh, SpreadSpheresMatchBrute) {
    expect_matches_brute(grid_spheres(100, 3.0f));
}

TEST(SceneBvh, OverlappingSpheresMatchBrute) {
    // Tight spacing → boxes overlap → less pruning, but result must still
    // be exact.
    expect_matches_brute(grid_spheres(64, 0.6f), 10.0f);
}

TEST(SceneBvh, MixedPrimitivesMatchBrute) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.8f, "sph"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "bx"), 2, 0, 0, "tb"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.6f, "s2"), -2, 1, 0, "ts"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.4f, 0.9f, 0.4f, "b2"), 0, 0, 2.5f, "tb2"));
    expect_matches_brute(s);
}

TEST(SceneBvh, UnboundedPlaneAlwaysEvaluated) {
    // A plane has an infinite AABB; it must go to the always-evaluated
    // list and still contribute to the min.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 2.0f, "floor"));
    accel::Bvh bvh;
    bvh.build(s);
    EXPECT_EQ(bvh.infinite_count(), 1u);
    expect_matches_brute(s);
    // Far below everything, the plane dominates.
    EXPECT_NEAR(bvh.distance(0, -10, 0), brute_distance(s, 0, -10, 0), 1e-3f);
}

TEST(SceneBvh, CsgObjectsTreatedAsUnits) {
    // Each top-level object may itself be a CSG tree; the BVH bounds the
    // whole object and evaluates it as a unit.
    SceneGraph s;
    auto csg = std::make_shared<DifferenceNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b"), 0.5f, 0, 0, "tb"),
        "diff");
    s.add_object(csg);
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "c"), 3, 0, 0, "tc"));
    expect_matches_brute(s);
}

TEST(SceneBvh, NodeCountIsLinear) {
    // A binary BVH over N leaves has 2N-1 nodes; sanity-check we didn't
    // build something pathological.
    SceneGraph s = grid_spheres(100, 3.0f);
    accel::Bvh bvh;
    bvh.build(s);
    EXPECT_EQ(bvh.object_count(), 100u);
    EXPECT_LE(bvh.node_count(), 2u * 100u);
    EXPECT_GE(bvh.node_count(), 100u);
}

TEST(SceneBvh, GpuNodeLayoutRoundTrips) {
    // The std430 GpuNode array must encode each node's box + child/obj indices
    // so a shader can bit-cast the integer lanes back. Verify the packing
    // matches the CPU nodes exactly.
    SceneGraph s = grid_spheres(16, 3.0f);
    accel::Bvh bvh;
    bvh.build(s);

    auto gpu = bvh.gpu_nodes();
    ASSERT_EQ(gpu.size(), bvh.nodes().size());

    auto as_int = [](float f) { std::int32_t i; std::memcpy(&i, &f, 4); return i; };
    for (std::size_t k = 0; k < gpu.size(); ++k) {
        const auto& n = bvh.nodes()[k];
        const auto& g = gpu[k];
        EXPECT_FLOAT_EQ(g.lane0[0], n.box.min_x);
        EXPECT_FLOAT_EQ(g.lane0[1], n.box.min_y);
        EXPECT_FLOAT_EQ(g.lane0[2], n.box.min_z);
        EXPECT_EQ(as_int(g.lane0[3]), n.left);
        EXPECT_FLOAT_EQ(g.lane1[0], n.box.max_x);
        EXPECT_FLOAT_EQ(g.lane1[1], n.box.max_y);
        EXPECT_FLOAT_EQ(g.lane1[2], n.box.max_z);
        EXPECT_EQ(as_int(g.lane1[3]), n.right);
        EXPECT_EQ(as_int(g.lane2[0]), n.obj);
    }
    // Flat-float view is the same bytes.
    auto flats = bvh.gpu_node_floats();
    EXPECT_EQ(flats.size(), gpu.size() * 12);
    EXPECT_EQ(std::memcmp(flats.data(), gpu.data(), flats.size()*4), 0);
}

TEST(SceneBvh, GpuNodeIs48Bytes) {
    // Stride must be 48 (three tightly-packed vec4) for std430 indexing.
    EXPECT_EQ(sizeof(accel::Bvh::GpuNode), 48u);
}
