// tests/test_triangle_bvh.cpp
//
// Unit tests for the TriangleBVH used by MeshSDF voxelization.

#include "core/mesh/triangle_bvh.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace frep::mesh_bvh;

// Brute-force ground-truth implementations for comparison.
namespace {

float brute_closest_distance2(const std::vector<Tri>& tris,
                              float px, float py, float pz)
{
    float best = std::numeric_limits<float>::infinity();
    for (const auto& t : tris) {
        float d2 = point_tri_distance2({px, py, pz}, t.a, t.b, t.c);
        if (d2 < best) best = d2;
    }
    return best;
}

int brute_ray_hits(const std::vector<Tri>& tris,
                   const std::array<float,3>& o,
                   const std::array<float,3>& d)
{
    int hits = 0;
    for (const auto& t : tris) {
        if (ray_tri_intersect(o, d, t.a, t.b, t.c) > 0) ++hits;
    }
    return hits;
}

} // anonymous

TEST(TriangleBVH, EmptyBvh) {
    TriangleBVH bvh({});
    EXPECT_TRUE(bvh.empty());
    EXPECT_EQ(bvh.triangle_count(), 0u);
    EXPECT_EQ(bvh.node_count(), 0u);
    EXPECT_EQ(std::isinf(bvh.closest_distance2(0, 0, 0)), true);
    EXPECT_EQ(bvh.ray_hits({0, 0, 0}, {1, 0, 0}), 0);
}

TEST(TriangleBVH, SingleTriangleClosest) {
    std::vector<Tri> tris = {
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
    };
    TriangleBVH bvh(tris);
    // A point at (0, 0, 1) — closest point is (0, 0, 0), distance² = 1.
    EXPECT_NEAR(bvh.closest_distance2(0, 0, 1), 1.0f, 1e-5f);
    // A point inside the triangle plane region — z² = 0.04.
    EXPECT_NEAR(bvh.closest_distance2(0.25f, 0.25f, 0.2f), 0.04f, 1e-5f);
}

TEST(TriangleBVH, MatchesBruteForceOnRandomMesh) {
    // Use a sphere mesh — non-trivial topology.
    frep::SceneGraph ref;
    ref.add_object(std::make_shared<frep::SphereNode>(1.0f));
    frep::mesh::MarchingCubesParams p;
    p.rx = p.ry = p.rz = 12;
    auto m = frep::mesh::extract_iso_mesh(ref, p);
    ASSERT_FALSE(m.vertices.empty());

    std::vector<Tri> tris;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const auto& a = m.vertices[m.indices[t  ]];
        const auto& b = m.vertices[m.indices[t+1]];
        const auto& c = m.vertices[m.indices[t+2]];
        tris.push_back({{a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}});
    }
    auto tris_for_brute = tris;  // BVH ctor reorders, keep a copy
    TriangleBVH bvh(std::move(tris));

    // Test closest-distance at 30 random query points.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < 30; ++i) {
        float px = dist(rng), py = dist(rng), pz = dist(rng);
        float ours  = bvh.closest_distance2(px, py, pz);
        float truth = brute_closest_distance2(tris_for_brute, px, py, pz);
        EXPECT_NEAR(ours, truth, 1e-4f)
            << "at (" << px << "," << py << "," << pz << ")";
    }
}

TEST(TriangleBVH, RayHitsMatchBruteForce) {
    frep::SceneGraph ref;
    ref.add_object(std::make_shared<frep::SphereNode>(1.0f));
    frep::mesh::MarchingCubesParams p;
    p.rx = p.ry = p.rz = 12;
    auto m = frep::mesh::extract_iso_mesh(ref, p);

    std::vector<Tri> tris;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const auto& a = m.vertices[m.indices[t  ]];
        const auto& b = m.vertices[m.indices[t+1]];
        const auto& c = m.vertices[m.indices[t+2]];
        tris.push_back({{a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}});
    }
    auto tris_brute = tris;
    TriangleBVH bvh(std::move(tris));

    // For a closed sphere mesh, a +X ray from any inside point should hit
    // an odd number of triangles; from outside, an even number.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < 10; ++i) {
        float px = dist(rng), py = dist(rng), pz = dist(rng);
        int ours  = bvh.ray_hits({px, py, pz}, {1, 0, 0});
        int truth = brute_ray_hits(tris_brute, {px, py, pz}, {1, 0, 0});
        EXPECT_EQ(ours, truth) << "at (" << px << "," << py << "," << pz << ")";
        // And inside the sphere → odd parity.
        EXPECT_EQ(ours & 1, 1) << "(should be inside, odd hits)";
    }

    // Far-outside points: parity should be even.
    std::uniform_real_distribution<float> outer(2.0f, 5.0f);
    for (int i = 0; i < 10; ++i) {
        float px = outer(rng), py = dist(rng), pz = dist(rng);
        int ours = bvh.ray_hits({px, py, pz}, {1, 0, 0});
        EXPECT_EQ(ours & 1, 0) << "at (" << px << "," << py << "," << pz << ")";
    }
}

TEST(TriangleBVH, LeafSizeAffectsTreeShape) {
    // With many triangles, the default leaf size 8 produces multiple nodes.
    std::vector<Tri> tris;
    for (int i = 0; i < 50; ++i) {
        float x = static_cast<float>(i);
        tris.push_back({{x, 0, 0}, {x+1, 0, 0}, {x, 1, 0}});
    }
    TriangleBVH bvh_small(tris, 4);
    TriangleBVH bvh_large(tris, 50);
    EXPECT_GT(bvh_small.node_count(), bvh_large.node_count());
    // With leaf_size >= triangle count, a single leaf suffices.
    EXPECT_EQ(bvh_large.node_count(), 1u);
}
