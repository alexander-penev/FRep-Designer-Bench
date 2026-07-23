// tests/test_mesh.cpp
//
// Tests for marching cubes mesh extraction.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/mesh/marching_cubes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <string>

using namespace frep;
using namespace frep::mesh;

// ── FRepNode::eval — basic correctness ──────────────────────────────────────

TEST(FRepEval, SphereMatchesSDF) {
    SphereNode s(1.0f);
    EXPECT_NEAR(s.eval(0, 0, 0), -1.0f, 1e-6f);
    EXPECT_NEAR(s.eval(1, 0, 0),  0.0f, 1e-6f);
    EXPECT_NEAR(s.eval(0, 0, 2),  1.0f, 1e-6f);
}

TEST(FRepEval, BoxMatchesSDF) {
    BoxNode b(1.0f, 2.0f, 3.0f);
    EXPECT_NEAR(b.eval(0, 0, 0), -1.0f, 1e-6f);    // most-inside dim is X
    EXPECT_NEAR(b.eval(1, 0, 0),  0.0f, 1e-6f);    // face
    EXPECT_NEAR(b.eval(2, 0, 0),  1.0f, 1e-6f);    // outside +x
}

TEST(FRepEval, UnionOfTwoSpheres) {
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<SphereNode>(1.0f, "b");
    auto bt = std::make_shared<TranslateNode>(b, 4.0f, 0, 0, "bt");
    UnionNode u(a, bt);
    // Inside a at origin.
    EXPECT_NEAR(u.eval(0, 0, 0), -1.0f, 1e-5f);
    // Inside b at x=4.
    EXPECT_NEAR(u.eval(4, 0, 0), -1.0f, 1e-5f);
    // Between them — outside both.
    EXPECT_GT(u.eval(2, 0, 0), 0);
}

TEST(FRepEval, IntersectionShrinks) {
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b");
    IntersectionNode i(a, b);
    // Inside both.
    EXPECT_LT(i.eval(0, 0, 0), 0);
    // Outside box but inside sphere — must be outside intersection.
    EXPECT_GT(i.eval(0.9f, 0, 0), 0);
}

TEST(FRepEval, ScaleScales) {
    auto s = std::make_shared<SphereNode>(1.0f, "s");
    ScaleNode sc(s, 2.0f);
    // r=1 scaled by 2 becomes effective r=2.
    EXPECT_NEAR(sc.eval(0, 0, 0), -2.0f, 1e-5f);
    EXPECT_NEAR(sc.eval(2, 0, 0),  0.0f, 1e-5f);
}

// ── Marching cubes — surface extraction ─────────────────────────────────────

TEST(MarchingCubes, SphereProducesClosedMesh) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));

    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 32;
    auto mesh = extract_iso_mesh(s, p);

    // A sphere is closed → should produce many triangles.
    EXPECT_GT(mesh.indices.size() / 3, 100u);
    EXPECT_EQ(mesh.indices.size() % 3, 0u);  // index count multiple of 3

    // All vertices must be finite and roughly on the unit sphere.
    int near_surface = 0;
    for (const auto& v : mesh.vertices) {
        ASSERT_TRUE(std::isfinite(v.x + v.y + v.z));
        float d = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        // Sampling is 32^3 over [-1.2, 1.2]^3 (auto_bounds) → cell size ~0.075.
        // Triangle vertices should be within one cell of the true surface.
        if (std::abs(d - 1.0f) < 0.08f) ++near_surface;
    }
    // The bulk should be on the surface (allow some slack for cell error).
    EXPECT_GT(near_surface, static_cast<int>(mesh.vertices.size()) * 8 / 10);
}

TEST(MarchingCubes, EmptySceneEmptyMesh) {
    SceneGraph s;  // no objects added
    auto mesh = extract_iso_mesh(s);
    EXPECT_TRUE(mesh.vertices.empty());
    EXPECT_TRUE(mesh.indices.empty());
}

TEST(MarchingCubes, AllOutsideEmptyMesh) {
    // Sphere of radius 0.1 inside a bounds box at [10,12] — surface never
    // intersects the sampling region.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(0.1f, "tiny"));
    MarchingCubesParams p;
    p.auto_bounds = false;
    p.bmin[0] = 10; p.bmin[1] = 10; p.bmin[2] = 10;
    p.bmax[0] = 12; p.bmax[1] = 12; p.bmax[2] = 12;
    p.rx = p.ry = p.rz = 8;
    auto mesh = extract_iso_mesh(s, p);
    EXPECT_TRUE(mesh.vertices.empty());
}

TEST(MarchingCubes, BoxHasCorrectExtent) {
    SceneGraph s;
    s.add_object(std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "cube"));

    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 32;
    auto mesh = extract_iso_mesh(s, p);
    ASSERT_FALSE(mesh.vertices.empty());

    // Extent should be roughly [-1, 1] on each axis.
    float minx = 1e9f, maxx = -1e9f;
    float miny = 1e9f, maxy = -1e9f;
    float minz = 1e9f, maxz = -1e9f;
    for (const auto& v : mesh.vertices) {
        minx = std::min(minx, v.x); maxx = std::max(maxx, v.x);
        miny = std::min(miny, v.y); maxy = std::max(maxy, v.y);
        minz = std::min(minz, v.z); maxz = std::max(maxz, v.z);
    }
    EXPECT_NEAR(minx, -1.0f, 0.1f);
    EXPECT_NEAR(maxx,  1.0f, 0.1f);
    EXPECT_NEAR(miny, -1.0f, 0.1f);
    EXPECT_NEAR(maxy,  1.0f, 0.1f);
    EXPECT_NEAR(minz, -1.0f, 0.1f);
    EXPECT_NEAR(maxz,  1.0f, 0.1f);
}

TEST(MarchingCubes, OBJSerializationRoundTrip) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto mesh = extract_iso_mesh(s, p);
    ASSERT_FALSE(mesh.vertices.empty());

    std::string path = ::testing::TempDir() + "/mc_test.obj";
    ASSERT_TRUE(save_obj(mesh, path));

    // Reload as plain text — verify "v" lines == vertex count and "f" lines
    // == triangle count.
    std::ifstream f(path);
    ASSERT_TRUE(f);
    int v_count = 0, f_count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("v ", 0) == 0) ++v_count;
        else if (line.rfind("f ", 0) == 0) ++f_count;
    }
    EXPECT_EQ(static_cast<std::size_t>(v_count), mesh.vertices.size());
    EXPECT_EQ(static_cast<std::size_t>(f_count), mesh.indices.size() / 3);
}
