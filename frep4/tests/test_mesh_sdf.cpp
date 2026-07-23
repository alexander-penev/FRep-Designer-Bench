// tests/test_mesh_sdf.cpp
//
// Tests for the mesh-import / mesh-as-SDF pipeline:
//   - load_obj / load_stl round-trip with save_obj / save_stl
//   - MeshSDFNode approximates the true SDF of the original primitive
//   - CSG operations work on a MeshSDFNode child

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/mesh/marching_cubes.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <memory>

using namespace frep;
using namespace frep::mesh;

// ── OBJ / STL round-trip ────────────────────────────────────────────────────

TEST(MeshIO, ObjRoundTrip) {
    // Generate a known mesh.
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto orig = extract_iso_mesh(s, p);
    ASSERT_FALSE(orig.vertices.empty());

    std::string path = ::testing::TempDir() + "/rt.obj";
    ASSERT_TRUE(save_obj(orig, path));
    auto loaded = load_obj(path);

    EXPECT_EQ(loaded.vertices.size(), orig.vertices.size());
    EXPECT_EQ(loaded.indices.size(),  orig.indices.size());
    // Verify a few vertices match.
    for (std::size_t i = 0; i < std::min<std::size_t>(10, orig.vertices.size()); ++i) {
        EXPECT_NEAR(loaded.vertices[i].x, orig.vertices[i].x, 1e-3f);
        EXPECT_NEAR(loaded.vertices[i].y, orig.vertices[i].y, 1e-3f);
        EXPECT_NEAR(loaded.vertices[i].z, orig.vertices[i].z, 1e-3f);
    }
}

TEST(MeshIO, StlAsciiRoundTrip) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto orig = extract_iso_mesh(s, p);
    ASSERT_FALSE(orig.vertices.empty());

    std::string path = ::testing::TempDir() + "/rt.stl";
    ASSERT_TRUE(save_stl(orig, path));
    auto loaded = load_stl(path);

    // STL stores 3 vertices per triangle, so vertex count = 3 * tri count.
    EXPECT_EQ(loaded.vertices.size(), orig.indices.size());
    EXPECT_EQ(loaded.indices.size(),  orig.indices.size());
}

TEST(MeshIO, LoadMissingFileEmpty) {
    auto m = load_obj("/nonexistent/does_not_exist.obj");
    EXPECT_TRUE(m.vertices.empty());
    EXPECT_TRUE(m.indices.empty());
}

// ── MeshSDFNode basic ───────────────────────────────────────────────────────

TEST(MeshSDF, ApproximatesSphere) {
    // Generate a sphere mesh, voxelize, and verify eval() approximates
    // the true sphere SDF (length(p) - 1).
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 24;  // higher resolution = smoother mesh
    auto m = extract_iso_mesh(ref, p);
    ASSERT_FALSE(m.vertices.empty());

    MeshSDFNode node(m, 32);

    auto check = [&](float x, float y, float z, float tol) {
        float truth = std::sqrt(x*x + y*y + z*z) - 1.0f;
        float v     = node.eval(x, y, z);
        EXPECT_NEAR(v, truth, tol)
            << "at (" << x << "," << y << "," << z << ")";
    };
    check( 0.0f, 0.0f, 0.0f, 0.15f);  // center, allow voxel-resolution error
    check( 0.5f, 0.0f, 0.0f, 0.05f);  // mid-inside
    check( 1.0f, 0.0f, 0.0f, 0.05f);  // on surface
    check( 1.5f, 0.0f, 0.0f, 0.05f);  // just outside
    check( 0.0f, 0.5f, 0.0f, 0.05f);  // different axis
    check( 3.0f, 0.0f, 0.0f, 0.05f);  // far outside — bbox correction kicks in
}

TEST(MeshSDF, AABBContainsMesh) {
    // The MeshSDFNode's AABB should fully contain the mesh's vertices.
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto m = extract_iso_mesh(ref, p);
    MeshSDFNode node(m, 32);

    auto a = node.aabb();
    for (const auto& v : m.vertices) {
        EXPECT_GE(v.x, a.min_x);
        EXPECT_LE(v.x, a.max_x);
        EXPECT_GE(v.y, a.min_y);
        EXPECT_LE(v.y, a.max_y);
        EXPECT_GE(v.z, a.min_z);
        EXPECT_LE(v.z, a.max_z);
    }
}

TEST(MeshSDF, EmptyMeshDoesNotCrash) {
    Mesh empty;
    MeshSDFNode node(empty, 16);
    // Should produce a "ball of nothingness" — eval is positive everywhere.
    EXPECT_GT(node.eval(0, 0, 0), 0);
    EXPECT_GT(node.eval(5, 5, 5), 0);
}

// ── CSG operations on a MeshSDFNode ─────────────────────────────────────────

TEST(MeshSDF, DifferenceCarvesHole) {
    // sphere - box: at a point well inside the box and inside the sphere,
    // the result should be positive (the box took out that region).
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto m = extract_iso_mesh(ref, p);

    auto mesh_sphere = std::make_shared<MeshSDFNode>(m, 32, "sph");
    auto box = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.4f, 0.4f, 0.4f, "b"),
        0.6f, 0.0f, 0.0f, "bt");
    DifferenceNode diff(mesh_sphere, box, "diff");

    // Inside the sphere only — should remain inside the result.
    EXPECT_LT(diff.eval(-0.5f, 0, 0), 0);

    // Inside both → carved out → should be outside the result.
    EXPECT_GT(diff.eval(0.6f, 0, 0), 0);

    // Outside both — outside.
    EXPECT_GT(diff.eval(3.0f, 0, 0), 0);
}

TEST(MeshSDF, UnionAddsBoth) {
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(0.8f, "ball"));
    MarchingCubesParams p;
    p.rx = p.ry = p.rz = 16;
    auto m = extract_iso_mesh(ref, p);

    auto mesh_sphere = std::make_shared<MeshSDFNode>(m, 32, "sph");
    auto cube = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.4f, 0.4f, 0.4f, "b"),
        2.0f, 0.0f, 0.0f, "bt");
    UnionNode u(mesh_sphere, cube, "u");

    // Inside the sphere only.
    EXPECT_LT(u.eval(0, 0, 0), 0);
    // Inside the cube only.
    EXPECT_LT(u.eval(2.0f, 0, 0), 0);
    // Between them — outside both, outside the union.
    EXPECT_GT(u.eval(1.0f, 0, 0), 0);
}
