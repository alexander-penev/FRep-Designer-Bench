// tests/test_sparse_sdf.cpp
//
// Tests for SparseSDFOctree and its integration with MeshSDFNode.

#include "core/mesh/sparse_sdf_octree.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/mesh_sdf.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace frep;
using namespace frep::mesh;

namespace {

// Build a small sphere-distance grid as a test fixture.
std::vector<float> sphere_grid(int N, float radius) {
    std::vector<float> g(static_cast<std::size_t>(N) * N * N);
    for (int k = 0; k < N; ++k) {
        float pz = -1.5f + 3.0f * k / (N - 1);
        for (int j = 0; j < N; ++j) {
            float py = -1.5f + 3.0f * j / (N - 1);
            for (int i = 0; i < N; ++i) {
                float px = -1.5f + 3.0f * i / (N - 1);
                g[(k*N + j)*N + i] =
                    std::sqrt(px*px + py*py + pz*pz) - radius;
            }
        }
    }
    return g;
}

} // anon

// ── Basic build ─────────────────────────────────────────────────────────────

TEST(SparseOctree, EmptyOnNonPowerOfTwo) {
    std::vector<float> dummy(7 * 7 * 7);
    auto oct = SparseSDFOctree::build(dummy, 7, 0.1f);
    EXPECT_TRUE(oct.empty());
}

TEST(SparseOctree, EmptyOnSizeMismatch) {
    std::vector<float> dummy(10);
    auto oct = SparseSDFOctree::build(dummy, 8, 0.1f);
    EXPECT_TRUE(oct.empty());
}

TEST(SparseOctree, ZeroToleranceMatchesDense) {
    auto g = sphere_grid(16, 0.5f);
    auto oct = SparseSDFOctree::build(g, 16, 0.0f);
    ASSERT_FALSE(oct.empty());
    EXPECT_EQ(oct.resolution(), 16);

    auto recon = oct.flatten_to_dense();
    ASSERT_EQ(recon.size(), g.size());
    float max_err = 0;
    for (std::size_t i = 0; i < g.size(); ++i)
        max_err = std::max(max_err, std::abs(recon[i] - g[i]));
    // tolerance 0 → every voxel ends up as its own leaf (exact).
    EXPECT_LT(max_err, 1e-6f);
}

TEST(SparseOctree, ReconstructionErrorWithinTolerance) {
    auto g = sphere_grid(32, 0.5f);
    float tol = 0.08f;
    auto oct = SparseSDFOctree::build(g, 32, tol);
    auto recon = oct.flatten_to_dense();

    // Each voxel's reconstructed value should be within `tol` of the
    // original (the leaf-merge criterion is range ≤ tol, so a voxel
    // can differ from the leaf mean by at most tol).
    float max_err = 0;
    for (std::size_t i = 0; i < g.size(); ++i)
        max_err = std::max(max_err, std::abs(recon[i] - g[i]));
    EXPECT_LE(max_err, tol);
}

TEST(SparseOctree, CompressesSmoothFieldAtHighRes) {
    // At 128³ with a meaningful tolerance, an SDF should compress
    // significantly. Anything below 2x is unexpected; we use 64³ for
    // test speed but pick a tolerance that compresses well.
    auto g = sphere_grid(64, 0.5f);
    auto oct = SparseSDFOctree::build(g, 64, 0.1f);
    ASSERT_FALSE(oct.empty());
    EXPECT_LT(oct.bytes(), oct.dense_bytes());
    // Leaves should be fewer than total voxels — that's the whole point.
    EXPECT_LT(oct.leaf_count(),
              static_cast<std::size_t>(64 * 64 * 64));
}

// ── sample() matches flatten_to_dense() output ─────────────────────────────

TEST(SparseOctree, SampleMatchesFlatten) {
    auto g = sphere_grid(16, 0.4f);
    auto oct = SparseSDFOctree::build(g, 16, 0.05f);
    auto recon = oct.flatten_to_dense();

    // Sample a handful of cells.
    for (int k : {0, 5, 10, 15})
        for (int j : {0, 5, 10, 15})
            for (int i : {0, 5, 10, 15}) {
                float a = oct.sample(i, j, k);
                float b = recon[(k*16 + j)*16 + i];
                EXPECT_FLOAT_EQ(a, b)
                    << "at (" << i << "," << j << "," << k << ")";
            }
}

// ── MeshSDFNode integration ────────────────────────────────────────────────

TEST(MeshSDFSparse, ZeroToleranceDisablesSparse) {
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    MarchingCubesParams p; p.rx = p.ry = p.rz = 12;
    auto mesh = extract_iso_mesh(ref, p);

    MeshSDFNode dense(mesh, 32, "ball", 0.0f);
    EXPECT_FALSE(dense.uses_sparse());
    EXPECT_EQ(dense.sparse_bytes(), 0u);
    EXPECT_EQ(dense.sparse_leaves(), 0u);
}

TEST(MeshSDFSparse, RecordsSparseStats) {
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    MarchingCubesParams p; p.rx = p.ry = p.rz = 12;
    auto mesh = extract_iso_mesh(ref, p);

    // Use 32³ at tolerance high enough for some merging. Note: at 32³ the
    // total voxel count is small enough that we don't expect massive
    // compression — but we should at least see leaves < total cells with
    // the chosen tolerance.
    MeshSDFNode sparse(mesh, 32, "ball", 0.15f);
    EXPECT_TRUE(sparse.uses_sparse());
    EXPECT_GT(sparse.sparse_bytes(), 0u);
    EXPECT_GT(sparse.sparse_leaves(), 0u);
    // At tol=0.15 the leaves should be fewer than total cells.
    EXPECT_LT(sparse.sparse_leaves(),
              static_cast<std::size_t>(32 * 32 * 32));
}

TEST(MeshSDFSparse, EvalRemainsCloseToDense) {
    // The sparse pass is lossy by design, but error must stay bounded.
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    MarchingCubesParams p; p.rx = p.ry = p.rz = 16;
    auto mesh = extract_iso_mesh(ref, p);

    MeshSDFNode dense (mesh, 32, "d", 0.0f);
    MeshSDFNode sparse(mesh, 32, "s", 0.05f);

    // Compare eval at a few points.
    float max_err = 0;
    for (float t : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
        float a = dense .eval(t, 0, 0);
        float b = sparse.eval(t, 0, 0);
        max_err = std::max(max_err, std::abs(a - b));
    }
    // Tolerance 0.05 + a small slack for trilinear interp differences.
    EXPECT_LT(max_err, 0.15f);
}
