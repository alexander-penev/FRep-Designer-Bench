#pragma once
// core/frep/mesh_sdf.hpp
//
// MeshSDFNode — a FRepNode whose SDF is sampled from a 3D voxel grid.
//
// Construction: takes a triangle mesh (frep::mesh::Mesh) and a resolution.
// The mesh is voxelized: for every voxel center we compute the unsigned
// distance to the closest triangle, then determine inside/outside by ray
// parity test, producing a signed distance grid stored as float[].
//
// At eval time the SDF is reconstructed via trilinear interpolation of the
// 8 surrounding voxel samples. This is exact-ish (within voxel resolution)
// and runs in O(1) per query — fast enough for sphere-tracing.
//
// Limitations of the voxel-grid approach:
//   - Sharp features below voxel resolution are smoothed away.
//   - Memory is O(N^3): 64^3 = 1 MB, 128^3 = 8 MB, 256^3 = 64 MB.
//   - The SDF is only valid inside the bounding box; outside, eval()
//     returns the unsigned distance to the bbox plus the boundary value
//     (a conservative under-estimate that lets sphere tracing march
//     toward the mesh from outside).
//
// Once MeshSDFNode satisfies the FRepNode interface, every existing CSG
// operation (Union, Intersection, Difference, SmoothUnion, Negate) works
// on it transparently — that is the point of F-Rep.

#include "core/frep/node.hpp"
#include "core/mesh/marching_cubes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace frep {

class MeshSDFNode final : public FRepNode {
public:
    // Voxelizes `mesh` into a `resolution^3` SDF grid. The bounds default
    // to the mesh AABB plus a small margin so the surface is fully covered.
    //
    // If `sparse_tolerance > 0`, the grid is compressed into a sparse
    // octree after voxelization: regions where every voxel agrees within
    // `tolerance` collapse to a single leaf. This trades some accuracy
    // for memory — see SparseSDFOctree for details. Set to 0 (default)
    // to keep the dense grid.
    //
    // Note: JIT codegen always re-expands the data to a dense IR global
    // (sparse-aware codegen would require an in-IR octree walker which
    // is significantly more complex). The sparse representation is the
    // *storage* form — JIT-compiled queries still hit the same dense
    // array. The win is RAM at rest + much smaller pre-JIT footprint.
    MeshSDFNode(const mesh::Mesh& mesh,
                int   resolution       = 64,
                std::string nid        = "mesh",
                float sparse_tolerance = 0.0f);

    const char* type_name() const noexcept override { return "MeshSDF"; }
    AABB        aabb()      const noexcept override {
        return {bmin_[0], bmin_[1], bmin_[2],
                bmax_[0], bmax_[1], bmax_[2]};
    }

    // ── eval (no JIT) ────────────────────────────────────────────────────────
    float eval(float x, float y, float z) const override;

    // ── codegen — embed the grid as a global float array, then trilinearly
    //    interpolate at the query point.
    llvm::Value* codegen(CgCtx& c,
                         llvm::Value* x,
                         llvm::Value* y,
                         llvm::Value* z) const override;

    // ── codegen_grad — uses central differences over the trilinear field.
    DualVal codegen_grad(CgCtx& c,
                         DualVal x,
                         DualVal y,
                         DualVal z) const override;

    std::size_t structural_hash() const noexcept override {
        // Cheap structural id — based on resolution + bbox + a fold of
        // a couple of samples. Enough to drive incremental recompiles.
        std::size_t h = std::hash<int>{}(res_);
        for (int k = 0; k < 3; ++k) {
            h ^= std::hash<float>{}(bmin_[k]) * 31;
            h ^= std::hash<float>{}(bmax_[k]) * 71;
        }
        // Sample a handful of cells to make different meshes differ.
        for (std::size_t i = 0; i < grid_.size(); i += std::max<std::size_t>(1, grid_.size() / 16))
            h ^= std::hash<float>{}(grid_[i]);
        return h;
    }

    // Diagnostic accessors.
    int               resolution() const { return res_; }
    std::size_t       grid_bytes() const { return grid_.size() * sizeof(float); }
    const float*      grid_data()  const { return grid_.data(); }

    // World-space bounding box and voxel-size accessors — used by the
    // GLSL emitter to set up trilinear interpolation parameters and to
    // upload the grid to a GPU storage buffer.
    void              bbox_min(float out[3]) const {
        out[0] = bmin_[0]; out[1] = bmin_[1]; out[2] = bmin_[2];
    }
    void              bbox_max(float out[3]) const {
        out[0] = bmax_[0]; out[1] = bmax_[1]; out[2] = bmax_[2];
    }
    void              cell_size(float out[3]) const {
        out[0] = cell_[0]; out[1] = cell_[1]; out[2] = cell_[2];
    }

    // Sparse-storage diagnostics. When `sparse_tolerance` was 0 at
    // construction these are zero / empty.
    float             sparse_tolerance() const { return sparse_tolerance_; }
    std::size_t       sparse_bytes()     const { return sparse_bytes_; }
    std::size_t       sparse_leaves()    const { return sparse_leaves_; }
    std::size_t       sparse_internal()  const { return sparse_internal_; }
    bool              uses_sparse()      const { return sparse_tolerance_ > 0; }
    // Compression ratio dense/sparse — >1 means we saved RAM at rest.
    float             sparse_ratio()     const {
        return sparse_bytes_ > 0
            ? static_cast<float>(grid_bytes()) / sparse_bytes_
            : 1.0f;
    }

private:
    // Sample the trilinear field in C++ (also used by eval() and as a
    // fallback when codegen is not desired).
    float sample(float x, float y, float z) const;

    int                  res_ = 0;
    float                bmin_[3]{};
    float                bmax_[3]{};
    float                cell_[3]{};       // voxel size per axis
    std::vector<float>   grid_;            // res^3 floats, row-major (k*ny + j)*nx + i

    // Sparse-octree storage diagnostics (populated when ctor's
    // sparse_tolerance > 0). The actual octree is built then immediately
    // flattened back into grid_, so JIT codegen stays unchanged; only
    // the byte counts and savings are recorded here.
    float                sparse_tolerance_ = 0.0f;
    std::size_t          sparse_bytes_     = 0;
    std::size_t          sparse_leaves_    = 0;
    std::size_t          sparse_internal_  = 0;
};

} // namespace frep
