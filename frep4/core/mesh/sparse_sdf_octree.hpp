#pragma once
// core/mesh/sparse_sdf_octree.hpp
//
// SparseSDFOctree — compresses a dense N×N×N float SDF grid into an
// octree where each node stores either:
//   (a) eight child indices (internal node), OR
//   (b) a single constant value (leaf — every voxel under this node
//       falls within `tolerance` of that value).
//
// Compression strategy:
//   - The dense grid must be power-of-two sized (we pad if needed).
//   - Recursively split into 8 octants until one of:
//       * voxel range (max - min) ≤ tolerance → leaf with mean value
//       * we hit a single voxel → leaf with that value
//   - Internal nodes hold 8 indices into the node array.
//
// Decompression (eval):
//   - Walk the octree from root, picking the octant containing the query.
//   - Returns the leaf's constant value at the resolved leaf.
//   - This is "nearest cell" sampling — slightly less accurate than the
//     trilinear interp of the dense path, but the octree leaf-merging
//     already implies all voxels in a leaf agree to within tolerance,
//     so the error is bounded by `tolerance`.
//
// Conversion back to dense:
//   - flatten_to_dense() reconstructs the original grid by replicating
//     leaf values. Used to keep JIT codegen unchanged (it still consumes
//     a flat [N×N×N float] global). The on-disk / in-RAM cost stays
//     low; only the JIT module is large.
//
// Memory layout: nodes_ is a flat vector. Internal nodes use children[0..7]
// holding indices ≥ 1 into nodes_. A leaf has children[0] == 0; the value
// lives in `value`. We pack {leaf_or_internal, value/children} into a
// single 36-byte struct for simplicity (small octrees stay cache-friendly).

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace frep::mesh {

class SparseSDFOctree {
public:
    // Compact node layout to keep memory low. A node is a single int32:
    //   if value < 0: it's an internal node; `~value` is the index into
    //                 `child_blocks_` (each block holds 8 child indices).
    //   if value ≥ 0: it's a leaf; `value` is the index into `leaf_values_`.
    //
    // This shrinks each node reference to 4 bytes; leaf payloads (floats)
    // and internal payloads (8 child refs) live in separate pooled arrays.
    using NodeRef = std::int32_t;

    // Builds a sparse octree from a dense N×N×N grid (row-major, k*N²+j*N+i).
    // Returns an empty octree if N is not a power of two.
    static SparseSDFOctree build(const std::vector<float>& dense,
                                 int N, float tolerance)
    {
        SparseSDFOctree o;
        o.N_         = N;
        o.tolerance_ = tolerance;
        if (N <= 0 || (N & (N - 1)) != 0) return o;
        if (static_cast<int>(dense.size()) != N * N * N) return o;
        o.dense_view_ = &dense;
        // Reserve a small head-start.
        o.leaf_values_.reserve(N);
        o.child_blocks_.reserve(N);
        o.root_ = o.build_recursive(0, 0, 0, N);
        o.dense_view_ = nullptr;
        return o;
    }

    // Re-creates the dense grid by walking the octree and replicating
    // leaf values. Used to keep JIT codegen unchanged (it still consumes
    // a flat [N×N×N float] global). The on-disk / in-RAM cost stays
    // low; only the JIT module is large.
    std::vector<float> flatten_to_dense() const {
        std::vector<float> out(static_cast<std::size_t>(N_) * N_ * N_, 0.0f);
        if (leaf_values_.empty() && child_blocks_.empty()) return out;
        flatten_recursive(root_, 0, 0, 0, N_, out);
        return out;
    }

    // Sample at integer voxel coordinates (i, j, k). Clamps to grid bounds.
    float sample(int i, int j, int k) const {
        if (leaf_values_.empty() && child_blocks_.empty()) return 0.0f;
        i = std::clamp(i, 0, N_ - 1);
        j = std::clamp(j, 0, N_ - 1);
        k = std::clamp(k, 0, N_ - 1);
        return walk(root_, i, j, k, 0, 0, 0, N_);
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────
    int          resolution()  const { return N_; }
    std::size_t  node_count()  const {
        return leaf_values_.size() + child_blocks_.size();
    }
    std::size_t  leaf_count()  const { return leaf_values_.size(); }
    std::size_t  internal_count() const { return child_blocks_.size(); }
    // Memory footprint in bytes — total payload (leaf floats + child blocks).
    std::size_t  bytes() const {
        return leaf_values_.size() * sizeof(float)
             + child_blocks_.size() * sizeof(ChildBlock);
    }
    std::size_t  dense_bytes() const {
        return static_cast<std::size_t>(N_) * N_ * N_ * sizeof(float);
    }
    float        tolerance() const { return tolerance_; }
    bool         empty() const {
        return leaf_values_.empty() && child_blocks_.empty();
    }

private:
    // 8 child NodeRefs — internal node payload.
    struct ChildBlock {
        std::array<NodeRef, 8> children {};
    };

    static NodeRef make_leaf(std::int32_t leaf_idx) {
        return leaf_idx;  // ≥ 0
    }
    static NodeRef make_internal(std::int32_t block_idx) {
        return ~block_idx;  // negative
    }
    static bool is_leaf(NodeRef r)     { return r >= 0; }
    static std::int32_t leaf_of(NodeRef r)     { return r; }
    static std::int32_t internal_of(NodeRef r) { return ~r; }

    NodeRef build_recursive(int x0, int y0, int z0, int size) {
        // Find min/max/mean over the cube.
        float vmin =  std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();
        float vsum = 0.0f;
        int   n    = 0;
        for (int kz = z0; kz < z0 + size && kz < N_; ++kz) {
            for (int jy = y0; jy < y0 + size && jy < N_; ++jy) {
                for (int ix = x0; ix < x0 + size && ix < N_; ++ix) {
                    float v = (*dense_view_)
                        [(static_cast<std::size_t>(kz) * N_ + jy) * N_ + ix];
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                    vsum += v;
                    ++n;
                }
            }
        }
        if (n == 0) {
            leaf_values_.push_back(0.0f);
            return make_leaf(static_cast<std::int32_t>(leaf_values_.size() - 1));
        }

        // Leaf if size == 1 or range fits the tolerance.
        if (size == 1 || (vmax - vmin) <= tolerance_) {
            leaf_values_.push_back(vsum / static_cast<float>(n));
            return make_leaf(static_cast<std::int32_t>(leaf_values_.size() - 1));
        }

        // Internal: allocate a child block now, build children, then
        // patch the block. We must reserve the block index BEFORE recursing
        // because the recursive call will append to child_blocks_.
        std::int32_t my_block = static_cast<std::int32_t>(child_blocks_.size());
        child_blocks_.emplace_back();

        int half = size / 2;
        std::array<NodeRef, 8> kids;
        for (int oct = 0; oct < 8; ++oct) {
            int dx = (oct & 1) ? half : 0;
            int dy = (oct & 2) ? half : 0;
            int dz = (oct & 4) ? half : 0;
            kids[oct] = build_recursive(x0 + dx, y0 + dy, z0 + dz, half);
        }
        // Write child indices back (vector may have reallocated during
        // recursion, so re-index here).
        child_blocks_[my_block].children = kids;
        return make_internal(my_block);
    }

    void flatten_recursive(NodeRef ref,
                           int x0, int y0, int z0, int size,
                           std::vector<float>& out) const
    {
        if (is_leaf(ref)) {
            float v = leaf_values_[leaf_of(ref)];
            for (int kz = z0; kz < z0 + size && kz < N_; ++kz) {
                for (int jy = y0; jy < y0 + size && jy < N_; ++jy) {
                    for (int ix = x0; ix < x0 + size && ix < N_; ++ix) {
                        out[(static_cast<std::size_t>(kz) * N_ + jy) * N_ + ix]
                            = v;
                    }
                }
            }
            return;
        }
        const auto& blk = child_blocks_[internal_of(ref)];
        int half = size / 2;
        for (int oct = 0; oct < 8; ++oct) {
            int dx = (oct & 1) ? half : 0;
            int dy = (oct & 2) ? half : 0;
            int dz = (oct & 4) ? half : 0;
            flatten_recursive(blk.children[oct],
                              x0 + dx, y0 + dy, z0 + dz, half, out);
        }
    }

    float walk(NodeRef ref, int qx, int qy, int qz,
               int x0, int y0, int z0, int size) const
    {
        if (is_leaf(ref)) return leaf_values_[leaf_of(ref)];
        int half = size / 2;
        int oct = 0;
        if (qx >= x0 + half) oct |= 1;
        if (qy >= y0 + half) oct |= 2;
        if (qz >= z0 + half) oct |= 4;
        const auto& blk = child_blocks_[internal_of(ref)];
        return walk(blk.children[oct], qx, qy, qz,
                    x0 + ((oct & 1) ? half : 0),
                    y0 + ((oct & 2) ? half : 0),
                    z0 + ((oct & 4) ? half : 0),
                    half);
    }

    int                       N_         = 0;
    float                     tolerance_ = 0.0f;
    NodeRef                   root_      = 0;
    std::vector<float>        leaf_values_;
    std::vector<ChildBlock>   child_blocks_;
    const std::vector<float>* dense_view_ = nullptr;  // build-time only
};

} // namespace frep::mesh
