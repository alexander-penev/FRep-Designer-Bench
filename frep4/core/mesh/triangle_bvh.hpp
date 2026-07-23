#pragma once
// core/mesh/triangle_bvh.hpp
//
// A binary BVH over a triangle list, used by the MeshSDFNode voxelizer for
// two query types:
//
//   1. closest_distance2(p) — minimum squared distance from p to any
//      triangle. Pruned by min-distance-to-AABB.
//   2. ray_hits(p, dir) — number of triangles a ray from p in `dir` hits.
//      Used for the inside/outside parity test. Pruned by the standard
//      slab method against each node's AABB.
//
// Build:
//   - Top-down recursive split.
//   - Split axis: longest dimension of the centroid bounding box.
//   - Split position: median of centroids along that axis.
//   - Leaves contain ≤ leaf_size triangles (default 8).
//
// This is intentionally minimal — no SAH, no quantization. Sufficient for
// preprocessing meshes up to a few hundred thousand triangles in << 1s.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace frep::mesh_bvh {

struct AABB3 {
    float min_x =  std::numeric_limits<float>::infinity();
    float min_y =  std::numeric_limits<float>::infinity();
    float min_z =  std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    void expand(float x, float y, float z) {
        min_x = std::min(min_x, x); max_x = std::max(max_x, x);
        min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        min_z = std::min(min_z, z); max_z = std::max(max_z, z);
    }
    void expand(const AABB3& o) {
        min_x = std::min(min_x, o.min_x); max_x = std::max(max_x, o.max_x);
        min_y = std::min(min_y, o.min_y); max_y = std::max(max_y, o.max_y);
        min_z = std::min(min_z, o.min_z); max_z = std::max(max_z, o.max_z);
    }

    // Squared distance from a point to the box; 0 if inside.
    float distance2(float px, float py, float pz) const {
        float dx = std::max({min_x - px, 0.0f, px - max_x});
        float dy = std::max({min_y - py, 0.0f, py - max_y});
        float dz = std::max({min_z - pz, 0.0f, pz - max_z});
        return dx*dx + dy*dy + dz*dz;
    }
};

// Internal node layout — flat array, indexable. Negative `left` = leaf,
// then `right` is the count of triangles starting at `first`.
struct Node {
    AABB3      box;
    std::int32_t left  = -1;  // child node index, or -1 if leaf
    std::int32_t right = 0;   // child node index (internal) / tri count (leaf)
    std::int32_t first = 0;   // tri index into the reordered list (leaf only)
    bool is_leaf() const { return left < 0; }
};

// Triangle, stored as three vertices.
struct Tri {
    std::array<float, 3> a, b, c;
    AABB3 box() const {
        AABB3 r;
        r.expand(a[0], a[1], a[2]);
        r.expand(b[0], b[1], b[2]);
        r.expand(c[0], c[1], c[2]);
        return r;
    }
    std::array<float, 3> centroid() const {
        return {(a[0] + b[0] + c[0]) / 3.0f,
                (a[1] + b[1] + c[1]) / 3.0f,
                (a[2] + b[2] + c[2]) / 3.0f};
    }
};

// Squared distance from p to triangle (a, b, c) — same algorithm used by
// MeshSDFNode but duplicated here to keep the BVH self-contained.
inline float point_tri_distance2(const std::array<float,3>& p,
                                 const std::array<float,3>& a,
                                 const std::array<float,3>& b,
                                 const std::array<float,3>& c)
{
    auto sub = [](const auto& u, const auto& v) {
        return std::array<float,3>{u[0]-v[0], u[1]-v[1], u[2]-v[2]};
    };
    auto dot = [](const auto& u, const auto& v) {
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
    };
    auto add_mul = [](const auto& u, const auto& v, float s) {
        return std::array<float,3>{u[0]+v[0]*s, u[1]+v[1]*s, u[2]+v[2]*s};
    };
    auto len2 = [&](const auto& u) { return dot(u, u); };

    auto ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return len2(sub(p, a));

    auto bp = sub(p, b);
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return len2(sub(p, b));

    float vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        return len2(sub(p, add_mul(a, ab, v)));
    }

    auto cp = sub(p, c);
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return len2(sub(p, c));

    float vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float w = d2 / (d2 - d6);
        return len2(sub(p, add_mul(a, ac, w)));
    }

    float va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        auto bc = sub(c, b);
        return len2(sub(p, add_mul(b, bc, w)));
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    return len2(sub(p, add_mul(add_mul(a, ab, v), ac, w)));
}

// Möller-Trumbore. Returns t > epsilon on hit, otherwise -1.
inline float ray_tri_intersect(const std::array<float,3>& o,
                               const std::array<float,3>& d,
                               const std::array<float,3>& v0,
                               const std::array<float,3>& v1,
                               const std::array<float,3>& v2)
{
    const float EPS = 1e-7f;
    std::array<float,3> e1{v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    std::array<float,3> e2{v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    std::array<float,3> pvec{
        d[1]*e2[2] - d[2]*e2[1],
        d[2]*e2[0] - d[0]*e2[2],
        d[0]*e2[1] - d[1]*e2[0]
    };
    float det = e1[0]*pvec[0] + e1[1]*pvec[1] + e1[2]*pvec[2];
    if (std::abs(det) < EPS) return -1.0f;
    float inv_det = 1.0f / det;
    std::array<float,3> tvec{o[0]-v0[0], o[1]-v0[1], o[2]-v0[2]};
    float u = (tvec[0]*pvec[0] + tvec[1]*pvec[1] + tvec[2]*pvec[2]) * inv_det;
    if (u < 0 || u > 1) return -1.0f;
    std::array<float,3> qvec{
        tvec[1]*e1[2] - tvec[2]*e1[1],
        tvec[2]*e1[0] - tvec[0]*e1[2],
        tvec[0]*e1[1] - tvec[1]*e1[0]
    };
    float v = (d[0]*qvec[0] + d[1]*qvec[1] + d[2]*qvec[2]) * inv_det;
    if (v < 0 || u + v > 1) return -1.0f;
    float t = (e2[0]*qvec[0] + e2[1]*qvec[1] + e2[2]*qvec[2]) * inv_det;
    return (t > EPS) ? t : -1.0f;
}

class TriangleBVH {
public:
    // Builds a BVH over the given triangle list. `tris` is consumed
    // (reordered in place to match leaf ranges).
    explicit TriangleBVH(std::vector<Tri> tris, int leaf_size = 8) {
        tris_ = std::move(tris);
        if (tris_.empty()) return;

        // Pre-compute centroids and indices.
        std::vector<std::int32_t> idx(tris_.size());
        for (std::size_t i = 0; i < tris_.size(); ++i)
            idx[i] = static_cast<std::int32_t>(i);

        nodes_.reserve(tris_.size() * 2);  // upper bound for binary tree
        build_recursive(idx, 0, static_cast<int>(idx.size()), leaf_size);

        // Reorder tris_ so leaf ranges are contiguous in tris_.
        std::vector<Tri> reordered;
        reordered.reserve(tris_.size());
        for (auto i : idx) reordered.push_back(tris_[i]);
        tris_ = std::move(reordered);
    }

    bool empty() const { return tris_.empty(); }
    std::size_t triangle_count() const { return tris_.size(); }
    std::size_t node_count()     const { return nodes_.size(); }

    // Closest-distance query. Returns infinity if the BVH is empty.
    float closest_distance2(float px, float py, float pz) const {
        if (nodes_.empty()) return std::numeric_limits<float>::infinity();
        float best = std::numeric_limits<float>::infinity();
        traverse_closest(0, px, py, pz, best);
        return best;
    }

    // Counts triangle hits of a ray from `o` in direction `d`. Used for
    // parity inside/outside test.
    int ray_hits(const std::array<float,3>& o,
                 const std::array<float,3>& d) const
    {
        if (nodes_.empty()) return 0;
        int hits = 0;
        traverse_ray(0, o, d, hits);
        return hits;
    }

private:
    // Builds a subtree over idx[begin..end) and returns the index of
    // the root node it wrote into nodes_.
    std::int32_t build_recursive(std::vector<std::int32_t>& idx,
                                 int begin, int end, int leaf_size)
    {
        std::int32_t my = static_cast<std::int32_t>(nodes_.size());
        nodes_.emplace_back();  // placeholder; fill in below

        Node& node = nodes_[my];
        // AABB over the triangles in this subtree.
        AABB3 box;
        for (int i = begin; i < end; ++i)
            box.expand(tris_[idx[i]].box());
        node.box = box;

        int count = end - begin;
        if (count <= leaf_size) {
            node.left  = -1;
            node.right = count;
            node.first = begin;
            return my;
        }

        // Centroid bounds → pick the longest axis.
        AABB3 cb;
        for (int i = begin; i < end; ++i) {
            auto c = tris_[idx[i]].centroid();
            cb.expand(c[0], c[1], c[2]);
        }
        float ex = cb.max_x - cb.min_x;
        float ey = cb.max_y - cb.min_y;
        float ez = cb.max_z - cb.min_z;
        int axis = 0;
        if (ey > ex && ey >= ez) axis = 1;
        else if (ez > ex && ez >= ey) axis = 2;

        // Median split.
        int mid = (begin + end) / 2;
        std::nth_element(idx.begin() + begin, idx.begin() + mid,
                         idx.begin() + end,
                         [&, axis](std::int32_t a, std::int32_t b) {
                             return tris_[a].centroid()[axis]
                                  < tris_[b].centroid()[axis];
                         });

        // Build children — record their returned indices.
        std::int32_t li = build_recursive(idx, begin, mid, leaf_size);
        std::int32_t ri = build_recursive(idx, mid,   end, leaf_size);
        // nodes_ may have been resized (and our `node` reference invalidated)
        // during recursion — rewrite via index.
        nodes_[my].left  = li;
        nodes_[my].right = ri;
        return my;
    }

    void traverse_closest(std::int32_t ni,
                          float px, float py, float pz, float& best) const
    {
        const Node& n = nodes_[ni];
        if (n.box.distance2(px, py, pz) >= best) return;
        if (n.is_leaf()) {
            for (int i = 0; i < n.right; ++i) {
                const Tri& t = tris_[n.first + i];
                float d2 = point_tri_distance2({px, py, pz}, t.a, t.b, t.c);
                if (d2 < best) best = d2;
            }
            return;
        }
        // Visit the closer child first — better pruning.
        float dl = nodes_[n.left ].box.distance2(px, py, pz);
        float dr = nodes_[n.right].box.distance2(px, py, pz);
        if (dl < dr) {
            traverse_closest(n.left,  px, py, pz, best);
            traverse_closest(n.right, px, py, pz, best);
        } else {
            traverse_closest(n.right, px, py, pz, best);
            traverse_closest(n.left,  px, py, pz, best);
        }
    }

    void traverse_ray(std::int32_t ni,
                      const std::array<float,3>& o,
                      const std::array<float,3>& d,
                      int& hits) const
    {
        const Node& n = nodes_[ni];
        if (!ray_aabb_hit(n.box, o, d)) return;
        if (n.is_leaf()) {
            for (int i = 0; i < n.right; ++i) {
                const Tri& t = tris_[n.first + i];
                float tt = ray_tri_intersect(o, d, t.a, t.b, t.c);
                if (tt > 0) ++hits;
            }
            return;
        }
        traverse_ray(n.left,  o, d, hits);
        traverse_ray(n.right, o, d, hits);
    }

    // Slab method ray-AABB. Direction d is assumed non-zero.
    static bool ray_aabb_hit(const AABB3& bx,
                             const std::array<float,3>& o,
                             const std::array<float,3>& d)
    {
        float tmin = 0.0f;
        float tmax = std::numeric_limits<float>::infinity();
        for (int k = 0; k < 3; ++k) {
            float ok = (k == 0 ? o[0] : k == 1 ? o[1] : o[2]);
            float dk = (k == 0 ? d[0] : k == 1 ? d[1] : d[2]);
            float lo = (k == 0 ? bx.min_x : k == 1 ? bx.min_y : bx.min_z);
            float hi = (k == 0 ? bx.max_x : k == 1 ? bx.max_y : bx.max_z);
            if (std::abs(dk) < 1e-20f) {
                if (ok < lo || ok > hi) return false;
            } else {
                float t1 = (lo - ok) / dk;
                float t2 = (hi - ok) / dk;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return false;
            }
        }
        return true;
    }

    std::vector<Tri>  tris_;
    std::vector<Node> nodes_;
};

} // namespace frep::mesh_bvh
