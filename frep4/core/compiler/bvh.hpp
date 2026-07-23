#pragma once
// core/compiler/bvh.hpp
//
// Bounding Volume Hierarchy for scene_material acceleration.
//
// The problem: scene_material walks all objects linearly, O(n) per hit
// pixel. With 50+ objects this becomes a bottleneck.
//
// The solution: build a BVH in C++ (median split), then emit the traversal
// in LLVM IR. For a point p, we visit only the BVH nodes whose AABB contains p
// -> on average O(log n) tests instead of O(n).
//
// Structure of the emitted code:
//   - each leaf node tests one object (sdf + albedo)
//   - each internal node does an AABB test -> recurse into children
//   - since LLVM IR has no easy recursion, we unroll the tree statically
//     (the build is known — n is fixed at compile time)

#include "core/frep/node.hpp"
#include "core/frep/scene.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <cstdint>

namespace frep {

// ── BVH node ────────────────────────────────────────────────────────────────
struct BVHNode {
    FRepNode::AABB box;

    // Leaf: object_index >= 0, children = nullptr
    // Internal: object_index = -1, left/right valid
    int                      object_index = -1;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;

    bool is_leaf() const { return object_index >= 0; }
};

// ── BVH builder ─────────────────────────────────────────────────────────────
class BVH {
public:
    // One record per object: geometry + material (albedo + pattern,
    // roughness, metallic) + bounding box.
    struct Entry {
        const FRepNode*       geom;
        std::array<float, 3>  albedo;
        std::array<float, 3>  albedo2;
        Material::Pattern     pattern;
        float                 pattern_scale;
        float                 roughness;
        float                 metallic;
        float                 reflectivity;
        std::vector<std::uint8_t> texture_rgba;
        int                       texture_width  = 0;
        int                       texture_height = 0;
        FRepNode::AABB        box;
    };

    // Builds a BVH from the visible objects of the scene.
    static BVH build(const SceneGraph& scene) {
        BVH bvh;
        for (const auto& [id, obj] : scene.objects()) {
            if (!obj.visible) continue;
            bvh.entries_.push_back({
                obj.geometry.get(),
                obj.material.albedo,
                obj.material.albedo2,
                obj.material.pattern,
                obj.material.pattern_scale,
                obj.material.roughness,
                obj.material.metallic,
                obj.material.reflectivity,
                obj.material.texture_rgba,
                obj.material.texture_width,
                obj.material.texture_height,
                obj.geometry->aabb()
            });
        }
        if (!bvh.entries_.empty()) {
            std::vector<int> indices(bvh.entries_.size());
            for (std::size_t i = 0; i < indices.size(); ++i)
                indices[i] = static_cast<int>(i);
            bvh.root_ = bvh.build_recursive(indices, 0);
        }
        return bvh;
    }

    const BVHNode*           root()    const { return root_.get(); }
    const std::vector<Entry>& entries() const { return entries_; }
    bool                     empty()   const { return entries_.empty(); }

    // Tree depth — for statistics.
    int depth() const { return depth_recursive(root_.get()); }

private:
    std::vector<Entry>       entries_;
    std::unique_ptr<BVHNode> root_;

    static int depth_recursive(const BVHNode* n) {
        if (!n || n->is_leaf()) return 1;
        return 1 + std::max(depth_recursive(n->left.get()),
                            depth_recursive(n->right.get()));
    }

    // Median split along the longest axis.
    std::unique_ptr<BVHNode> build_recursive(std::vector<int>& indices, int axis) {
        auto node = std::make_unique<BVHNode>();

        // Bounding box of all objects in this group
        FRepNode::AABB box = entries_[indices[0]].box;
        for (std::size_t i = 1; i < indices.size(); ++i)
            box = FRepNode::AABB::merge(box, entries_[indices[i]].box);
        node->box = box;

        // Leaf — only 1 object
        if (indices.size() == 1) {
            node->object_index = indices[0];
            return node;
        }

        // Pick the split axis — the longest one
        float ex = box.max_x - box.min_x;
        float ey = box.max_y - box.min_y;
        float ez = box.max_z - box.min_z;
        int   split_axis = 0;
        if (ey > ex && ey >= ez) split_axis = 1;
        else if (ez > ex && ez > ey) split_axis = 2;

        // Sort by the center along the chosen axis
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                const auto& ba = entries_[a].box;
                const auto& bb = entries_[b].box;
                float ca = (split_axis == 0) ? ba.center_x()
                         : (split_axis == 1) ? ba.center_y() : ba.center_z();
                float cb = (split_axis == 0) ? bb.center_x()
                         : (split_axis == 1) ? bb.center_y() : bb.center_z();
                return ca < cb;
            });

        // Median split
        std::size_t mid = indices.size() / 2;
        std::vector<int> left_idx(indices.begin(), indices.begin() + mid);
        std::vector<int> right_idx(indices.begin() + mid, indices.end());

        node->left  = build_recursive(left_idx,  (split_axis + 1) % 3);
        node->right = build_recursive(right_idx, (split_axis + 1) % 3);
        return node;
    }
};

} // namespace frep
