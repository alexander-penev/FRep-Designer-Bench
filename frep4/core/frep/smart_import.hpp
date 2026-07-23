#pragma once
// core/frep/smart_import.hpp
//
// Structure recovery for imported / flattened scenes. A model that arrives as a
// flat union of many transformed copies of the same shape (a colonnade's
// columns, a gear's teeth, a field of identical crystals) carries no sharing:
// every copy is its own sub-tree, so codegen emits it N times and the RTX path
// would build N unrelated BLASes. factor_instances recovers the repetition:
// union members that are a transform of a *structurally identical* sub-tree are
// rewired to reference ONE shared prototype through an InstanceNode. Downstream
// this lets the CPU/GPU codegen emit the prototype once and call it (Level-2
// shared subprogram), and gives the RTX path a natural shared-BLAS instance set.
//
// Only members whose inner geometry is identical *including parameters* are
// merged (grouped by structural_hash), so the rendered field is unchanged — the
// pass is a pure structural rewrite, verifiable by rendering before/after.

#include "core/frep/scene.hpp"
#include "core/frep/instance.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace frep {

namespace detail {

inline void collect_union_members(const FRepNode::Ptr& n,
                                  std::vector<FRepNode::Ptr>& out) {
    if (!n) return;
    if (n->kind == NodeKind::Union)
        for (const auto& c : n->children) collect_union_members(c, out);
    else
        out.push_back(n);
}

inline bool is_transform(const FRepNode& n) {
    switch (n.kind) {
        case NodeKind::Translate: case NodeKind::Scale:
        case NodeKind::RotateX:   case NodeKind::RotateY: case NodeKind::RotateZ:
            return true;
        default: return false;
    }
}

inline int subtree_size(const FRepNode& n) {
    int c = 1;
    for (const auto& ch : n.children) if (ch) c += subtree_size(*ch);
    return c;
}

}  // namespace detail

// Rewrite repeated union members into instances of a shared prototype. Returns
// the number of instances created (0 if nothing repeats). Prototypes are added
// to the scene as hidden objects (the shared "definition"). `min_nodes` skips
// trivially small prototypes for which sharing would not pay off.
inline int factor_instances(SceneGraph& scene, int min_nodes = 1) {
    FRepNode::Ptr root;
    for (const auto& [id, obj] : scene.objects())
        if (obj.visible && obj.geometry) { root = obj.geometry; break; }
    if (!root) return 0;

    std::vector<FRepNode::Ptr> members;
    detail::collect_union_members(root, members);

    // Group transform-wrapped members by the (params-sensitive) hash of their
    // inner geometry: same hash == same shape, safe to share.
    std::unordered_map<std::size_t, std::vector<FRepNode*>> groups;
    for (const auto& m : members)
        if (detail::is_transform(*m) && m->children.size() == 1 && m->children[0])
            groups[m->children[0]->structural_hash()].push_back(m.get());

    int created = 0;
    std::vector<FRepNode::Ptr> protos;
    for (auto& [h, ms] : groups) {
        if (ms.size() < 2) continue;
        if (detail::subtree_size(*ms[0]->children[0]) < min_nodes) continue;

        FRepNode::Ptr proto = ms[0]->children[0];   // canonical shared body
        std::string pid = "proto_" + std::to_string(h);
        proto->id = pid;                             // object key + instance target_id
        protos.push_back(proto);
        for (FRepNode* mt : ms) {
            mt->children[0] = std::make_shared<InstanceNode>(proto, pid);
            ++created;
        }
    }
    for (const auto& proto : protos) {
        scene.add_object(proto);
        scene.set_visibility(proto->id, false);      // a definition, not drawn
    }
    return created;
}

}  // namespace frep
