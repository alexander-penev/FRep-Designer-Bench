#pragma once
// core/compiler/scene_bindings.hpp
//
// Adapter from a live SceneGraph / FRepNode forest to the LLVM-free
// ParamBindingTable. Builds one table (the shared slot authority) for an
// entire scene under a given CompilePolicy, so every backend that renders the
// scene reads the same runtime-buffer layout.

#include "core/compiler/param_binding_table.hpp"
#include "core/compiler/compile_policy.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/node.hpp"

namespace frep {

// Recursively convert an FRepNode into a ParamBindingTable::NodeView. Children
// by value — scene trees are tiny, so the copy is negligible and keeps the
// table builder free of any node.hpp (LLVM) dependency.
inline ParamBindingTable::NodeView to_view(const FRepNode& n) {
    ParamBindingTable::NodeView v;
    v.kind   = static_cast<int>(n.kind);
    v.id     = n.id;
    v.params = &n.params;
    v.children.reserve(n.children.size());
    for (const auto& c : n.children)
        if (c) v.children.push_back(to_view(*c));
    return v;
}

// Build the shared binding table for a whole scene under `policy`. A synthetic
// Scene root holds every visible object's geometry tree, so all parameters get
// slots in one deterministic pass and the layout is identical across backends.
inline ParamBindingTable build_bindings(const SceneGraph& scene,
                                        const CompilePolicy& policy) {
    ParamBindingTable::NodeView root;
    root.kind = pk::Scene;
    root.id   = "$scene";
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        root.children.push_back(to_view(*obj.geometry));
    }
    return ParamBindingTable::build(root, policy);
}

} // namespace frep
