// InstanceNode — a lightweight reference to another object's geometry, not a
// copy. In FRep terms: the target's SDF is a function already defined in the
// scene, and an instance is a *call* to that function rather than a duplicated
// body. Multiple instances of the same target share one geometry subtree
// (children[0] holds the *same* shared_ptr as the original), so:
//   * editing the target's nodes in place is seen live by every instance
//     (this is a live geometric modeller, not a static-scene loader);
//   * an instance carries only its own placement — the usual Translate/Rotate/
//     Scale wrap it externally (semantics: the instance references the target's
//     *bare* geometry, applying its own transform in full).
//
// Level 1 (this file): correct sharing + delegation semantics. codegen still
// inlines the shared subtree at each instance, so the emitted-code memory does
// not yet shrink — that is Level 2 (emit the target as one GLSL subprogram and
// call it), for which the shared children[0] pointer is exactly the identity a
// node-pointer memo will key on.
//
// target_id is kept for serialization and for re-resolving the pointer after a
// structural swap (SetGeometryCommand replaces a root pointer rather than
// mutating it); resolve_instances() below rebinds children[0] from the scene.
#pragma once
#include "core/frep/node.hpp"
#include <string>
#include <utility>

namespace frep {

class InstanceNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Instance"; }
    std::string target_id_;
public:
    // `target` is the shared geometry root of the referenced object; target_id
    // is that object's scene id (for save/load + re-resolution).
    InstanceNode(FRepNode::Ptr target, std::string target_id, std::string nid = "inst") {
        kind = NodeKind::Instance;
        id = std::move(nid);
        target_id_ = std::move(target_id);
        children = { std::move(target) };   // shared, NOT cloned
    }

    const std::string& target_id() const noexcept { return target_id_; }
    void set_target_id(std::string t) { target_id_ = std::move(t); }
    bool resolved() const noexcept { return !children.empty() && children[0] != nullptr; }

    // Rebind the shared target pointer (after re-resolution against the scene).
    void rebind(FRepNode::Ptr target) {
        if (children.empty()) children.resize(1);
        children[0] = std::move(target);
    }

    // ── delegation: an instance evaluates exactly as its target ──────────────
    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        if (!resolved()) return c.fc(1e30f);           // dangling -> empty (far)
        // Level 2: if the context provides a shared-function callback, emit a
        // call to the target's shared subprogram instead of inlining it.
        if (c.instance_call) {
            if (llvm::Value* v = c.instance_call(children[0].get(), x, y, z))
                return v;
        }
        return children[0]->codegen(c, x, y, z);       // Level 1 inline fallback
    }
    float eval(float x, float y, float z) const override {
        return resolved() ? children[0]->eval(x, y, z) : 1e30f;
    }
    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override {
        if (!resolved()) return FRepNode::codegen_grad(c, x, y, z);
        // Level 2: shared gradient subprogram if the context provides it.
        if (c.instance_grad_call) {
            llvm::Value* ov = nullptr; llvm::Value* od = nullptr;
            if (c.instance_grad_call(children[0].get(),
                                     x.val, x.dot, y.val, y.dot, z.val, z.dot,
                                     ov, od))
                return DualVal{ov, od};
        }
        return children[0]->codegen_grad(c, x, y, z);   // Level 1 inline fallback
    }
    AABB aabb() const override {
        return resolved() ? children[0]->aabb() : AABB::infinite();
    }
    // Instances of the same target with the same params must hash identically so
    // the incremental cache treats them as the same emitted subprogram. The hash
    // folds in the target's hash (not the instance id, which is per-placement).
    std::size_t structural_hash() const noexcept override {
        std::size_t h = resolved() ? children[0]->structural_hash() : 0;
        return h ^ 0x1257A9CEull;   // "instance" tag, distinct from the target itself
    }
    std::size_t structure_hash() const noexcept override {
        std::size_t h = resolved() ? children[0]->structure_hash() : 0;
        return h ^ 0x1257A9CEull;
    }
};

} // namespace frep
