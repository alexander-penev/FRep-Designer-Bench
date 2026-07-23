#pragma once
// core/frep/node.hpp
//
// FRepNode — the base AST node of the visual language.
// Each node represents a function f(x,y,z) -> float (signed distance).
// F-Rep* convention: f(X) <= 0 means inside the object.
// The FRepNode tree is the program — the model is source code.

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace frep {

enum class NodeKind {
    Sphere, Box, Plane,
    Union, Intersection, Difference, SmoothUnion,
    Negate,
    Translate, Scale, RotateY, RotateX, RotateZ,
    TwistY, BendXY, TaperY,
    Scene,
    Instance,
    // Plugin-defined nodes — emit goes through the FRepNode::emit_glsl
    // virtual fallback rather than the built-in switch table. Plugin
    // authors should set `kind = NodeKind::Plugin` in their node ctor.
    Plugin,
};

// Context passed during code generation — holds the LLVM state.
struct CgCtx {
    llvm::LLVMContext& lc;
    llvm::Module&      mod;
    llvm::IRBuilder<>& b;

    unsigned     width = 1;   // >1 emits <width x float> (SIMD broadcast path)
    llvm::Value* vsplat(llvm::Value* s) const {
        return width > 1 ? b.CreateVectorSplat(width, s) : s;
    }

    llvm::Type*  f32()  const { return llvm::Type::getFloatTy(lc); }
    llvm::Type*  i32()  const { return llvm::Type::getInt32Ty(lc); }
    llvm::Type*  vd()   const { return llvm::Type::getVoidTy(lc); }
    llvm::Value* fc(float v)  const {
        auto* s = llvm::ConstantFP::get(llvm::Type::getFloatTy(lc), v);
        return width > 1 ? b.CreateVectorSplat(width, s) : s;
    }
    llvm::Value* ic(int v)    const { return llvm::ConstantInt::get(llvm::Type::getInt32Ty(lc), v, true); }

    // ── Incremental parameter mode ──────────────────────────────────────────
    // When `params_buffer` is non-null and `incremental_params` is true,
    // node parameters are loaded from a runtime float array instead of
    // baked as IR constants. The codegen calls back into `slot_for_param`
    // to obtain the buffer index for a given (node_id, param_name) pair,
    // recording the default value for the host-side allocator to seed the
    // buffer with.
    //
    // The callback returns -1 when this parameter must remain a constant
    // (e.g. an unsupported node type, or the calling code passed
    // incremental_params=false).
    bool                 incremental_params = false;
    llvm::Value*         params_buffer      = nullptr;
    std::function<int(const std::string& node_id,
                      const std::string& param_name,
                      float default_value,
                      int param_class)> slot_for_param;

    // ── Instancing Level 2 (shared subprograms) ─────────────────────────────
    // When set, an InstanceNode emits a call to a shared LLVM function for its
    // target's geometry instead of inlining the target subtree. The callback
    // returns (creating on first use, memoised by target pointer) the function
    // `float f(x, y, z, params)` computing the target's SDF. When null, an
    // InstanceNode falls back to inlining its target (Level 1). This is the IR
    // twin of the GLSL emitter's _inst_fn_N sharing.
    std::function<llvm::Value*(const class FRepNode* target,
                               llvm::Value* x, llvm::Value* y, llvm::Value* z)>
        instance_call;

    // Level 2 for the AD (dual-number) path: an InstanceNode emits a call to a
    // shared gradient function taking the incoming duals and returning the
    // target's (value, derivative). Null -> inline (Level 1). Returns the two
    // components via the out-params; the return value is non-null on success.
    std::function<bool(const class FRepNode* target,
                       llvm::Value* xv, llvm::Value* xd,
                       llvm::Value* yv, llvm::Value* yd,
                       llvm::Value* zv, llvm::Value* zd,
                       llvm::Value*& out_val, llvm::Value*& out_dot)>
        instance_grad_call;

    // Returns an LLVM Value* for the parameter — either a constant
    // (Constant mode, or when no slot was assigned) or a load from
    // params_buffer[slot] (Incremental mode). `param_class` lets the
    // CompilePolicy decide placement by class (see compile_policy.hpp);
    // it defaults to Geometry (0) since most node parameters are geometric.
    llvm::Value* param_value(const std::string& node_id,
                             const std::string& param_name,
                             float default_value,
                             int param_class = 0)
    {
        if (incremental_params && params_buffer && slot_for_param) {
            int slot = slot_for_param(node_id, param_name, default_value,
                                      param_class);
            if (slot >= 0) {
                auto* idx = ic(slot);
                auto* gep = b.CreateGEP(f32(), params_buffer, idx,
                                        node_id + "." + param_name + "_addr");
                auto* ld = b.CreateLoad(f32(), gep,
                                    node_id + "." + param_name);
                return vsplat(ld);
            }
        }
        return fc(default_value);
    }
};

// Base class.
//
// ── SDF invariant (the contract every node guarantees) ──────────────────────
//
// Every FRepNode evaluates to a SIGNED DISTANCE FIELD with this system's
// fixed convention:
//
//   • Sign: POSITIVE outside the object, NEGATIVE inside, zero on the
//     surface. The value is an *oriented distance* — how far the query
//     point is from the surface, signed by which side it's on.
//   • Metric: the field is (approximately) Euclidean and 1-Lipschitz —
//     |f(a) - f(b)| <= |a - b|. Equivalently |∇f| <= 1 where defined. The
//     value never *over*-estimates the true distance to the surface.
//
// This is not a stylistic choice; three independent mechanisms depend on
// it and silently misbehave if it's violated:
//
//   1. Sphere tracing steps t += f(p): requires f > 0 outside so the step
//      moves forward, and 1-Lipschitz so the step never overshoots the
//      surface. A non-metric or sign-flipped field fails to converge.
//   2. CSG / scene composition via min()/max(): requires "smaller = nearer"
//      (a positive, comparable metric) for the booleans to mean what they
//      should.
//   3. BVH pruning (core/accel/bvh.hpp): requires the point→AABB distance
//      to be a LOWER BOUND on f inside the box. A field that under-
//      estimates distance (e.g. the Chebyshev box metric we once shipped
//      by mistake) breaks the prune and yields wrong results.
//
// Built-in primitives, operations, transforms, and deformations all honour
// this. Externally-sourced geometry (CustomExpr, Plugin, MeshSDF) MUST be
// normalised to this convention at import time — by construction, by a
// Lipschitz bound (f/L is a conservative 1-Lipschitz field), or by voxel
// redistancing — rather than relying on per-use checks in the render/BVH
// hot paths. The test suite enforces the invariant on the built-ins
// (see tests/test_sdf_invariant.cpp).
class FRepNode {
public:
    using Ptr = std::shared_ptr<FRepNode>;

    // Dual<float> in LLVM IR: a stuck-together (val, dot) pair
    struct DualVal {
        llvm::Value* val;
        llvm::Value* dot;
    };

    // Axis-aligned bounding box — for BVH acceleration.
    struct AABB {
        float min_x, min_y, min_z;
        float max_x, max_y, max_z;

        // Union of two AABBs.
        static AABB merge(const AABB& a, const AABB& b) {
            return {
                std::min(a.min_x, b.min_x), std::min(a.min_y, b.min_y),
                std::min(a.min_z, b.min_z),
                std::max(a.max_x, b.max_x), std::max(a.max_y, b.max_y),
                std::max(a.max_z, b.max_z)
            };
        }
        // "Unbounded" AABB — for primitives like Plane.
        static AABB infinite() {
            constexpr float inf = 1e9f;
            return {-inf, -inf, -inf, inf, inf, inf};
        }
        float center_x() const { return 0.5f * (min_x + max_x); }
        float center_y() const { return 0.5f * (min_y + max_y); }
        float center_z() const { return 0.5f * (min_z + max_z); }
    };

    NodeKind    kind;
    std::string id;
    std::vector<Ptr> children;
    std::unordered_map<std::string, float> params;

    // Generates LLVM IR — returns a float Value* (the SDF value at point x,y,z).
    // Marked AlwaysInline → O3 inlines it directly into render_tile.
    virtual llvm::Value* codegen(CgCtx& ctx,
                                 llvm::Value* x,
                                 llvm::Value* y,
                                 llvm::Value* z) const = 0;

    // Evaluates the SDF in plain C++ — no JIT, no LLVM.
    // Used by tooling that needs to sample the SDF without going through the
    // JIT (mesh extraction, AABB refinement, debug probes).
    //
    // Default: not implemented — primitives, ops, and transforms override
    // this. CustomExprNode does not override and will throw at runtime; users
    // wanting to mesh a CustomExprNode should JIT it first.
    // RTTI-free downcast for the JIT interval/SIMD paths (core is -fno-rtti).
    // CustomExprNode overrides to return its expr AST; all others return null.
    virtual const void* custom_expr_ast() const { return nullptr; }

    virtual float eval(float x, float y, float z) const {
        throw std::runtime_error(
            std::string("FRepNode::eval not implemented for type '") +
            type_name() + "'");
    }

    // Generates LLVM IR for forward-mode AD: takes dual numbers
    // (x.val, x.dot), (y.val, y.dot), (z.val, z.dot)
    // and returns the dual {sdf_val, sdf_dot}.
    //
    // Default implementation: finite-difference (slower but universal).
    // Overridden in the primitives for clean symbolic differentiation.
    virtual DualVal codegen_grad(CgCtx&  ctx,
                                 DualVal x,
                                 DualVal y,
                                 DualVal z) const;

    // Hash of the subtree — for the incremental cache.
    virtual std::size_t structural_hash() const noexcept = 0;

    // Hash of the subtree's *structure* only — ignoring parameter values.
    // Two subtrees with the same structure_hash differ only in parameter
    // values (e.g. SphereNode(r=1) and SphereNode(r=2)).
    //
    // Default implementation: combine the node kind with the structure_hash
    // of every child. Primitive nodes inherit this without override; nodes
    // with deeper structure (e.g. CustomExprNode with text) should override.
    virtual std::size_t structure_hash() const noexcept {
        std::size_t h = std::hash<int>{}(static_cast<int>(kind));
        for (const auto& c : children)
            h = h * 2654435761ull ^ c->structure_hash();
        return h;
    }

    // Conservative axis-aligned bounding box of this subtree.
    // Used for BVH construction in scene_material.
    // Default: infinite (safe — the object is always tested).
    virtual AABB aabb() const { return AABB::infinite(); }

    // Stable textual name of the type — for JSON serialization.
    // Must match the key in the node factory (see core/io/scene_io.cpp).
    virtual const char* type_name() const noexcept = 0;

    // ── GLSL emission for the GPU compute path ────────────────────────────────
    // Default implementation returns false, meaning "I don't know how to
    // emit GLSL for myself; the renderer should fall back to CPU only for
    // any scene containing me". Built-in nodes (primitives, ops,
    // transforms, deformations) are handled directly by the GlslEmitter's
    // dispatch table. Plugin nodes that want to run on GPU override this
    // method, writing a GLSL expression that computes the SDF at a point.
    //
    // Conventions for the implementation:
    //  - The output stream gets a GLSL expression (no trailing semicolon).
    //  - `child_exprs` contains pre-emitted expressions for each child
    //    (in `children` order) — typically one per child.
    //  - `var_prefix` is a unique name (e.g. "n12_") to avoid SSA name
    //    clashes when emitted inside larger expressions. Implementations
    //    that need temporary variables should prefix them with this.
    //  - Return true on success, false to fall back to CPU.
    //
    // Example for a Capsule SDF (length=h, radius=r, along Y):
    //   out << "(length(vec3(x, clamp(y, -" << flit(h*0.5f)
    //       << ", " << flit(h*0.5f) << "), z)) - " << flit(r) << ")";
    //   return true;
    virtual bool emit_glsl(std::ostream& out,
                           const std::vector<std::string>& child_exprs,
                           const std::string& var_prefix) const {
        (void)out; (void)child_exprs; (void)var_prefix;
        return false;
    }

    virtual ~FRepNode() = default;
};

// Whether a scene tree needs a reduced raymarch step (safety_factor < 1).
//
// A single primitive or an affine/twist transform of one yields a true
// (or near-true) Euclidean SDF, so a full `t += d` step never oversteps
// and safety_factor 1.0 is safe — and ~20% faster, since every ray takes
// fewer, larger steps. CSG operations (Union/Intersection/Difference/
// SmoothUnion) and arbitrary plugin / mesh / custom-expression nodes do
// NOT preserve the distance property — near a seam the reported distance
// exceeds the true distance — so those require the conservative step.
//
// Returns true if ANY node in the tree is a distance-non-preserving kind,
// meaning the renderer should clamp safety_factor below 1. A scene of
// only primitives + transforms returns false and can run at full step.
inline bool node_requires_safety_step(const FRepNode& n) {
    switch (n.kind) {
        case NodeKind::Union:
        case NodeKind::Intersection:
        case NodeKind::Difference:
        case NodeKind::SmoothUnion:
        case NodeKind::Plugin:        // mesh / custom-expr / plugin SDFs
            return true;
        default:
            break;
    }
    for (const auto& c : n.children)
        if (c && node_requires_safety_step(*c)) return true;
    return false;
}

// Whether the field is provably 1-Lipschitz, so the tile cull's Lipschitz box
// rule is sound at L=1. Metric primitives (Sphere/Box/Plane), boolean/blend
// composition (min/max of 1-Lipschitz stays 1-Lipschitz), and rigid transforms
// (Translate/RotateY) all preserve |grad f| = 1. What breaks it: CustomExpr /
// Plugin / Mesh (arbitrary implicit, unknown gradient) and a Scale by s < 1,
// which multiplies the gradient by 1/s > 1. A scene that returns true can be
// culled with L=1; anything else needs an estimated L or the interval method.
inline bool node_is_unit_lipschitz(const FRepNode& n) {
    switch (n.kind) {
        case NodeKind::Sphere:
        case NodeKind::Box:
        case NodeKind::Plane:
        case NodeKind::Union:
        case NodeKind::Intersection:
        case NodeKind::Difference:
        case NodeKind::SmoothUnion:
        case NodeKind::Negate:
        case NodeKind::Translate:
        case NodeKind::RotateY:
        case NodeKind::RotateX:
        case NodeKind::RotateZ:
        case NodeKind::Scene:
        case NodeKind::Instance:      // metric-ness follows the shared target
            break;                       // gradient-preserving
        case NodeKind::Scale: {          // sound only if every |factor| >= 1 (never amplifies)
            auto ax = n.params.find("sx");
            auto ay = n.params.find("sy");
            auto az = n.params.find("sz");
            float sx = ax==n.params.end()?1.0f:ax->second;
            float sy = ay==n.params.end()?1.0f:ay->second;
            float sz = az==n.params.end()?1.0f:az->second;
            if (std::abs(sx) < 1.0f || std::abs(sy) < 1.0f || std::abs(sz) < 1.0f) return false;
            break;
        }
        default:                         // TwistY/BendXY/TaperY/Plugin/CustomExpr/...
            return false;
    }
    for (const auto& c : n.children)
        if (c && !node_is_unit_lipschitz(*c)) return false;
    return true;
}

// Number of nodes in a subtree (this node + all descendants). A cheap
// proxy for per-object SDF evaluation cost, used by the spatial-guard
// heuristic: a bare sphere is 1-2 nodes (cheap — guarding hurts), a
// twisted box smooth-unioned with a sphere is ~6-8 (expensive — guarding
// wins 3-6×). The calibration finds the crossover node count on the
// host; this counts it for a given object.
inline int node_count(const FRepNode& n) {
    int total = 1;
    for (const auto& c : n.children)
        if (c) total += node_count(*c);
    return total;
}

// Find a node by its id anywhere in a subtree (depth-first). Returns nullptr if
// no node has that id. Used by the property grid to locate the node whose
// parameter is being edited.
inline FRepNode* find_node_by_id(FRepNode& root, const std::string& id) {
    if (root.id == id) return &root;
    for (auto& c : root.children)
        if (c) if (FRepNode* r = find_node_by_id(*c, id)) return r;
    return nullptr;
}

} // namespace frep
