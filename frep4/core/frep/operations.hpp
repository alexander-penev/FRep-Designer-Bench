#pragma once
// core/frep/operations.hpp
//
// Boolean operations via R-functions (F-Rep* convention: f<=0 inside).
//
// Derivation: A v B = {x : f_A(x)<=0 OR f_B(x)<=0}  = {x : min(f_A,f_B)<=0}
//             A ^ B = {x : f_A(x)<=0 AND f_B(x)<=0} = {x : max(f_A,f_B)<=0}
//             A \ B = A ^ !B                        = {x : max(f_A,-f_B)<=0}
//
//   Union        = min(f1, f2)
//   Intersection = max(f1, f2)
//   Difference   = max(f1, -f2)
//   SmoothUnion  = IQ smin(f1, f2, k)

#include "node.hpp"
#include "core/compiler/llvm_compat.hpp"
#include <algorithm>
#include <cmath>

namespace frep {

// ── Union ─────────────────────────────────────────────────────────────────────
class UnionNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Union"; }
public:
    UnionNode(FRepNode::Ptr a, FRepNode::Ptr b, std::string nid = "union") {
        kind = NodeKind::Union; id = std::move(nid);
        children = {std::move(a), std::move(b)};
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto v0 = children[0]->codegen(c, x, y, z);
        auto v1 = children[1]->codegen(c, x, y, z);
        // min_num: LLVM-version-compatible wrapper for the minnum intrinsic.
        return frep::llvm_compat::min_num(c.b, v0, v1, "un");
    }


    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;
    float eval(float x, float y, float z) const override {
        return std::min(children[0]->eval(x,y,z), children[1]->eval(x,y,z));
    }
    std::size_t structural_hash() const noexcept override {
        return (children[0]->structural_hash() * 2654435761ull)
             ^ (children[1]->structural_hash() * 40503ull)
             ^ 0xCCDD'EEFFull;
    }
};

// ── Intersection ──────────────────────────────────────────────────────────────
class IntersectionNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Intersection"; }
public:
    IntersectionNode(FRepNode::Ptr a, FRepNode::Ptr b, std::string nid = "isect") {
        kind = NodeKind::Intersection; id = std::move(nid);
        children = {std::move(a), std::move(b)};
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto v0 = children[0]->codegen(c, x, y, z);
        auto v1 = children[1]->codegen(c, x, y, z);
        return frep::llvm_compat::max_num(c.b, v0, v1, "isect");
    }


    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;
    float eval(float x, float y, float z) const override {
        return std::max(children[0]->eval(x,y,z), children[1]->eval(x,y,z));
    }
    std::size_t structural_hash() const noexcept override {
        return (children[0]->structural_hash() * 2654435761ull)
             ^ (children[1]->structural_hash() * 40503ull)
             ^ 0x1122'3344ull;
    }
};

// ── Difference ────────────────────────────────────────────────────────────────
class DifferenceNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Difference"; }
public:
    DifferenceNode(FRepNode::Ptr a, FRepNode::Ptr b, std::string nid = "diff") {
        kind = NodeKind::Difference; id = std::move(nid);
        children = {std::move(a), std::move(b)};
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto v0 = children[0]->codegen(c, x, y, z);
        auto v1 = children[1]->codegen(c, x, y, z);
        return frep::llvm_compat::max_num(c.b, v0, c.b.CreateFNeg(v1, "neg"), "diff");
    }


    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;
    float eval(float x, float y, float z) const override {
        return std::max(children[0]->eval(x,y,z), -children[1]->eval(x,y,z));
    }
    std::size_t structural_hash() const noexcept override {
        return (children[0]->structural_hash() * 2654435761ull)
             ^ (children[1]->structural_hash() * 40503ull)
             ^ 0x5566'7788ull;
    }
};

// ── SmoothUnion (Inigo Quilez smin) ──────────────────────────────────────────
// smin(a,b,k): h = clamp(0.5 + 0.5*(b-a)/k, 0,1); mix(b,a,h) - k*h*(1-h)*0.5
class SmoothUnionNode final : public FRepNode {
    const char* type_name() const noexcept override { return "SmoothUnion"; }
public:
    SmoothUnionNode(FRepNode::Ptr a, FRepNode::Ptr b, float k, std::string nid = "smin") {
        kind = NodeKind::SmoothUnion; id = std::move(nid);
        params["k"] = k;
        children = {std::move(a), std::move(b)};
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        float kv = params.at("k");
        auto  da = children[0]->codegen(c, x, y, z);
        auto  db = children[1]->codegen(c, x, y, z);

        // Cubic polynomial smin (Inigo Quilez), C2-continuous so the normal has
        // no kink at the blend boundary. The internal scale 2.0 (not IQ's 6) is
        // calibrated so `k` keeps the same blend depth as the previous quadratic
        // smin — i.e. existing scenes' k values look the same, just smooth.
        // Pure polynomial: JITs identically on every path (incl. NVPTX).
        //   kk = k*0.75;  h = max(kk - |a-b|, 0)/kk;  smin = min(a,b) - h^3*kk/6
        auto kk   = c.fc(kv * 2.0f);
        auto diff = b.CreateFSub(da, db);
        // |a-b| via fabs intrinsic.
        auto adiff = frep::llvm_compat::unary_intrinsic(
            b, llvm::Intrinsic::fabs, diff, "adiff");
        auto h_num = frep::llvm_compat::max_num(
            b, b.CreateFSub(kk, adiff), c.fc(0.0f));
        auto h = b.CreateFDiv(h_num, kk, "h");
        auto mn = frep::llvm_compat::min_num(b, da, db, "mn");
        // h^3 * kk / 6
        auto h3 = b.CreateFMul(b.CreateFMul(h, h), h);
        auto corr = b.CreateFMul(b.CreateFMul(h3, kk), c.fc(1.0f / 6.0f), "corr");
        return b.CreateFSub(mn, corr, "smin");
    }


    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;
    float eval(float x, float y, float z) const override {
        float a = children[0]->eval(x,y,z);
        float b = children[1]->eval(x,y,z);
        float k = params.at("k");
        if (k <= 0.0f) return std::min(a, b);
        // Cubic polynomial smin (must match codegen() exactly for parity).
        float kk = k * 2.0f;
        float h  = std::max(kk - std::fabs(a - b), 0.0f) / kk;
        return std::min(a, b) - h*h*h*kk*(1.0f/6.0f);
    }
    std::size_t structural_hash() const noexcept override {
        return (children[0]->structural_hash() * 2654435761ull)
             ^ std::hash<float>{}(params.at("k"))
             ^ 0x9900'AABBull;
    }
};

// ── Negate ────────────────────────────────────────────────────────────────────
class NegateNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Negate"; }
public:
    explicit NegateNode(FRepNode::Ptr a, std::string nid = "neg") {
        kind = NodeKind::Negate; id = std::move(nid);
        children = {std::move(a)};
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        return c.b.CreateFNeg(children[0]->codegen(c, x, y, z), "neg");
    }


    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;
    float eval(float x, float y, float z) const override {
        return -children[0]->eval(x,y,z);
    }
    std::size_t structural_hash() const noexcept override {
        return children[0]->structural_hash() ^ 0xDEAD'BEEFull;
    }
};

// Helper: builds a Union tree from a vector of nodes (left-associative).
inline FRepNode::Ptr union_all(std::vector<FRepNode::Ptr> nodes) {
    if (nodes.empty()) return nullptr;
    auto root = std::move(nodes[0]);
    for (std::size_t i = 1; i < nodes.size(); ++i)
        root = std::make_shared<UnionNode>(std::move(root), std::move(nodes[i]));
    return root;
}

} // namespace frep
