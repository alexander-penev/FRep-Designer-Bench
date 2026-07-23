#pragma once
// core/frep/ad_ir.hpp
//
// Forward-mode AD over LLVM IR Value*.
// Emits instructions that compute (val, dot) for every operation.
//
// Reference: for every operation f(a, b) with derivatives df/da, df/db:
//   result.val = f(a.val, b.val)
//   result.dot = df/da * a.dot + df/db * b.dot   (chain rule)

#include "node.hpp"
#include "core/compiler/llvm_compat.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>

namespace frep::ad_ir {

using D = FRepNode::DualVal;

// ── Construction ────────────────────────────────────────────────────────────
inline D constant(CgCtx& c, float v) {
    return {c.fc(v), c.fc(0.0f)};
}

// ── Arithmetic ──────────────────────────────────────────────────────────────
inline D add(CgCtx& c, D a, D b) {
    return {c.b.CreateFAdd(a.val, b.val), c.b.CreateFAdd(a.dot, b.dot)};
}

inline D sub(CgCtx& c, D a, D b) {
    return {c.b.CreateFSub(a.val, b.val), c.b.CreateFSub(a.dot, b.dot)};
}

inline D mul(CgCtx& c, D a, D b) {
    // (a*b)' = a*b' + b*a'
    auto v = c.b.CreateFMul(a.val, b.val);
    auto d = c.b.CreateFAdd(
        c.b.CreateFMul(a.val, b.dot),
        c.b.CreateFMul(b.val, a.dot));
    return {v, d};
}

inline D div(CgCtx& c, D a, D b) {
    // (a/b)' = (a'b - ab') / b^2
    auto v   = c.b.CreateFDiv(a.val, b.val);
    auto num = c.b.CreateFSub(c.b.CreateFMul(a.dot, b.val),
                              c.b.CreateFMul(a.val, b.dot));
    auto den = c.b.CreateFMul(b.val, b.val);
    return {v, c.b.CreateFDiv(num, den)};
}

inline D neg(CgCtx& c, D a) {
    return {c.b.CreateFNeg(a.val), c.b.CreateFNeg(a.dot)};
}

// ── Scalars ─────────────────────────────────────────────────────────────────
inline D add_s(CgCtx& c, D a, float s) {
    return {c.b.CreateFAdd(a.val, c.fc(s)), a.dot};
}
inline D sub_s(CgCtx& c, D a, float s) {
    return {c.b.CreateFSub(a.val, c.fc(s)), a.dot};
}
inline D mul_s(CgCtx& c, D a, float s) {
    return {c.b.CreateFMul(a.val, c.fc(s)),
            c.b.CreateFMul(a.dot, c.fc(s))};
}

// Variants taking a runtime llvm::Value* instead of a literal — used for
// incremental compilation mode, where node parameters are loaded from a
// runtime buffer rather than baked as constants. The math is identical
// (parameter `s` has zero derivative wrt the spatial variable), but the
// derivative of `s` itself is fc(0) — the gradient chain stays correct
// for downstream consumers.
inline D add_s(CgCtx& c, D a, llvm::Value* s) {
    return {c.b.CreateFAdd(a.val, s), a.dot};
}
inline D sub_s(CgCtx& c, D a, llvm::Value* s) {
    return {c.b.CreateFSub(a.val, s), a.dot};
}
inline D mul_s(CgCtx& c, D a, llvm::Value* s) {
    return {c.b.CreateFMul(a.val, s),
            c.b.CreateFMul(a.dot, s)};
}

// ── Math functions ─────────────────────────────────────────────────────────
inline D sqrt(CgCtx& c, D a) {
    // sqrt(x)' = 1/(2*sqrt(x))
    auto sv = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sqrt, a.val);
    auto two_sv = c.b.CreateFMul(c.fc(2.0f), sv);
    return {sv, c.b.CreateFDiv(a.dot, two_sv)};
}

inline D fabs(CgCtx& c, D a) {
    // |x|' = sign(x) — discontinuous at zero; we use (x>=0 ? 1 : -1)
    auto v = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::fabs, a.val);
    auto is_pos = c.b.CreateFCmpOGE(a.val, c.fc(0.0f));
    auto sign = c.b.CreateSelect(is_pos, c.fc(1.0f), c.fc(-1.0f));
    return {v, c.b.CreateFMul(sign, a.dot)};
}

inline D sin(CgCtx& c, D a) {
    // sin(x)' = cos(x) * x'
    auto sv = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sin, a.val);
    auto cv = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::cos, a.val);
    return {sv, c.b.CreateFMul(cv, a.dot)};
}

inline D cos(CgCtx& c, D a) {
    // cos(x)' = -sin(x) * x'
    auto cv = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::cos, a.val);
    auto sv = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sin, a.val);
    auto neg_sv = c.b.CreateFNeg(sv);
    return {cv, c.b.CreateFMul(neg_sv, a.dot)};
}

// clamp against scalar bounds: constant outside [lo,hi] (gradient zero there).
inline D clamp_s(CgCtx& c, D a, float lo, float hi) {
    auto below = c.b.CreateFCmpOLT(a.val, c.fc(lo));
    auto above = c.b.CreateFCmpOGT(a.val, c.fc(hi));
    auto v1 = c.b.CreateSelect(below, c.fc(lo), a.val);
    auto v  = c.b.CreateSelect(above, c.fc(hi), v1);
    // gradient is a.dot inside the range, 0 outside
    auto in_range = c.b.CreateAnd(c.b.CreateNot(below), c.b.CreateNot(above));
    auto g = c.b.CreateSelect(in_range, a.dot, c.fc(0.0f));
    return {v, g};
}

// max against a scalar: constant (gradient 0) when the scalar wins.
inline D max_s(CgCtx& c, D a, float s) {
    auto a_wins = c.b.CreateFCmpOGT(a.val, c.fc(s));
    auto v = c.b.CreateSelect(a_wins, a.val, c.fc(s));
    auto g = c.b.CreateSelect(a_wins, a.dot, c.fc(0.0f));
    return {v, g};
}

// ── min/max ─────────────────────────────────────────────────────────────────
inline D min(CgCtx& c, D a, D b) {
    auto v = frep::llvm_compat::min_num(c.b, a.val, b.val);
    // dot = (a.val < b.val) ? a.dot : b.dot
    auto pick = c.b.CreateFCmpOLT(a.val, b.val);
    auto d = c.b.CreateSelect(pick, a.dot, b.dot);
    return {v, d};
}

inline D max(CgCtx& c, D a, D b) {
    auto v = frep::llvm_compat::max_num(c.b, a.val, b.val);
    auto pick = c.b.CreateFCmpOGT(a.val, b.val);
    auto d = c.b.CreateSelect(pick, a.dot, b.dot);
    return {v, d};
}

} // namespace frep::ad_ir
