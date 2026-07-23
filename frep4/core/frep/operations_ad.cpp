// core/frep/operations_ad.cpp
//
// AD codegen for CSG operations and transforms.

#include "ad_ir.hpp"
#include "operations.hpp"
#include "transforms.hpp"
#include "core/compiler/llvm_compat.hpp"

namespace frep {

namespace ai = ad_ir;

// ── Union: min(f1, f2) ──────────────────────────────────────────────────────
FRepNode::DualVal UnionNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a = children[0]->codegen_grad(c, x, y, z);
    auto b = children[1]->codegen_grad(c, x, y, z);
    return ai::min(c, a, b);
}

// ── Intersection: max(f1, f2) ───────────────────────────────────────────────
FRepNode::DualVal IntersectionNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a = children[0]->codegen_grad(c, x, y, z);
    auto b = children[1]->codegen_grad(c, x, y, z);
    return ai::max(c, a, b);
}

// ── Difference: max(f1, -f2) ────────────────────────────────────────────────
FRepNode::DualVal DifferenceNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a = children[0]->codegen_grad(c, x, y, z);
    auto b = children[1]->codegen_grad(c, x, y, z);
    return ai::max(c, a, ai::neg(c, b));
}

// ── SmoothUnion: cubic polynomial smin(a,b,k) ───────────────────────────────
// kk = k*2.0;  h = max(kk - |a-b|, 0)/kk;  smin = min(a,b) - h^3*kk/6
// C2-continuous → no normal kink at the blend boundary. Must match codegen().
FRepNode::DualVal SmoothUnionNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    float k = params.at("k");
    auto da = children[0]->codegen_grad(c, x, y, z);
    auto db = children[1]->codegen_grad(c, x, y, z);
    if (k <= 0.0f) return ai::min(c, da, db);

    float kk = k * 2.0f;
    // adiff = |a - b|
    auto adiff = ai::fabs(c, ai::sub(c, da, db));
    // h = max(kk - adiff, 0) / kk
    auto h = ai::mul_s(c, ai::max(c, ai::sub_s(c, ai::mul_s(c, adiff, -1.0f), -kk),
                                     ai::constant(c, 0.0f)),
                       1.0f / kk);
    // mn = min(a, b)
    auto mn = ai::min(c, da, db);
    // corr = h^3 * kk / 6
    auto h3   = ai::mul(c, ai::mul(c, h, h), h);
    auto corr = ai::mul_s(c, h3, kk / 6.0f);
    return ai::sub(c, mn, corr);
}

// ── Negate: -f ──────────────────────────────────────────────────────────────
FRepNode::DualVal NegateNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    return ai::neg(c, children[0]->codegen_grad(c, x, y, z));
}

// ════════════════════════════════════════════════════════════════════════════
// Transforms
// ════════════════════════════════════════════════════════════════════════════

// ── Translate: f(x-tx, y-ty, z-tz) ──────────────────────────────────────────
// The derivative w.r.t. position does not change — only val is shifted.
FRepNode::DualVal TranslateNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto tx_v = c.param_value(id, "tx", params.at("tx"));
    auto ty_v = c.param_value(id, "ty", params.at("ty"));
    auto tz_v = c.param_value(id, "tz", params.at("tz"));
    DualVal xt{c.b.CreateFSub(x.val, tx_v), x.dot};
    DualVal yt{c.b.CreateFSub(y.val, ty_v), y.dot};
    DualVal zt{c.b.CreateFSub(z.val, tz_v), z.dot};
    return children[0]->codegen_grad(c, xt, yt, zt);
}

// ── Scale: f(p/s) * s ───────────────────────────────────────────────────────
// Note: dot must be scaled too — d/dp [f(p/s)*s] = f'(p/s).
// Here p/s means: val/s, dot/s. Then the result is multiplied by s,
// which leaves dot unchanged again (s * dot/s = dot). Correct.
FRepNode::DualVal ScaleNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    // Non-uniform: divide each axis by its own factor, then multiply the result
    // by the smallest factor (conservative Lipschitz correction, matching the
    // scalar codegen). In Constant mode the factors fold to literals.
    auto sx = c.param_value(id, "sx", params.at("sx"));
    auto sy = c.param_value(id, "sy", params.at("sy"));
    auto sz = c.param_value(id, "sz", params.at("sz"));
    auto ix = c.b.CreateFDiv(c.fc(1.0f), sx);
    auto iy = c.b.CreateFDiv(c.fc(1.0f), sy);
    auto iz = c.b.CreateFDiv(c.fc(1.0f), sz);
    DualVal xs{c.b.CreateFMul(x.val, ix), c.b.CreateFMul(x.dot, ix)};
    DualVal ys{c.b.CreateFMul(y.val, iy), c.b.CreateFMul(y.dot, iy)};
    DualVal zs{c.b.CreateFMul(z.val, iz), c.b.CreateFMul(z.dot, iz)};
    auto inner = children[0]->codegen_grad(c, xs, ys, zs);
    auto mn = frep::llvm_compat::binary_intrinsic(c.b, llvm::Intrinsic::minnum, sx, sy);
    mn = frep::llvm_compat::binary_intrinsic(c.b, llvm::Intrinsic::minnum, mn, sz);
    return ai::mul_s(c, inner, mn);
}

// ── RotateY ─────────────────────────────────────────────────────────────────
// Rotation is a linear transform — applied identically to val and dot.
//   xr =  cos*x + sin*z
//   zr = -sin*x + cos*z   (transpose = rot(-a))
FRepNode::DualVal RotateYNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a_v = c.param_value(id, "a", params.at("a"));
    // cos/sin via llvm intrinsics — they fold to constants when a_v is one.
    auto ca = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::cos, a_v);
    auto sa = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sin, a_v);
    auto& b = c.b;

    DualVal xr{
        b.CreateFAdd(b.CreateFMul(ca, x.val), b.CreateFMul(sa, z.val)),
        b.CreateFAdd(b.CreateFMul(ca, x.dot), b.CreateFMul(sa, z.dot))
    };
    DualVal zr{
        b.CreateFSub(b.CreateFMul(ca, z.val), b.CreateFMul(sa, x.val)),
        b.CreateFSub(b.CreateFMul(ca, z.dot), b.CreateFMul(sa, x.dot))
    };
    return children[0]->codegen_grad(c, xr, y, zr);
}

FRepNode::DualVal RotateXNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a_v = c.param_value(id, "a", params.at("a"));
    auto ca = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::cos, a_v);
    auto sa = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sin, a_v);
    auto& b = c.b;
    DualVal yr{
        b.CreateFAdd(b.CreateFMul(ca, y.val), b.CreateFMul(sa, z.val)),
        b.CreateFAdd(b.CreateFMul(ca, y.dot), b.CreateFMul(sa, z.dot))
    };
    DualVal zr{
        b.CreateFSub(b.CreateFMul(ca, z.val), b.CreateFMul(sa, y.val)),
        b.CreateFSub(b.CreateFMul(ca, z.dot), b.CreateFMul(sa, y.dot))
    };
    return children[0]->codegen_grad(c, x, yr, zr);
}

FRepNode::DualVal RotateZNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto a_v = c.param_value(id, "a", params.at("a"));
    auto ca = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::cos, a_v);
    auto sa = frep::llvm_compat::unary_intrinsic(c.b, llvm::Intrinsic::sin, a_v);
    auto& b = c.b;
    DualVal xr{
        b.CreateFAdd(b.CreateFMul(ca, x.val), b.CreateFMul(sa, y.val)),
        b.CreateFAdd(b.CreateFMul(ca, x.dot), b.CreateFMul(sa, y.dot))
    };
    DualVal yr{
        b.CreateFSub(b.CreateFMul(ca, y.val), b.CreateFMul(sa, x.val)),
        b.CreateFSub(b.CreateFMul(ca, y.dot), b.CreateFMul(sa, x.dot))
    };
    return children[0]->codegen_grad(c, xr, yr, z);
}

} // namespace frep
