// LLVM-IR interval emitter for a node tree — the third interval implementation,
// alongside the CPU evaluator (node_interval.hpp) and the GLSL string emitter
// (core/gpu/glsl_node_interval.hpp). Same node kinds and same interval rules,
// but each interval is a pair of llvm::Value* (lo, hi) and every operation emits
// LLVM instructions into a builder. This lets the IR raymarch paths (CpuIr /
// GpuIr) do interval tile culling — the piece the GLSL path already had via its
// string emitter but the IR paths lacked. Kept rule-for-rule in step with the
// other two; if one changes, change all three.
#pragma once
#include "core/frep/node.hpp"
#include "core/compiler/llvm_compat.hpp"
#include <llvm/IR/IRBuilder.h>
#include <cmath>

namespace frep::jit {

// An interval as two SSA values.
struct IvV { llvm::Value* lo; llvm::Value* hi; };

class NodeIntervalIR {
public:
    NodeIntervalIR(llvm::LLVMContext& ctx, llvm::IRBuilder<>& b) : ctx_(ctx), b_(b) {}

    // Emit the interval bound of a node tree over the box [X,Y,Z].
    IvV emit(const FRepNode& n, IvV X, IvV Y, IvV Z) { return node(n, X, Y, Z); }

private:
    llvm::LLVMContext& ctx_;
    llvm::IRBuilder<>& b_;

    llvm::Type* f32() { return llvm::Type::getFloatTy(ctx_); }
    llvm::Value* fc(float v) { return llvm::ConstantFP::get(f32(), v); }
    IvV pt(float v) { return {fc(v), fc(v)}; }          // point interval [v,v]
    IvV ptv(llvm::Value* v) { return {v, v}; }

    llvm::Value* fmin(llvm::Value* a, llvm::Value* c) {
        return frep::llvm_compat::binary_intrinsic(b_, llvm::Intrinsic::minnum, a, c); }
    llvm::Value* fmax(llvm::Value* a, llvm::Value* c) {
        return frep::llvm_compat::binary_intrinsic(b_, llvm::Intrinsic::maxnum, a, c); }
    llvm::Value* fsqrt(llvm::Value* a) {
        return frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::sqrt, a); }
    llvm::Value* fabsv(llvm::Value* a) {
        return frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::fabs, a); }
    llvm::Value* min4(llvm::Value* a, llvm::Value* c, llvm::Value* d, llvm::Value* e) {
        return fmin(fmin(a, c), fmin(d, e)); }
    llvm::Value* max4(llvm::Value* a, llvm::Value* c, llvm::Value* d, llvm::Value* e) {
        return fmax(fmax(a, c), fmax(d, e)); }

    // ── Interval arithmetic (mirrors ivdetail:: in node_interval.hpp) ─────────
    IvV add(IvV a, IvV b) { return {b_.CreateFAdd(a.lo, b.lo), b_.CreateFAdd(a.hi, b.hi)}; }
    IvV sub(IvV a, IvV b) { return {b_.CreateFSub(a.lo, b.hi), b_.CreateFSub(a.hi, b.lo)}; }
    IvV neg(IvV a) { return {b_.CreateFNeg(a.hi), b_.CreateFNeg(a.lo)}; }
    IvV mul(IvV a, IvV b) {
        auto* p0 = b_.CreateFMul(a.lo, b.lo);
        auto* p1 = b_.CreateFMul(a.lo, b.hi);
        auto* p2 = b_.CreateFMul(a.hi, b.lo);
        auto* p3 = b_.CreateFMul(a.hi, b.hi);
        return {min4(p0,p1,p2,p3), max4(p0,p1,p2,p3)};
    }
    IvV sqr(IvV a) {
        // [lo,hi]^2: if the interval straddles 0 the low is 0, else min(lo^2,hi^2).
        auto* l = b_.CreateFMul(a.lo, a.lo);
        auto* h = b_.CreateFMul(a.hi, a.hi);
        auto* straddle = b_.CreateAnd(
            b_.CreateFCmpOLE(a.lo, fc(0.0f)), b_.CreateFCmpOGE(a.hi, fc(0.0f)));
        auto* lo = b_.CreateSelect(straddle, fc(0.0f), fmin(l, h));
        return {lo, fmax(l, h)};
    }
    IvV absi(IvV a) {
        auto* l = fabsv(a.lo);
        auto* h = fabsv(a.hi);
        auto* straddle = b_.CreateAnd(
            b_.CreateFCmpOLE(a.lo, fc(0.0f)), b_.CreateFCmpOGE(a.hi, fc(0.0f)));
        auto* lo = b_.CreateSelect(straddle, fc(0.0f), fmin(l, h));
        return {lo, fmax(l, h)};
    }
    IvV sqrti(IvV a) {
        return {fsqrt(fmax(a.lo, fc(0.0f))), fsqrt(fmax(a.hi, fc(0.0f)))};
    }
    IvV divi(IvV a, IvV b) {
        // If the denominator straddles 0 the quotient is unbounded; return a
        // huge interval (disables the cull for that sub-expression, soundly).
        auto* straddle = b_.CreateAnd(
            b_.CreateFCmpOLE(b.lo, fc(0.0f)), b_.CreateFCmpOGE(b.hi, fc(0.0f)));
        auto* q0 = b_.CreateFDiv(a.lo, b.lo);
        auto* q1 = b_.CreateFDiv(a.lo, b.hi);
        auto* q2 = b_.CreateFDiv(a.hi, b.lo);
        auto* q3 = b_.CreateFDiv(a.hi, b.hi);
        auto* lo = b_.CreateSelect(straddle, fc(-1e30f), min4(q0,q1,q2,q3));
        auto* hi = b_.CreateSelect(straddle, fc( 1e30f), max4(q0,q1,q2,q3));
        return {lo, hi};
    }
    IvV imin(IvV a, IvV b) { return {fmin(a.lo, b.lo), fmin(a.hi, b.hi)}; }
    IvV imax(IvV a, IvV b) { return {fmax(a.lo, b.lo), fmax(a.hi, b.hi)}; }

    // sin/cos over an interval. crosses_pi checks whether the interval spans an
    // extremum at phase `ph`; if so the bound saturates to ±1.
    llvm::Value* crosses(IvV a, float ph) {
        const float PI = 3.14159265359f;
        // floor((hi-ph)/PI) >= ceil((lo-ph)/PI)
        auto* fl = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::floor,
            b_.CreateFDiv(b_.CreateFSub(a.hi, fc(ph)), fc(PI)));
        auto* cl = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::ceil,
            b_.CreateFDiv(b_.CreateFSub(a.lo, fc(ph)), fc(PI)));
        return b_.CreateFCmpOGE(fl, cl);
    }
    IvV isin(IvV a) {
        auto* e0 = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::sin, a.lo);
        auto* e1 = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::sin, a.hi);
        auto* lo = fmin(e0, e1); auto* hi = fmax(e0, e1);
        hi = b_.CreateSelect(crosses(a,  1.57079632679f), fc( 1.0f), hi);
        lo = b_.CreateSelect(crosses(a, -1.57079632679f), fc(-1.0f), lo);
        return {lo, hi};
    }
    IvV icos(IvV a) {
        auto* e0 = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::cos, a.lo);
        auto* e1 = frep::llvm_compat::unary_intrinsic(b_, llvm::Intrinsic::cos, a.hi);
        auto* lo = fmin(e0, e1); auto* hi = fmax(e0, e1);
        hi = b_.CreateSelect(crosses(a, 0.0f),           fc( 1.0f), hi);
        lo = b_.CreateSelect(crosses(a, 3.14159265359f), fc(-1.0f), lo);
        return {lo, hi};
    }

    llvm::Value* clampv(llvm::Value* v, float lo, float hi) {
        return fmin(fmax(v, fc(lo)), fc(hi));
    }

    float pf(const FRepNode& n, const char* k, float d) {
        auto it = n.params.find(k); return it != n.params.end() ? it->second : d;
    }

    // ── Node dispatch (mirrors node_interval() in node_interval.hpp) ──────────
    IvV node(const FRepNode& n, IvV X, IvV Y, IvV Z) {
        using K = NodeKind;
        switch (n.kind) {
            case K::Sphere: {
                IvV s = add(add(sqr(X), sqr(Y)), sqr(Z));
                IvV r = sqrti(s); float rad = pf(n,"r",1.0f);
                return {b_.CreateFSub(r.lo, fc(rad)), b_.CreateFSub(r.hi, fc(rad))};
            }
            case K::Box: {
                IvV qx = sub(absi(X), pt(pf(n,"hx",0.5f)));
                IvV qy = sub(absi(Y), pt(pf(n,"hy",0.5f)));
                IvV qz = sub(absi(Z), pt(pf(n,"hz",0.5f)));
                IvV mx = {fmax(qx.lo, fc(0.0f)), fmax(qx.hi, fc(0.0f))};
                IvV my = {fmax(qy.lo, fc(0.0f)), fmax(qy.hi, fc(0.0f))};
                IvV mz = {fmax(qz.lo, fc(0.0f)), fmax(qz.hi, fc(0.0f))};
                IvV out = sqrti(add(add(sqr(mx), sqr(my)), sqr(mz)));
                IvV mxa = imax(qx, imax(qy, qz));
                IvV ins = {fmin(mxa.lo, fc(0.0f)), fmin(mxa.hi, fc(0.0f))};
                return add(out, ins);
            }
            case K::Plane: {
                IvV tx = mul(pt(pf(n,"nx",0.0f)), X);
                IvV ty = mul(pt(pf(n,"ny",1.0f)), Y);
                IvV tz = mul(pt(pf(n,"nz",0.0f)), Z);
                float d = pf(n,"d",0.0f);
                IvV s = add(add(tx, ty), tz);
                return {b_.CreateFSub(s.lo, fc(d)), b_.CreateFSub(s.hi, fc(d))};
            }
            case K::Union:        return imin(node(*n.children[0],X,Y,Z), node(*n.children[1],X,Y,Z));
            case K::Intersection: return imax(node(*n.children[0],X,Y,Z), node(*n.children[1],X,Y,Z));
            case K::Difference:   return imax(node(*n.children[0],X,Y,Z), neg(node(*n.children[1],X,Y,Z)));
            case K::SmoothUnion: {
                IvV h = imin(node(*n.children[0],X,Y,Z), node(*n.children[1],X,Y,Z));
                float k = pf(n,"k",0.1f);
                return {b_.CreateFSub(h.lo, fc(k)), h.hi};
            }
            case K::Negate: return neg(node(*n.children[0],X,Y,Z));
            case K::Translate:
                return node(*n.children[0],
                    {b_.CreateFSub(X.lo, fc(pf(n,"tx",0.0f))), b_.CreateFSub(X.hi, fc(pf(n,"tx",0.0f)))},
                    {b_.CreateFSub(Y.lo, fc(pf(n,"ty",0.0f))), b_.CreateFSub(Y.hi, fc(pf(n,"ty",0.0f)))},
                    {b_.CreateFSub(Z.lo, fc(pf(n,"tz",0.0f))), b_.CreateFSub(Z.hi, fc(pf(n,"tz",0.0f)))});
            case K::Scale: {
                float sx = pf(n,"sx",1.0f), sy = pf(n,"sy",1.0f), sz = pf(n,"sz",1.0f);
                float mn = std::min(std::abs(sx), std::min(std::abs(sy), std::abs(sz)));
                IvV c = node(*n.children[0], divi(X, pt(sx)), divi(Y, pt(sy)), divi(Z, pt(sz)));
                auto* a0 = b_.CreateFMul(c.lo, fc(mn));
                auto* a1 = b_.CreateFMul(c.hi, fc(mn));
                return {fmin(a0, a1), fmax(a0, a1)};
            }
            case K::RotateY: {
                float ang = pf(n,"a",0.0f), cs = std::cos(ang), sn = std::sin(ang);
                IvV nx = add(mul(pt(cs), X), mul(pt(sn), Z));
                IvV nz = add(mul(pt(-sn), X), mul(pt(cs), Z));
                return node(*n.children[0], nx, Y, nz);
            }
            case K::RotateX: {
                float ang = pf(n,"a",0.0f), cs = std::cos(ang), sn = std::sin(ang);
                IvV ny = add(mul(pt(cs), Y), mul(pt(sn), Z));
                IvV nz = add(mul(pt(-sn), Y), mul(pt(cs), Z));
                return node(*n.children[0], X, ny, nz);
            }
            case K::RotateZ: {
                float ang = pf(n,"a",0.0f), cs = std::cos(ang), sn = std::sin(ang);
                IvV nx = add(mul(pt(cs), X), mul(pt(sn), Y));
                IvV ny = add(mul(pt(-sn), X), mul(pt(cs), Y));
                return node(*n.children[0], nx, ny, Z);
            }
            case K::TwistY: {
                float k = pf(n,"k",1.0f);
                IvV ang = mul(pt(k), Y), ca = icos(ang), sa = isin(ang);
                IvV nx = add(mul(ca, X), mul(sa, Z));
                IvV nz = add(mul({fc(-1.0f), fc(1.0f)}, mul(sa, X)), mul(ca, Z));
                return node(*n.children[0], nx, Y, nz);
            }
            case K::BendXY: {
                float k = pf(n,"k",1.0f); float ks = std::fabs(k) < 1e-6f ? 1e-6f : k;
                float invk = 1.0f / ks;
                IvV th = mul(pt(ks), X), r = add(pt(invk), Y);
                IvV nx = mul(r, isin(th));
                IvV ny = sub(mul(r, icos(th)), pt(invk));
                IvV child = node(*n.children[0], nx, ny, Z);
                IvV ksr = absi(mul(pt(ks), r));
                IvV lip = {fmax(fc(1.0f), ksr.lo), fmax(fc(1.0f), ksr.hi)};
                return divi(child, lip);
            }
            case K::TaperY: {
                float t = pf(n,"t",1.0f), h = pf(n,"h",1.0f);
                IvV u = mul(add(Y, pt(0.5f*h)), pt(1.0f/h));
                u = {clampv(u.lo, 0.0f, 1.0f), clampv(u.hi, 0.0f, 1.0f)};
                IvV s = add(pt(1.0f), mul(u, pt(t-1.0f)));
                s = {fmax(s.lo, fc(1e-3f)), fmax(s.hi, fc(1e-3f))};
                IvV is = divi(pt(1.0f), s);
                IvV child = node(*n.children[0], mul(X, is), Y, mul(Z, is));
                IvV lip = {fmax(fc(1.0f), is.lo), fmax(fc(1.0f), is.hi)};
                return divi(child, lip);
            }
            case K::Scene:
                return n.children.empty() ? pt(1e30f) : node(*n.children[0],X,Y,Z);
            case K::Instance:
                return (n.children.empty() || !n.children[0])
                    ? pt(1e30f) : node(*n.children[0],X,Y,Z);
            default: break;
        }
        // CustomExpr / plugin / mesh: no interval rule -> unbounded (disables the
        // cull for this object, soundly).
        return {fc(-1e30f), fc(1e30f)};
    }
};

} // namespace frep::jit
