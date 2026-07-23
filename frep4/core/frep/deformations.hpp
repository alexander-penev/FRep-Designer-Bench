#pragma once
// core/frep/deformations.hpp
//
// Non-linear F-Rep deformations: TwistY, BendXY, TaperY.
//
// These are the standard "warp the space, then evaluate the child SDF in
// the warped space" trick. They produce visually striking results from
// simple primitives — a cylinder twisted around its axis, a box bent over
// an arc, a cone made by tapering a cylinder. All three are commonly used
// in F-Rep / signed-distance modeling.
//
// Important: deforming an SDF non-uniformly produces an *approximate* SDF.
// The level set is still correct (zero crossings unchanged), but distances
// to the surface are scaled by the local Jacobian. We follow common
// practice and divide the resulting distance by a conservative Lipschitz
// constant so sphere tracing remains stable — this is the same trick used
// in Inigo Quilez's blog posts on the topic. We use a coarse upper bound;
// it slows tracing a touch but never produces over-shoot.
//
// AABBs of deformations are hard to compute analytically. We return the
// child's AABB scaled outward by an axis-dependent factor — conservative
// and correct, just not tight. Picking and BVH will work; just less culling.

#include "core/compiler/llvm_compat.hpp"
#include "core/frep/node.hpp"

#include <cmath>
#include <memory>
#include <string>

namespace frep {

// ── TwistY ──────────────────────────────────────────────────────────────────
// Rotates space around the Y axis by an angle that is linear in y:
//   angle(y) = k * y
// The 'k' parameter is the twist rate in radians per unit. Inverse warp:
//   x' =  cos(k*y)*x + sin(k*y)*z
//   z' = -sin(k*y)*x + cos(k*y)*z
//   y' =  y
class TwistYNode final : public FRepNode {
public:
    TwistYNode(FRepNode::Ptr child, float k, std::string nid = "twist") {
        kind = NodeKind::TwistY;
        id   = std::move(nid);
        params["k"] = k;
        children = {std::move(child)};
    }

    const char* type_name() const noexcept override { return "TwistY"; }

    AABB aabb() const noexcept override {
        // Conservative bound: extent in X and Z grows as the twist sweeps
        // the corners around. We just return the child bbox enlarged by
        // sqrt(2) in X/Z. Picking and BVH stay correct.
        AABB a = children[0]->aabb();
        float mx = std::max(std::abs(a.min_x), std::abs(a.max_x));
        float mz = std::max(std::abs(a.min_z), std::abs(a.max_z));
        float r = 1.4142f * std::max(mx, mz);
        a.min_x = std::min(a.min_x, -r);
        a.max_x = std::max(a.max_x,  r);
        a.min_z = std::min(a.min_z, -r);
        a.max_z = std::max(a.max_z,  r);
        return a;
    }

    float eval(float x, float y, float z) const override {
        float k = params.at("k");
        float a = k * y;
        float ca = std::cos(a), sa = std::sin(a);
        float xr =  ca*x + sa*z;
        float zr = -sa*x + ca*z;
        // Lipschitz correction: locally the warp's Jacobian has spectral
        // norm <= sqrt(1 + (k*r)^2). We approximate r by xz-radius.
        float r = std::sqrt(x*x + z*z);
        float lip = std::sqrt(1.0f + (k*r)*(k*r));
        return children[0]->eval(xr, y, zr) / lip;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto k = c.param_value(id, "k", params.at("k"));
        auto a = b.CreateFMul(k, y);
        auto ca = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::cos, a);
        auto sa = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sin, a);
        auto xr = b.CreateFAdd(b.CreateFMul(ca, x), b.CreateFMul(sa, z), "twist_x");
        auto zr = b.CreateFSub(b.CreateFMul(ca, z), b.CreateFMul(sa, x), "twist_z");
        auto inner = children[0]->codegen(c, xr, y, zr);
        // Lipschitz divisor: sqrt(1 + (k*r)^2) where r^2 = x^2 + z^2.
        auto r2  = b.CreateFAdd(b.CreateFMul(x, x), b.CreateFMul(z, z));
        auto kr2 = b.CreateFMul(b.CreateFMul(k, k), r2);
        auto one = c.fc(1.0f);
        auto lip2 = b.CreateFAdd(one, kr2);
        auto lip  = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, lip2);
        return b.CreateFDiv(inner, lip, "twist_sdf");
    }

    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;

    std::size_t structural_hash() const noexcept override {
        return children[0]->structural_hash() ^ 0xB1AC'E11Cull;
    }
};

// ── BendXY ──────────────────────────────────────────────────────────────────
// Bends space in the XY plane around the origin. The Y axis curves into
// an arc of radius 1/k; x = 0 lies on the spine, +y is "outward".
//
//   theta = k * x
//   r     = 1/k + y     (radial coordinate; for k=0 we'd have linear y,
//                         so we treat k≈0 as no-op below.)
//   x' = r * sin(theta) - 1/k
//   y' = r * cos(theta) - 1/k
//   z' = z
//
// Caveat: for |k*x| >> π/2 the warp becomes self-overlapping and the
// resulting field is no longer single-valued. The user is expected to
// keep the input within the safe envelope.
class BendXYNode final : public FRepNode {
public:
    BendXYNode(FRepNode::Ptr child, float k, std::string nid = "bend") {
        kind = NodeKind::BendXY;
        id   = std::move(nid);
        params["k"] = k;
        children = {std::move(child)};
    }

    const char* type_name() const noexcept override { return "BendXY"; }

    AABB aabb() const noexcept override {
        // Conservative bound: bending an extent of L by curvature k turns
        // it into an arc of radius 1/k swept over angle k*L. Bbox grows
        // by at most the arc's diameter. We enlarge by max(child extent)
        // in both X and Y directions.
        AABB a = children[0]->aabb();
        float r = std::max(a.max_x - a.min_x, a.max_y - a.min_y);
        a.min_x -= r; a.max_x += r;
        a.min_y -= r; a.max_y += r;
        return a;
    }

    float eval(float x, float y, float z) const override {
        float k = params.at("k");
        if (std::abs(k) < 1e-6f) return children[0]->eval(x, y, z);
        float invk  = 1.0f / k;
        float theta = k * x;
        float r     = invk + y;
        float xr    = r * std::sin(theta);
        float yr    = r * std::cos(theta) - invk;
        // Lipschitz correction analogous to twist.
        float lip = std::max(1.0f, std::abs(k) * std::abs(r));
        return children[0]->eval(xr, yr, z) / lip;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto k    = c.param_value(id, "k", params.at("k"));
        auto invk = b.CreateFDiv(c.fc(1.0f), k);
        auto th   = b.CreateFMul(k, x);
        auto rv   = b.CreateFAdd(invk, y);
        auto sth  = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sin, th);
        auto cth  = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::cos, th);
        auto xr   = b.CreateFMul(rv, sth, "bend_x");
        auto yr   = b.CreateFSub(b.CreateFMul(rv, cth), invk, "bend_y");
        auto inner = children[0]->codegen(c, xr, yr, z);
        // Lip = max(1, |k * rv|). Use fabs intrinsic.
        auto kr   = b.CreateFMul(k, rv);
        auto akr  = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, kr);
        auto lip  = frep::llvm_compat::max_num(b, c.fc(1.0f), akr);
        return b.CreateFDiv(inner, lip, "bend_sdf");
    }

    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;

    std::size_t structural_hash() const noexcept override {
        return children[0]->structural_hash() ^ 0xBE7D'DAFEull;
    }
};

// ── TaperY ──────────────────────────────────────────────────────────────────
// Scales the XZ plane linearly with Y. At y = -h/2 the scale is 1.0; at
// y = +h/2 the scale is `t` (e.g. 0 gives a cone tip; 0.5 gives a frustum).
// The inverse warp is to *divide* X and Z by the scale at that height.
//
//   s(y) = lerp(1, t, (y + h/2) / h)  = clamped to [min(1,t), max(1,t)]
//   x' = x / s(y)
//   z' = z / s(y)
//   y' = y
//
// We keep s above a small floor (1e-3) so we never divide by zero — this
// caps the maximum sharpness at the tip but keeps tracing stable.
class TaperYNode final : public FRepNode {
public:
    // h: total height of the taper region. t: scale factor at y = +h/2.
    TaperYNode(FRepNode::Ptr child, float t, float h = 2.0f,
               std::string nid = "taper")
    {
        kind = NodeKind::TaperY;
        id   = std::move(nid);
        params["t"] = t;
        params["h"] = h;
        children = {std::move(child)};
    }

    const char* type_name() const noexcept override { return "TaperY"; }

    AABB aabb() const noexcept override {
        // Bbox: child bbox scaled outward by max(1, t) in X/Z.
        AABB a = children[0]->aabb();
        float t = params.at("t");
        float s = std::max(1.0f, std::abs(t));
        a.min_x *= s; a.max_x *= s;
        a.min_z *= s; a.max_z *= s;
        return a;
    }

    float eval(float x, float y, float z) const override {
        float t = params.at("t");
        float h = params.at("h");
        float u = std::clamp((y + 0.5f*h) / h, 0.0f, 1.0f);
        float s = 1.0f + u * (t - 1.0f);
        s = std::max(s, 1e-3f);
        float xr = x / s;
        float zr = z / s;
        // Lipschitz: 1/s on x and z, 1 on y → spectral norm 1/s when s < 1,
        // 1 otherwise. Use the conservative 1/min(s,1).
        float lip = std::max(1.0f, 1.0f / s);
        return children[0]->eval(xr, y, zr) / lip;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto t = c.param_value(id, "t", params.at("t"));
        auto h = c.param_value(id, "h", params.at("h"));

        auto half_h = b.CreateFMul(h, c.fc(0.5f));
        auto u_raw  = b.CreateFDiv(b.CreateFAdd(y, half_h), h);
        // Clamp u to [0, 1].
        auto u_lo   = frep::llvm_compat::max_num(b, u_raw, c.fc(0.0f));
        auto u      = frep::llvm_compat::min_num(b, u_lo,  c.fc(1.0f));

        // s = 1 + u * (t - 1)
        auto t_m_1 = b.CreateFSub(t, c.fc(1.0f));
        auto s_raw = b.CreateFAdd(c.fc(1.0f), b.CreateFMul(u, t_m_1));
        // Floor to 1e-3 so the divide is safe.
        auto s = frep::llvm_compat::max_num(b, s_raw, c.fc(1e-3f));
        auto inv_s = b.CreateFDiv(c.fc(1.0f), s);

        auto xr = b.CreateFMul(x, inv_s, "taper_x");
        auto zr = b.CreateFMul(z, inv_s, "taper_z");
        auto inner = children[0]->codegen(c, xr, y, zr);

        // lip = max(1, 1/s)
        auto lip = frep::llvm_compat::max_num(b, c.fc(1.0f), inv_s);
        return b.CreateFDiv(inner, lip, "taper_sdf");
    }

    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;

    std::size_t structural_hash() const noexcept override {
        return children[0]->structural_hash() ^ 0x7AE6'EAB1ull;
    }
};

} // namespace frep
