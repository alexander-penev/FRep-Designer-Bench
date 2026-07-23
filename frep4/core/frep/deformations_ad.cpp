// core/frep/deformations_ad.cpp
//
// codegen_grad for the deformation nodes. For non-linear deformations the
// closed-form chain rule is tedious; we use central differences in IR.
// The 3 extra codegen() calls cost ~3x the SDF eval, but normals are only
// computed at hit points (one per pixel), so the overhead is negligible.

#include "core/frep/deformations.hpp"
#include "core/frep/ad_ir.hpp"

namespace frep {

namespace ai = ad_ir;

namespace {

// Central-difference gradient: f at the three offset pairs along x/y/z,
// combined with the input dot directions.
FRepNode::DualVal central_diff_grad(CgCtx& c, const FRepNode& self,
                                    FRepNode::DualVal x,
                                    FRepNode::DualVal y,
                                    FRepNode::DualVal z)
{
    auto& b = c.b;
    auto* f32 = c.f32();
    float h = 1e-3f;  // step size — small enough for smooth deformations
    auto fh    = llvm::ConstantFP::get(f32, h);
    auto inv2h = llvm::ConstantFP::get(f32, 0.5f / h);

    auto f = self.codegen(c, x.val, y.val, z.val);

    auto fxp = self.codegen(c, b.CreateFAdd(x.val, fh), y.val, z.val);
    auto fxm = self.codegen(c, b.CreateFSub(x.val, fh), y.val, z.val);
    auto fyp = self.codegen(c, x.val, b.CreateFAdd(y.val, fh), z.val);
    auto fym = self.codegen(c, x.val, b.CreateFSub(y.val, fh), z.val);
    auto fzp = self.codegen(c, x.val, y.val, b.CreateFAdd(z.val, fh));
    auto fzm = self.codegen(c, x.val, y.val, b.CreateFSub(z.val, fh));

    auto gx = b.CreateFMul(b.CreateFSub(fxp, fxm), inv2h);
    auto gy = b.CreateFMul(b.CreateFSub(fyp, fym), inv2h);
    auto gz = b.CreateFMul(b.CreateFSub(fzp, fzm), inv2h);

    auto dot = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(gx, x.dot), b.CreateFMul(gy, y.dot)),
        b.CreateFMul(gz, z.dot));
    return {f, dot};
}

} // anon

FRepNode::DualVal TwistYNode::codegen_grad(CgCtx& c,
                                           DualVal x, DualVal y, DualVal z) const {
    // Analytic forward-mode AD through the twist, matching the GLSL dual
    // emitter exactly (so CPU_IR and GPU_GLSL normals agree, not just values):
    //   a = k*y ; xr = ca*x + sa*z ; zr = ca*z - sa*x
    //   child(xr, y, zr) / sqrt(1 + (k r)^2),  r^2 = x.v^2 + z.v^2
    float k = params.at("k");
    auto a  = ai::mul_s(c, y, k);
    auto ca = ai::cos(c, a);
    auto sa = ai::sin(c, a);
    auto xr = ai::add(c, ai::mul(c, ca, x), ai::mul(c, sa, z));
    auto zr = ai::sub(c, ai::mul(c, ca, z), ai::mul(c, sa, x));
    auto inner = children[0]->codegen_grad(c, xr, y, zr);
    // Lipschitz divisor sqrt(1 + k^2 (x^2 + z^2)).
    auto r2  = ai::add(c, ai::mul(c, x, x), ai::mul(c, z, z));
    auto kr2 = ai::mul_s(c, r2, k * k);
    auto lip = ai::sqrt(c, ai::add_s(c, kr2, 1.0f));
    return ai::div(c, inner, lip);
}

FRepNode::DualVal BendXYNode::codegen_grad(CgCtx& c,
                                           DualVal x, DualVal y, DualVal z) const {
    return central_diff_grad(c, *this, x, y, z);
}

FRepNode::DualVal TaperYNode::codegen_grad(CgCtx& c,
                                           DualVal x, DualVal y, DualVal z) const {
    // Analytic forward-mode AD through the taper, matching the GLSL dual
    // emitter and the float codegen exactly:
    //   u = clamp((y + 0.5h)/h, 0, 1) ; s = max(1 + u(t-1), 1e-3)
    //   child(x/s, y, z/s) / max(1, 1/s)
    float t = params.at("t");
    float h = params.count("h") ? params.at("h") : 2.0f;
    auto u_raw = ai::mul_s(c, ai::add_s(c, y, 0.5f * h), 1.0f / h);
    auto u     = ai::clamp_s(c, u_raw, 0.0f, 1.0f);
    auto s     = ai::max_s(c, ai::add_s(c, ai::mul_s(c, u, t - 1.0f), 1.0f), 1e-3f);
    auto xr = ai::div(c, x, s);
    auto zr = ai::div(c, z, s);
    auto inner = children[0]->codegen_grad(c, xr, y, zr);
    // Lipschitz divisor max(1, 1/s).
    auto inv_s = ai::div(c, ai::constant(c, 1.0f), s);
    auto lip   = ai::max_s(c, inv_s, 1.0f);
    return ai::div(c, inner, lip);
}

} // namespace frep
