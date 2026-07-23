// core/frep/primitives_ad.cpp
//
// AD codegen implementations for the basic primitives.

#include "ad_ir.hpp"
#include "primitives.hpp"

namespace frep {

namespace ai = ad_ir;

// ── Sphere: f = sqrt(x^2+y^2+z^2) - r ──────────────────────────────────────
FRepNode::DualVal SphereNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    // |p|^2 = x^2+y^2+z^2
    auto x2 = ai::mul(c, x, x);
    auto y2 = ai::mul(c, y, y);
    auto z2 = ai::mul(c, z, z);
    auto sum = ai::add(c, ai::add(c, x2, y2), z2);
    auto len = ai::sqrt(c, sum);
    return ai::sub_s(c, len, c.param_value(id, "r", params.at("r")));
}

// ── Box: f = max(|x|-hx, max(|y|-hy, |z|-hz)) ──────────────────────────────
FRepNode::DualVal BoxNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    auto dx = ai::sub_s(c, ai::fabs(c, x), c.param_value(id, "hx", params.at("hx")));
    auto dy = ai::sub_s(c, ai::fabs(c, y), c.param_value(id, "hy", params.at("hy")));
    auto dz = ai::sub_s(c, ai::fabs(c, z), c.param_value(id, "hz", params.at("hz")));
    return ai::max(c, dx, ai::max(c, dy, dz));
}

// ── Plane: f = dot(n, p) + d ────────────────────────────────────────────────
FRepNode::DualVal PlaneNode::codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
    // dot = nx*x + ny*y + nz*z
    auto nx = ai::mul_s(c, x, c.param_value(id, "nx", params.at("nx")));
    auto ny = ai::mul_s(c, y, c.param_value(id, "ny", params.at("ny")));
    auto nz = ai::mul_s(c, z, c.param_value(id, "nz", params.at("nz")));
    auto dot = ai::add(c, ai::add(c, nx, ny), nz);
    return ai::add_s(c, dot, c.param_value(id, "d", params.at("d")));
}

} // namespace frep
