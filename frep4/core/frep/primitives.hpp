#pragma once
// core/frep/primitives.hpp

#include "node.hpp"
#include "core/compiler/llvm_compat.hpp"
#include <cmath>
#include <functional>

namespace frep {

// ── Sphere ────────────────────────────────────────────────────────────────────
// True SDF:  f(x,y,z) = sqrt(x^2+y^2+z^2) - r   (<=0 inside)
// Unlike the traditional F-Rep form (x^2+y^2+z^2 - r^2), here the value
// equals the Euclidean distance — important for correct sphere tracing.
class SphereNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Sphere"; }
public:
    explicit SphereNode(float r, std::string nid = "sphere") {
        kind = NodeKind::Sphere; id = std::move(nid);
        params["r"] = r;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b  = c.b;
        auto  sx = b.CreateFMul(x, x, "x2");
        auto  sy = b.CreateFMul(y, y, "y2");
        auto  sz = b.CreateFMul(z, z, "z2");
        auto  s  = b.CreateFAdd(b.CreateFAdd(sx, sy), sz, "len2");
        // unary_intrinsic: LLVM-version-compatible wrapper for unary intrinsics.
        auto  ln = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, s, "len");
        return b.CreateFSub(ln, c.param_value(id, "r", params.at("r")), "sph");
    }

    // Exact AD for the sphere SDF.
    // f = sqrt(x^2+y^2+z^2) - r
    // df/dx = x / sqrt(x^2+y^2+z^2)
    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;

    float eval(float x, float y, float z) const override {
        return std::sqrt(x*x + y*y + z*z) - params.at("r");
    }

    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("r")) ^ 0xF5E8'A1C3ull;
    }
};

// ── Box ───────────────────────────────────────────────────────────────────────
// f = max(|x|-hx, max(|y|-hy, |z|-hz))
class BoxNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Box"; }
public:
    BoxNode(float hx, float hy, float hz, std::string nid = "box") {
        kind = NodeKind::Box; id = std::move(nid);
        params["hx"] = hx; params["hy"] = hy; params["hz"] = hz;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto fabs_v = [&](llvm::Value* v) {
            return frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, v);
        };
        auto mx = [&](llvm::Value* a, llvm::Value* bv) {
            return frep::llvm_compat::max_num(b, a, bv);
        };
        auto* f0 = c.fc(0.0f);  // width-aware (splats in SIMD mode)
        auto dx = b.CreateFSub(fabs_v(x), c.param_value(id, "hx", params.at("hx")), "dx");
        auto dy = b.CreateFSub(fabs_v(y), c.param_value(id, "hy", params.at("hy")), "dy");
        auto dz = b.CreateFSub(fabs_v(z), c.param_value(id, "hz", params.at("hz")), "dz");
        // True Euclidean box SDF: length(max(d,0)) + min(max(dx,dy,dz),0).
        // Matches eval() (see the note there) — Chebyshev max(d) alone
        // under-estimates the distance to far corners, breaking BVH
        // pruning and slowing sphere-tracing.
        auto ox = mx(dx, f0), oy = mx(dy, f0), oz = mx(dz, f0);
        auto sx = b.CreateFMul(ox, ox), sy = b.CreateFMul(oy, oy), sz = b.CreateFMul(oz, oz);
        auto sum = b.CreateFAdd(sx, b.CreateFAdd(sy, sz));
        auto outside = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, sum);
        auto cheb = mx(dx, mx(dy, dz));
        auto inside = frep::llvm_compat::min_num(b, cheb, f0);
        return b.CreateFAdd(outside, inside);
    }

    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    AABB aabb() const override;

    float eval(float x, float y, float z) const override {
        // True Euclidean box SDF (Inigo Quilez). The old version returned
        // max(dx,dy,dz) — the Chebyshev (L∞) distance — which is correct
        // inside and on-axis but UNDER-estimates the distance to a far
        // corner (a point off the diagonal is sqrt-farther than its
        // largest single-axis gap). That under-estimate breaks the BVH's
        // pruning invariant (an object's AABB distance must be a lower
        // bound on its SDF) and also makes sphere-tracing take smaller
        // steps than necessary. The exact form: outside distance is the
        // length of the positive part of (|p|-half); inside distance is
        // the (negative) largest gap.
        const float dx = std::abs(x) - params.at("hx");
        const float dy = std::abs(y) - params.at("hy");
        const float dz = std::abs(z) - params.at("hz");
        const float ox = std::max(dx, 0.0f);
        const float oy = std::max(dy, 0.0f);
        const float oz = std::max(dz, 0.0f);
        const float outside = std::sqrt(ox*ox + oy*oy + oz*oz);
        const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
        return outside + inside;
    }

    std::size_t structural_hash() const noexcept override {
        std::size_t h = 0xBBB2;
        for (auto& [k, v] : params)
            h ^= std::hash<float>{}(v) + 0x9e37'79b9ull + (h << 6) + (h >> 2);
        return h;
    }
};

// ── Plane ─────────────────────────────────────────────────────────────────────
// f = dot(n, p) + d    (n is a unit normal vector)
class PlaneNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Plane"; }
public:
    PlaneNode(float nx, float ny, float nz, float d, std::string nid = "plane") {
        kind = NodeKind::Plane; id = std::move(nid);
        params["nx"] = nx; params["ny"] = ny; params["nz"] = nz; params["d"] = d;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto dot = b.CreateFAdd(
            b.CreateFAdd(
                b.CreateFMul(c.param_value(id, "nx", params.at("nx")), x),
                b.CreateFMul(c.param_value(id, "ny", params.at("ny")), y)),
            b.CreateFMul(c.param_value(id, "nz", params.at("nz")), z), "dot");
        return b.CreateFAdd(dot, c.param_value(id, "d", params.at("d")), "plane");
    }

    DualVal codegen_grad(CgCtx& c, DualVal x, DualVal y, DualVal z) const override;
    // PlaneNode does not override aabb() — the plane is infinite.

    float eval(float x, float y, float z) const override {
        return params.at("nx")*x + params.at("ny")*y + params.at("nz")*z
             + params.at("d");
    }

    std::size_t structural_hash() const noexcept override {
        std::size_t h = 0xF1A7;
        for (auto& [k, v] : params)
            h ^= std::hash<float>{}(v) + 0x9e37'79b9ull + (h << 6) + (h >> 2);
        return h;
    }
};

} // namespace frep
