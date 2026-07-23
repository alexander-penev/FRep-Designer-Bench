#pragma once
// plugins/extra_primitives.hpp
//
// Example plugins demonstrating the PrimitivePlugin concept.
// The two plugins (Torus, Octahedron) are registered at startup
// and become available through the PluginRegistry.

#include "core/frep/node.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/compiler/llvm_compat.hpp"
#include <cmath>

#include <llvm/IR/Intrinsics.h>

#include <array>
#include <span>

namespace frep {

// ─────────────────────────────────────────────────────────────────────────────
// TorusNode — a true SDF torus
// f(p) = length(vec2(length(p.xz) - R, p.y)) - r
//   R = major radius, r = minor radius
// ─────────────────────────────────────────────────────────────────────────────
class TorusNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Torus"; }
public:
    TorusNode(float R, float r, std::string nid = "torus") {
        kind = NodeKind::Sphere;   // reused for the demo
        id = std::move(nid);
        params["R"] = R;
        params["r"] = r;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b  = c.b;
        // q.x = sqrt(x^2+z^2) - R
        auto x2  = b.CreateFMul(x, x);
        auto z2  = b.CreateFMul(z, z);
        auto xz  = b.CreateFAdd(x2, z2);
        auto len_xz = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, xz);
        auto qx  = b.CreateFSub(len_xz, c.fc(params.at("R")));
        // q.y = y
        // result = sqrt(qx^2 + y^2) - r
        auto qx2 = b.CreateFMul(qx, qx);
        auto y2  = b.CreateFMul(y, y);
        auto q2  = b.CreateFAdd(qx2, y2);
        auto len_q = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, q2);
        return b.CreateFSub(len_q, c.fc(params.at("r")), "torus");
    }

    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("R")) * 31
             ^ std::hash<float>{}(params.at("r"))
             ^ 0xFEED'F00Dull;
    }

    float eval(float x, float y, float z) const override {
        float R = params.at("R"), r = params.at("r");
        float qx = std::sqrt(x*x + z*z) - R;
        return std::sqrt(qx*qx + y*y) - r;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// OctahedronNode — a true SDF octahedron (L1 norm)
// f(p) = (|x| + |y| + |z| - s) * 0.57735
// ─────────────────────────────────────────────────────────────────────────────
class OctahedronNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Octahedron"; }
public:
    explicit OctahedronNode(float size, std::string nid = "oct") {
        kind = NodeKind::Sphere;
        id = std::move(nid);
        params["size"] = size;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        auto ax = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, x);
        auto ay = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, y);
        auto az = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, z);
        auto sum = b.CreateFAdd(b.CreateFAdd(ax, ay), az);
        auto diff = b.CreateFSub(sum, c.fc(params.at("size")));
        return b.CreateFMul(diff, c.fc(0.57735027f), "oct");
    }

    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("size")) ^ 0xACDC'1234ull;
    }

    float eval(float x, float y, float z) const override {
        float s = params.at("size");
        return (std::abs(x) + std::abs(y) + std::abs(z) - s) * 0.57735027f;
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Plugin wrappers (PrimitivePlugin concept)
// ═════════════════════════════════════════════════════════════════════════════

struct TorusPlugin {
    plugin::PluginInfo info() const {
        return {"torus", "1.0", "Torus with a major and minor radius"};
    }
    std::span<const std::string_view> param_names() const {
        static constexpr std::array<std::string_view, 2> n = {"R", "r"};
        return n;
    }
    std::span<const float> param_defaults() const {
        static constexpr std::array<float, 2> d = {1.5f, 0.4f};
        return d;
    }
    FRepNode::Ptr create(std::span<const float> p, std::string id) const {
        return std::make_shared<TorusNode>(p[0], p[1], std::move(id));
    }
};

struct OctahedronPlugin {
    plugin::PluginInfo info() const {
        return {"octahedron", "1.0", "Octahedron (L1 norm)"};
    }
    std::span<const std::string_view> param_names() const {
        static constexpr std::array<std::string_view, 1> n = {"size"};
        return n;
    }
    std::span<const float> param_defaults() const {
        static constexpr std::array<float, 1> d = {1.0f};
        return d;
    }
    FRepNode::Ptr create(std::span<const float> p, std::string id) const {
        return std::make_shared<OctahedronNode>(p[0], std::move(id));
    }
};

static_assert(plugin::PrimitivePlugin<TorusPlugin>);
static_assert(plugin::PrimitivePlugin<OctahedronPlugin>);

// ─────────────────────────────────────────────────────────────────────────────
// Registration (compile-time)
// ─────────────────────────────────────────────────────────────────────────────
inline void register_extra_primitives_into(plugin::PluginRegistry& reg) {
    reg.register_primitive(TorusPlugin{});
    reg.register_primitive(OctahedronPlugin{});
}

inline void register_extra_primitives() {
    register_extra_primitives_into(plugin::PluginRegistry::instance());
}

} // namespace frep
