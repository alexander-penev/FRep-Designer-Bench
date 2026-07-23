// plugins/dynamic/capsule_plugin.cpp
//
// Example of a dynamic plugin (.so file).
// Built as a SHARED library and loaded via dlopen.
//
// A capsule is a cylinder with rounded ends:
//   f(p) = length(p - clamp(p.y, -h, h) * (0,1,0)) - r

#include "core/frep/node.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/compiler/llvm_compat.hpp"

#include <llvm/IR/Intrinsics.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace frep {

// ─────────────────────────────────────────────────────────────────────────────
// CapsuleNode — a vertical capsule (Y-axis aligned)
// f(x,y,z) = sqrt(x^2 + max(0,|y|-h)^2 + z^2) - r
//
// Simplified definition: clamp y and measure the distance to the segment [-h, h].
// ─────────────────────────────────────────────────────────────────────────────
class CapsuleNode final : public FRepNode {
    const char* type_name() const noexcept override { return "Capsule"; }
public:
    CapsuleNode(float h, float r, std::string nid = "capsule") {
        kind = NodeKind::Plugin;  // dispatched via emit_glsl fallback
        id   = std::move(nid);
        params["height"] = h;
        params["radius"] = r;
    }

    // CPU evaluation — same SDF as the GLSL emit and LLVM codegen.
    // Without this, the picker / marching cubes / BVH path throws
    // when scenes containing a Capsule are interacted with on CPU.
    float eval(float x, float y, float z) const override {
        float h = params.at("height");
        float r = params.at("radius");
        float y_off = y - std::clamp(y, -h, h);
        return std::sqrt(x*x + y_off*y_off + z*z) - r;
    }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x, llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        float h = params.at("height");
        float r = params.at("radius");

        // y_clamped = y - clamp(y, -h, h)
        auto ny = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, y, c.fc(-h));
        auto cy = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, ny, c.fc(h));
        auto y_off = b.CreateFSub(y, cy);

        // length(x, y_off, z) - r
        auto x2 = b.CreateFMul(x, x);
        auto y2 = b.CreateFMul(y_off, y_off);
        auto z2 = b.CreateFMul(z, z);
        auto sum = b.CreateFAdd(b.CreateFAdd(x2, y2), z2);
        auto len = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, sum);
        return b.CreateFSub(len, c.fc(r), "capsule");
    }

    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("height")) * 31
             ^ std::hash<float>{}(params.at("radius"))
             ^ 0xCAFE'D00Dull;
    }

    // GPU support: emit the same capsule SDF as a GLSL expression.
    // The emitter has already evaluated children (none for a Capsule)
    // and given us a unique variable prefix we could use for locals,
    // but the formula fits in one expression so we just inline it.
    //
    // Capsule SDF (Y-axis aligned, half-height h, radius r):
    //   y_clamped = y - clamp(y, -h, h)
    //   d = length(vec3(x, y_clamped, z)) - r
    bool emit_glsl(std::ostream& out,
                   const std::vector<std::string>& /*child_exprs*/,
                   const std::string& /*var_prefix*/) const override {
        float h = params.at("height");
        float r = params.at("radius");
        // Format float literals with enough precision to round-trip.
        auto flit = [](float v) {
            char buf[32]; std::snprintf(buf, sizeof(buf), "%.7g", v);
            std::string s(buf);
            // Append .0 if it parses as an integer (GLSL requires float
            // suffix for clarity, and `2` ≠ `2.0` in some contexts).
            if (s.find('.') == std::string::npos
             && s.find('e') == std::string::npos)
                s += ".0";
            return s;
        };
        out << "(length(vec3(x, y - clamp(y, -" << flit(h) << ", "
            << flit(h) << "), z)) - " << flit(r) << ")";
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CapsulePlugin — wraps CapsuleNode under the PrimitivePlugin concept
// ─────────────────────────────────────────────────────────────────────────────
struct CapsulePlugin {
    plugin::PluginInfo info() const {
        return {"capsule", "1.0", "Capsule (cylinder with rounded ends) - dynamic plugin"};
    }
    std::span<const std::string_view> param_names() const {
        static constexpr std::array<std::string_view, 2> n = {"height", "radius"};
        return n;
    }
    std::span<const float> param_defaults() const {
        static constexpr std::array<float, 2> d = {1.0f, 0.4f};
        return d;
    }
    FRepNode::Ptr create(std::span<const float> p, std::string id) const {
        return std::make_shared<CapsuleNode>(p[0], p[1], std::move(id));
    }
};

static_assert(plugin::PrimitivePlugin<CapsulePlugin>);

} // namespace frep

// ═════════════════════════════════════════════════════════════════════════════
// Entry point — called by the plugin loader on dlopen.
// ═════════════════════════════════════════════════════════════════════════════
extern "C" void frep_plugin_register(frep::plugin::PluginRegistry& reg) {
    reg.register_primitive(frep::CapsulePlugin{});
}
