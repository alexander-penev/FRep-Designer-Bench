#pragma once
// core/plugin/plugin_api.hpp
//
// Plugin API for F-Rep Designer 4.0.
// Uses C++23 concepts (conceptually C++26-ready).
//
// Plugin kinds:
//   1. PrimitivePlugin — new F-Rep primitives (Sphere, Torus, Mandelbulb...)
//   2. OperationPlugin — new CSG/blending operations (BoundedBlend, Twist...)
//   3. RayTracerPlugin — alternative visualization algorithms
//   4. RetargetPlugin — new target platforms (SPIR-V, CUDA, FPGA...)

#include "core/frep/node.hpp"

#include <llvm/IR/Module.h>

#include <concepts>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace frep::plugin {

// ─────────────────────────────────────────────────────────────────────────────
// PluginInfo — common metadata
// ─────────────────────────────────────────────────────────────────────────────
struct PluginInfo {
    std::string_view name;
    std::string_view version;
    std::string_view description;
};

// ─────────────────────────────────────────────────────────────────────────────
// PrimitivePlugin
//
// Defines a new FRepNode-based primitive with GUI metadata.
// ─────────────────────────────────────────────────────────────────────────────
template<typename P>
concept PrimitivePlugin = requires(P p, std::span<const float> params) {
    { p.info() } -> std::same_as<PluginInfo>;
    // Parameter names for GUI generation.
    { p.param_names() } -> std::convertible_to<std::span<const std::string_view>>;
    // Default parameter values.
    { p.param_defaults() } -> std::convertible_to<std::span<const float>>;
    // Creates a new FRepNode from the parameter values.
    { p.create(params, std::string{}) } -> std::convertible_to<FRepNode::Ptr>;
};

// ─────────────────────────────────────────────────────────────────────────────
// RayTracerPlugin
//
// Defines an alternative ray tracer algorithm.
// Receives the scene_sdf, scene_normal, scene_material and shader functions
// (already emitted in the module) and must emit a "render_tile" function.
// ─────────────────────────────────────────────────────────────────────────────
struct TracerInputs {
    llvm::Function* sdf_fn;
    llvm::Function* normal_fn;
    llvm::Function* material_fn;
    llvm::Function* shader_fn;
    llvm::Module*   module;
};

template<typename T>
concept RayTracerPlugin = requires(T t, TracerInputs in) {
    { t.info() } -> std::same_as<PluginInfo>;
    // Emits a render_tile function into the given module.
    { t.emit_tracer(in) } -> std::convertible_to<llvm::Function*>;
};

// ─────────────────────────────────────────────────────────────────────────────
// RetargetPlugin
//
// Transforms an LLVM IR module into code for an alternative target platform.
// Returns serialized bytes (SPIR-V, PTX, FPGA bitstream, etc.).
// ─────────────────────────────────────────────────────────────────────────────
struct RetargetResult {
    std::vector<std::uint8_t> bytes;     // serialized code
    std::string               assembly;  // human-readable form (for debug)
};

template<typename R>
concept RetargetPlugin = requires(R r, llvm::Module& m) {
    { r.info() } -> std::same_as<PluginInfo>;
    { r.target_triple() } -> std::convertible_to<std::string_view>;
    { r.retarget(m) } -> std::same_as<std::expected<RetargetResult, std::string>>;
};

// ─────────────────────────────────────────────────────────────────────────────
// PluginRegistry — a runtime registry of plugins
//
// For the PoC: a static registry (compile-time registration).
// For production: dynamic loading via dlopen / Win32 LoadLibrary.
// ─────────────────────────────────────────────────────────────────────────────
class PluginRegistry {
public:
    static PluginRegistry& instance() {
        static PluginRegistry r;
        return r;
    }

    // Type-erased holders — std::function-style.
    struct PrimitiveSlot {
        PluginInfo                 info;
        // FRepNode::type_name() of the produced node — used by scene_io for
        // JSON deserialization (the stable type identifier in saved scenes).
        // Auto-derived during register_primitive() by creating a probe node.
        std::string                node_type_name;
        std::span<const std::string_view> param_names;
        std::span<const float>     param_defaults;
        std::function<FRepNode::Ptr(std::span<const float>, std::string)> create;
    };
    struct RetargetSlot {
        PluginInfo                                                       info;
        std::string                                                      triple;
        std::function<std::expected<RetargetResult, std::string>(llvm::Module&)> retarget;
    };

    template<PrimitivePlugin P>
    void register_primitive(P plugin) {
        // Probe the plugin to find out what FRepNode::type_name() its nodes
        // report. We create a throwaway instance with default params just
        // to read type_name() from it.
        auto probe = plugin.create(plugin.param_defaults(), "_probe");
        std::string type_name_str = probe ? probe->type_name() : "";

        primitives_.push_back(PrimitiveSlot{
            plugin.info(),
            std::move(type_name_str),
            plugin.param_names(),
            plugin.param_defaults(),
            [p = std::move(plugin)](std::span<const float> params, std::string id) mutable {
                return p.create(params, std::move(id));
            }
        });
    }

    template<RetargetPlugin R>
    void register_retarget(R plugin) {
        retargets_.push_back(RetargetSlot{
            plugin.info(),
            std::string(plugin.target_triple()),
            [p = std::move(plugin)](llvm::Module& m) mutable {
                return p.retarget(m);
            }
        });
    }

    std::span<const PrimitiveSlot> primitives() const { return primitives_; }
    std::span<const RetargetSlot>  retargets()  const { return retargets_; }

    // Finds a primitive plugin by the FRepNode::type_name() its nodes produce
    // (the stable identifier used in serialized scenes). Returns nullptr if no
    // such plugin is registered.
    const PrimitiveSlot* find_primitive_by_type_name(const std::string& type_name) const {
        for (const auto& p : primitives_)
            if (p.node_type_name == type_name)
                return &p;
        return nullptr;
    }

private:
    std::vector<PrimitiveSlot> primitives_;
    std::vector<RetargetSlot>  retargets_;
};

} // namespace frep::plugin
