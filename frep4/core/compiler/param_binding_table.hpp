#pragma once
// core/compiler/param_binding_table.hpp
//
// ParamBindingTable — the single, backend-agnostic authority for where each
// model parameter lives (baked constant vs. runtime buffer slot) and, for the
// runtime ones, which slot in the shared parameter buffer they occupy.
//
// One deterministic scene walk plus a CompilePolicy produce a table that EVERY
// backend consults:
//
//   CPU_IR    — load from the JIT'd render_tile's `float* params`
//   GPU_IR    — OpenCL/CUDA kernel's `global float* params` (same ABI/layout)
//   GPU_GLSL  — std430 `Params { float v[]; }` at a fixed binding
//   GPU_RTX   — std430 SSBO read in the intersection/closest-hit shaders
//
// Because every backend reads the SAME slot for the SAME (node_id, param), the
// runtime buffer layout is identical across paths: the host fills one buffer
// and all executors agree on it. That is what lets a parameter edit refresh a
// buffer instead of regenerating a shader/kernel on ALL paths, not just CPU —
// provided the parameter was placed Runtime. Constant-placed parameters are
// still baked, so changing one (or changing the placement itself) is a
// structural change that does require regeneration.
//
// Determinism: the table is built by walking the node schema (a fixed,
// canonical parameter order per node kind) in pre-order, NOT by iterating the
// unordered params map. Same scene + same policy ⇒ byte-identical table across
// runs and across backends.
//
// This header is intentionally free of any LLVM/Vulkan/CUDA dependency so the
// placement logic can be unit-tested on its own. The project builds a table
// from a real FRepNode through the thin NodeView adapter (see scene_bindings).

#include "core/compiler/compile_policy.hpp"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace frep {

// NodeKind mirrored as plain ints so this header need not include the
// LLVM-coupled node.hpp. Keep in sync with frep::NodeKind.
namespace pk {
enum : int {
    Sphere = 0, Box, Plane,
    Union, Intersection, Difference, SmoothUnion,
    Negate,
    Translate, Scale, RotateY,
    TwistY, BendXY, TaperY,
    Scene,
    Plugin,
};
} // namespace pk

// One bound runtime parameter.
struct ParamSlot {
    std::string node_id;
    std::string param_name;
    int         slot = -1;       // index into the shared runtime buffer
    float       default_value = 0.0f;
    ParamClass  cls = ParamClass::Geometry;
};

// The canonical, ordered (param, class) schema a node of `kind` exposes. The
// order fixes slot assignment; classes drive class-based policies. Centralises
// what used to be the scattered `param_class` argument at each emit site.
inline const std::vector<std::pair<std::string, ParamClass>>&
node_param_schema(int kind) {
    static const std::vector<std::pair<std::string, ParamClass>> empty{};
    static const std::vector<std::pair<std::string, ParamClass>> sphere{
        {"r", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> box{
        {"hx", ParamClass::Geometry}, {"hy", ParamClass::Geometry},
        {"hz", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> plane{
        {"nx", ParamClass::Geometry}, {"ny", ParamClass::Geometry},
        {"nz", ParamClass::Geometry}, {"d", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> smooth{
        {"k", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> translate{
        {"tx", ParamClass::Geometry}, {"ty", ParamClass::Geometry},
        {"tz", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> scale{
        {"s", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> rotatey{
        {"a", ParamClass::Geometry}};
    static const std::vector<std::pair<std::string, ParamClass>> twist{
        {"k", ParamClass::Deform}};
    static const std::vector<std::pair<std::string, ParamClass>> bend{
        {"k", ParamClass::Deform}};
    static const std::vector<std::pair<std::string, ParamClass>> taper{
        {"t", ParamClass::Deform}, {"h", ParamClass::Deform}};
    switch (kind) {
        case pk::Sphere:     return sphere;
        case pk::Box:        return box;
        case pk::Plane:      return plane;
        case pk::SmoothUnion:return smooth;
        case pk::Translate:  return translate;
        case pk::Scale:      return scale;
        case pk::RotateY:    return rotatey;
        case pk::TwistY:     return twist;
        case pk::BendXY:     return bend;
        case pk::TaperY:     return taper;
        default:             return empty;  // Union/Diff/Negate/Scene/Plugin
    }
}

// (kind, param) -> class, via the schema. Geometry if unknown.
inline ParamClass classify_param(int kind, const std::string& param) {
    for (const auto& ps : node_param_schema(kind))
        if (ps.first == param) return ps.second;
    return ParamClass::Geometry;
}

class ParamBindingTable {
public:
    // A minimal read-only view of a node so the table can be built and unit
    // tested without depending on the LLVM-coupled FRepNode. The project
    // adapts a real scene into this view (children by value; trees are small).
    struct NodeView {
        int                                          kind = pk::Scene;
        std::string                                  id;
        const std::unordered_map<std::string, float>* params = nullptr;
        std::vector<NodeView>                        children;
    };

    static ParamBindingTable build(const NodeView& root,
                                   const CompilePolicy& policy) {
        ParamBindingTable t;
        t.walk(root, policy);
        return t;
    }

    // Slot for a runtime parameter, or -1 if it is Constant / unknown.
    int slot_of(const std::string& node_id, const std::string& param) const {
        auto it = slot_.find(node_id + "::" + param);
        return it == slot_.end() ? -1 : it->second;
    }
    bool is_runtime(const std::string& node_id, const std::string& param) const {
        return slot_of(node_id, param) >= 0;
    }

    const std::vector<ParamSlot>& slots() const { return slots_; }
    int  runtime_count() const { return static_cast<int>(slots_.size()); }
    bool empty()         const { return slots_.empty(); }

    // Seed values for the runtime buffer, in slot order. The host uploads
    // these once; later edits overwrite individual slots in place.
    std::vector<float> seed_buffer() const {
        std::vector<float> b(slots_.size(), 0.0f);
        for (const auto& s : slots_) b[s.slot] = s.default_value;
        return b;
    }

    // A stable fingerprint of the *placement* (which params are runtime, in
    // what slots) — but NOT of their runtime values. Backends fold this into
    // their shader/IR cache key so that editing a runtime value is a cache hit
    // while changing the placement (or a constant value) is a miss.
    std::size_t placement_hash() const {
        std::size_t h = 1469598103934665603ull;  // FNV-1a
        auto mix = [&](std::size_t v) { h ^= v; h *= 1099511628211ull; };
        for (const auto& s : slots_) {
            for (char c : s.node_id)    mix(static_cast<unsigned char>(c));
            for (char c : s.param_name) mix(static_cast<unsigned char>(c));
            mix(static_cast<std::size_t>(s.slot));
        }
        return h;
    }

private:
    std::vector<ParamSlot>              slots_;
    std::unordered_map<std::string,int> slot_;

    void walk(const NodeView& n, const CompilePolicy& policy) {
        for (const auto& ps : node_param_schema(n.kind)) {
            const std::string& name = ps.first;
            if (!n.params || !n.params->count(name)) continue;  // not set
            if (policy.decide(n.id, name, ps.second) != ParamPlacement::Runtime)
                continue;                                       // baked constant
            const std::string key = n.id + "::" + name;
            if (slot_.count(key)) continue;                     // already bound
            int slot = static_cast<int>(slots_.size());
            slot_[key] = slot;
            slots_.push_back({n.id, name, slot,
                              n.params->at(name), ps.second});
        }
        for (const auto& c : n.children) walk(c, policy);
    }
};

} // namespace frep
