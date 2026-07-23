#pragma once
// core/compiler/compile_policy.hpp
//
// CompilePolicy — decides, per parameter, where on the data↔code spectrum a
// model parameter lives: baked into the generated program as a constant, or
// read from a runtime buffer.
//
// This is the central knob of the "a model is a program" approach. Two
// extremes bracket a spectrum (see docs/ARCHITECTURE_PATHS.md):
//
//   * Constant  — the parameter is a compile-time constant. The optimiser
//     can fold it, kill dead branches, drop ×0 terms, specialise the whole
//     computation around it. Best for values that rarely change, or for a
//     one-shot final render. Cost: changing it needs a recompile.
//
//   * Runtime   — the parameter is read from a buffer at run time. Editing
//     it costs nothing (no recompile) but forgoes specialisation on it.
//     Best for values dragged interactively.
//
// The right choice is per parameter and driven by how often it changes and
// what the visualisation is for. A CompilePolicy encodes that choice. Today
// the decision is by parameter *class* (geometry, material, …); the
// interface is deliberately able to grow toward per-parameter, and later
// frequency- or statistics-driven, promotion without changing call sites.

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace frep {

// Coarse classification of a model parameter, used by policies that decide
// by class. A node passes the class when it asks codegen for a parameter
// value (see FRepNode::param_value); geometry is the default since most
// node parameters are geometric.
enum class ParamClass {
    Geometry,    // radius, extents, translation, rotation angle, …
    Material,    // albedo, roughness, metallic, …
    Deform,      // twist/bend strength, …
    Render,      // SSAA, shadow, AO, light — algorithm/observer settings
    Other,
};

// A decision for one parameter.
enum class ParamPlacement { Constant, Runtime };

// Base policy. Subclass to implement a placement rule.
class CompilePolicy {
public:
    virtual ~CompilePolicy() = default;

    // Decide where a parameter lives. `node_id`/`param_name` identify it;
    // `cls` is its class. Implementations should be pure (same inputs →
    // same output) within one compile, so the slot table stays consistent.
    virtual ParamPlacement decide(const std::string& node_id,
                                  const std::string& param_name,
                                  ParamClass cls) const = 0;
};

// Everything baked as constants — the maximal-optimisation extreme. Suitable
// for a one-shot final render of a finished scene.
class AllConstantPolicy final : public CompilePolicy {
public:
    ParamPlacement decide(const std::string&, const std::string&,
                          ParamClass) const override {
        return ParamPlacement::Constant;
    }
};

// Runtime for a chosen set of classes, constant for the rest. The current
// interactive default — geometry (and, when wanted, material/deform) is
// editable without recompiling, while render/observer settings that change
// rarely stay constant for speed.
//
// NOTE: render/lighting/camera are intentionally NOT in the runtime set for
// now. Camera is already a runtime kernel argument (not a baked param), so
// orbiting is smooth regardless; render/lighting changes are rare and may
// trigger a recompile.
class ByParamClassPolicy final : public CompilePolicy {
public:
    explicit ByParamClassPolicy(std::unordered_set<ParamClass> runtime_classes)
        : runtime_(std::move(runtime_classes)) {}

    // Convenience: the interactive default (geometry + material + deform
    // runtime; render/other constant).
    static ByParamClassPolicy interactive() {
        return ByParamClassPolicy({ParamClass::Geometry,
                                   ParamClass::Material,
                                   ParamClass::Deform});
    }

    ParamPlacement decide(const std::string&, const std::string&,
                          ParamClass cls) const override {
        return runtime_.count(cls) ? ParamPlacement::Runtime
                                   : ParamPlacement::Constant;
    }

private:
    std::unordered_set<ParamClass> runtime_;
};

// Everything read from the runtime buffer — the maximal-flexibility extreme.
// No parameter edit ever forces a recompile; the optimiser forgoes folding on
// all of them. Symmetric counterpart to AllConstantPolicy.
class AllRuntimePolicy final : public CompilePolicy {
public:
    ParamPlacement decide(const std::string&, const std::string&,
                          ParamClass) const override {
        return ParamPlacement::Runtime;
    }
};

// Per-parameter edit statistics with time decay. A policy can use these to
// promote frequently-edited parameters to Runtime and let idle ones fall back
// to Constant. `note_edit` is called by the GUI whenever a parameter value is
// changed; `decay` is called on a tick (e.g. once per second) so heat from old
// edits fades and a parameter that stops being dragged eventually demotes.
class ParamEditStats {
public:
    void note_edit(const std::string& node_id, const std::string& param) {
        heat_[node_id + "::" + param] += 1.0;
    }
    double heat(const std::string& node_id, const std::string& param) const {
        auto it = heat_.find(node_id + "::" + param);
        return it == heat_.end() ? 0.0 : it->second;
    }
    void decay(double factor = 0.5) {
        for (auto& kv : heat_) kv.second *= factor;
    }
    void clear() { heat_.clear(); }

private:
    std::unordered_map<std::string, double> heat_;
};

// Per-parameter, statistics-driven placement. A parameter becomes Runtime once
// its edit heat reaches `promote_at`; otherwise it defers to a base policy
// (e.g. ByParamClassPolicy or AllConstantPolicy) — or stays Constant if none.
//
// This realises the continuum of intermediate tiers the two extremes bracket:
// at any instant the runtime set is exactly the currently-hot parameters, so a
// scene can sit anywhere between all-constant and all-runtime and migrate as
// the user's editing focus shifts — without changing any emit call site, since
// every backend consults the resulting ParamBindingTable the same way.
class PerParamPolicy final : public CompilePolicy {
public:
    explicit PerParamPolicy(const ParamEditStats& stats,
                            double promote_at = 2.0,
                            const CompilePolicy* base = nullptr)
        : stats_(stats), promote_at_(promote_at), base_(base) {}

    ParamPlacement decide(const std::string& node_id,
                          const std::string& param_name,
                          ParamClass cls) const override {
        if (stats_.heat(node_id, param_name) >= promote_at_)
            return ParamPlacement::Runtime;
        if (base_) return base_->decide(node_id, param_name, cls);
        return ParamPlacement::Constant;
    }

private:
    const ParamEditStats& stats_;
    double                promote_at_;
    const CompilePolicy*  base_;
};

} // namespace frep
