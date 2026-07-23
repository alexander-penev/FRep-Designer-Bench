#pragma once
// core/compiler/incremental_params.hpp
//
// Helper for the runtime parameter buffer used by Incremental compilation
// mode. Wraps a float[] that the JIT'd render_tile reads on every call.
//
// Typical use:
//   SceneCodegen cg(ctx, {.incremental_params = true});
//   cg.emit_render_tile(scene);
//   IncrementalParams ip(cg);              // copies the binding table
//   // ... JIT and load the module ...
//   TileScheduler::render(fn, out, cam, rp, ip.buffer());
//   ip.set("ball", "r", 0.5f);             // edit a parameter
//   TileScheduler::render(fn, out, cam, rp, ip.buffer());  // re-renders, no JIT

#include "core/compiler/codegen.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace frep {

class IncrementalParams {
public:
    // Captures the binding table and seeds the buffer with each binding's
    // default value. The codegen must have already populated its bindings
    // (i.e. emit_render_tile was called with incremental_params=true).
    explicit IncrementalParams(const SceneCodegen& cg) {
        int n = cg.param_slot_count();
        buffer_.assign(n, 0.0f);
        for (const auto& b : cg.param_bindings()) {
            buffer_[b.slot] = b.default_value;
            key_to_slot_[b.node_id + "::" + b.param_name] = b.slot;
        }
    }

    // Updates the value for a specific (node_id, param_name) — no-op if
    // the parameter is not in the binding table (e.g. not a primitive/
    // transform parameter, or the codegen ran in Constant mode).
    // Returns true if the value was actually written.
    bool set(const std::string& node_id,
             const std::string& param_name,
             float value)
    {
        auto it = key_to_slot_.find(node_id + "::" + param_name);
        if (it == key_to_slot_.end()) return false;
        buffer_[it->second] = value;
        return true;
    }

    // Reads back the current value of a parameter.
    float get(const std::string& node_id,
              const std::string& param_name) const
    {
        auto it = key_to_slot_.find(node_id + "::" + param_name);
        if (it == key_to_slot_.end()) return 0.0f;
        return buffer_[it->second];
    }

    // Whether this (node, param) pair is in the binding table.
    bool has(const std::string& node_id,
             const std::string& param_name) const
    {
        return key_to_slot_.count(node_id + "::" + param_name) > 0;
    }

    // Raw access for the JIT'd function. Pointer remains valid for the
    // lifetime of this object.
    float*       buffer()       { return buffer_.data(); }
    const float* buffer() const { return buffer_.data(); }
    int          size()   const { return static_cast<int>(buffer_.size()); }

private:
    std::vector<float>                     buffer_;
    std::unordered_map<std::string, int>   key_to_slot_;
};

} // namespace frep
