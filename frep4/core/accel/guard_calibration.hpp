// core/accel/guard_calibration.hpp
//
// Runtime calibration for the build-time spatial guard (SceneSdfMode::
// Guarded). Whether guarding an object pays off depends on per-object SDF
// cost, and where the crossover sits depends on the host CPU (vectorised
// inline min() vs a guard's branch+sqrt). Rather than hardcode a node-count
// threshold, we measure it once on the actual machine and cache it.
//
// The calibration renders a few tiny scenes (small object count, low
// resolution) at increasing per-object complexity, inlined vs guarded, and
// finds the lowest node-count at which guarded becomes faster. That
// threshold is cached to disk keyed by a CPU identifier; subsequent runs
// load it instantly. Recalibration happens when the cache is missing, the
// CPU id changed (moved machine / different build host), or on explicit
// request.
//
// Cheap by construction: tiny scenes + 96×96 render, a few compiles. The
// calibration never touches large object counts (those can take minutes to
// JIT), so it adds at most ~a second or two, once.

#pragma once

#include <optional>
#include <string>

namespace frep::accel {

struct GuardCalibration {
    // Minimum per-object node count at which Guarded beats Inlined on this
    // host. An object subtree with >= this many nodes is "expensive enough"
    // to be worth guarding. A sentinel of INT_MAX-ish (see kNeverGuard)
    // means guarding never won in the measured range — treat as "disabled".
    int   node_threshold = 0;
    // The CPU this was measured on; recalibrate if it changes.
    std::string cpu_id;
    // Wall-time the calibration itself took, for diagnostics.
    double calibration_ms = 0;
    bool   valid = false;
};

// Threshold value meaning "guarding never paid off" — heuristic should
// then always choose Inlined.
inline constexpr int kNeverGuard = 1 << 20;

// A short identifier for the current CPU (model name on Linux, else a
// generic fallback). Used as the cache key so moving to a different
// machine triggers recalibration.
std::string host_cpu_id();

// Path to the calibration cache file (under $XDG_CACHE_HOME or ~/.cache,
// falling back to /tmp). Created lazily.
std::string calibration_cache_path();

// Load a cached calibration if present and matching the current CPU.
// Returns nullopt if absent, unreadable, or for a different CPU (caller
// should then run calibrate()).
std::optional<GuardCalibration> load_calibration();

// Persist a calibration to the cache file. Returns false on I/O failure
// (non-fatal — calibration just won't be cached).
bool save_calibration(const GuardCalibration&);

// Measure the guard crossover on this host. Renders tiny inlined-vs-guarded
// scenes at increasing per-object complexity and records the lowest node
// count where guarded wins. Pure measurement; does not read or write the
// cache. Safe to call without a display (uses the offscreen CPU path).
GuardCalibration calibrate();

// Convenience: return a usable calibration, loading the cache if valid or
// calibrating (and caching) otherwise. `force` skips the cache and always
// re-measures. This is the entry point the app/tooling calls.
GuardCalibration get_or_calibrate(bool force = false);

// Decide the SDF emission mode for a scene given a calibration: Guarded
// when the scene has enough objects AND their average complexity meets the
// calibrated threshold, else Inlined. Defined inline so callers don't need
// to link extra code; takes the average node count and object count the
// caller computes from the scene.
inline bool should_guard(const GuardCalibration& cal,
                          int object_count, double avg_node_count) {
    if (!cal.valid || cal.node_threshold >= kNeverGuard) return false;
    // Guard overhead needs a handful of objects to amortise; below that
    // the inline path is fine regardless of complexity.
    constexpr int kMinObjects = 8;
    return object_count >= kMinObjects
        && avg_node_count >= (double)cal.node_threshold;
}

} // namespace frep::accel
