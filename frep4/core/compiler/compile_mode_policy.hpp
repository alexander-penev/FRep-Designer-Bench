#pragma once
// core/compiler/compile_mode_policy.hpp
//
// Policy class that decides whether to use Constant or Incremental codegen
// at any given compile request. Implements the "Auto" mode: track recent
// recompile rate and switch to Incremental when it exceeds a threshold.
//
// Usage:
//   CompileModePolicy policy(TracerConfig::CompileMode::Auto);
//   // For each render request that may trigger a recompile:
//   policy.note_recompile();
//   TracerConfig cfg = base_cfg;
//   cfg.incremental_params = policy.use_incremental();
//   // ... emit and JIT with cfg ...

#include "core/compiler/codegen.hpp"

#include <chrono>
#include <deque>

namespace frep {

class CompileModePolicy {
public:
    using Mode  = TracerConfig::CompileMode;
    using Clock = std::chrono::steady_clock;

    explicit CompileModePolicy(Mode mode = Mode::Constant)
        : mode_(mode) {}

    // Switches the requested mode. Resets the auto-policy's history.
    void set_mode(Mode mode) {
        mode_ = mode;
        history_.clear();
        sticky_incremental_ = false;
    }
    Mode mode() const { return mode_; }

    // Should be called every time the host triggers a recompile (i.e.
    // emit_render_tile + JIT load). Records the timestamp; trims any
    // older than `window_` seconds.
    void note_recompile() {
        auto now = Clock::now();
        history_.push_back(now);
        const auto cutoff = now - window_;
        while (!history_.empty() && history_.front() < cutoff) {
            history_.pop_front();
        }
        // Latch into Incremental once the threshold is exceeded — staying
        // there avoids thrashing when the user pauses briefly.
        if (mode_ == Mode::Auto && !sticky_incremental_
            && static_cast<int>(history_.size()) >= threshold_)
        {
            sticky_incremental_ = true;
        }
    }

    // Resets the auto-policy when the scene structure changes (which
    // invalidates the slot table anyway).
    void reset_auto_state() {
        history_.clear();
        sticky_incremental_ = false;
    }

    // The decision: should this next compile be Incremental?
    bool use_incremental() const {
        switch (mode_) {
            case Mode::Constant:    return false;
            case Mode::Incremental: return true;
            case Mode::Auto:        return sticky_incremental_;
        }
        return false;
    }

    // Tunables.
    void set_threshold(int n)            { threshold_ = n; }
    void set_window(std::chrono::seconds w) { window_ = w; }
    int  threshold() const               { return threshold_; }
    std::chrono::seconds window() const  { return window_; }
    int  history_size() const            { return static_cast<int>(history_.size()); }

private:
    Mode                            mode_;
    std::deque<Clock::time_point>   history_;
    bool                            sticky_incremental_ = false;
    int                             threshold_          = 3;   // 3 recompiles
    std::chrono::seconds            window_             = std::chrono::seconds(5);  // within 5 s
};

} // namespace frep
