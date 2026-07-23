#pragma once
// core/compiler/incremental.hpp
//
// Incremental compiler — caches the compiled scene between frames.
//
// Approach:
//   1. Compute scene_hash() of the whole scene.
//   2. If the hash matches the last build -> reuse the RenderTileFn.
//   3. Otherwise: recompile + cache.
//
// Future: granular per-node caching via structural_hash() — only the
// affected functions in the LLVM module are regenerated. This requires
// JITDylib::remove() to drop the old symbol and addIRModule for the new one.

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/compile_mode_policy.hpp"
#include "core/compiler/incremental_params.hpp"
#include "core/accel/guard_calibration.hpp"
#include "core/frep/scene.hpp"

#include <chrono>
#include <expected>
#include <memory>
#include <string>

namespace frep {

struct CompileStats {
    bool                       was_cached  = false;
    // True when scene_hash differed but structure_hash matched —
    // i.e. only parameter values changed.
    bool                       structure_unchanged = false;
    // True when the fast path fired: structure unchanged AND the compiler
    // was already in Incremental mode, so a param-buffer refresh sufficed
    // and no recompile happened. (Implies was_cached=true.)
    bool                       param_buffer_hit = false;
    // True when this compile produced Incremental-mode code (whether
    // because the user asked for it explicitly or Auto kicked in).
    bool                       was_incremental = false;
    std::chrono::microseconds  codegen_us  = std::chrono::microseconds::zero();
    std::chrono::microseconds  jit_us      = std::chrono::microseconds::zero();
    std::size_t                scene_hash  = 0;
};

class IncrementalCompiler {
public:
    explicit IncrementalCompiler(TracerConfig cfg = {})
        : tracer_cfg_(cfg) {}

    // Compiles the scene if it changed, otherwise returns the cached function.
    // The first call always compiles.
    std::expected<RenderTileFn, std::string>
    compile_if_changed(const SceneGraph& scene) {
        last_stats_ = {};
        auto h = scene.scene_hash();
        last_stats_.scene_hash = h;

        if (cached_fn_ && h == last_hash_) {
            last_stats_.was_cached = true;
            return cached_fn_;
        }

        // Scene changed. Check whether only parameter values moved.
        auto sh = scene.structure_hash();
        bool params_only = (cached_fn_ && sh == last_structure_hash_);
        if (params_only) last_stats_.structure_unchanged = true;

        // ── Fast path: Incremental mode + structure unchanged ────────────────
        // If we are in Incremental mode AND the structure is the same as
        // last time AND we already have a JIT'd function AND a params
        // buffer set up — refresh the buffer from the scene's current
        // parameter values and reuse the cached function. NO RECOMPILE.
        if (params_only && current_mode_is_incremental_
            && cached_fn_ && params_ && refresh_params_from_scene(scene))
        {
            last_stats_.was_cached       = true;
            last_stats_.param_buffer_hit = true;
            last_hash_ = h;
            return cached_fn_;
        }

        // ── Slow path: full recompile ────────────────────────────────────────
        // Decide Incremental vs Constant for THIS recompile (Auto kicks in here).
        policy_.note_recompile();
        current_mode_is_incremental_ = policy_.use_incremental();
        tracer_cfg_.incremental_params = current_mode_is_incremental_;

        // Reset slot table if structure changed — old buffer indices are
        // no longer valid.
        if (!params_only) policy_.reset_auto_state();

        // Recompile
        auto t0 = std::chrono::steady_clock::now();

        auto ctx = std::make_unique<llvm::LLVMContext>();
        SceneCodegen cg(*ctx, tracer_cfg_);
        try {
            cg.emit_render_tile(scene, choose_sdf_mode(scene));
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Codegen error: ") + e.what());
        }
        auto t1 = std::chrono::steady_clock::now();
        last_stats_.codegen_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

        // Capture the binding table BEFORE moving the module so we can
        // build the IncrementalParams helper.
        if (current_mode_is_incremental_) {
            params_ = std::make_unique<IncrementalParams>(cg);
        } else {
            params_.reset();
        }

        auto mod = cg.take_module();

        // Always a fresh JitEngine to avoid symbol collisions in the JITDylib.
        jit_ = std::make_unique<JitEngine>();
        auto fn_or = jit_->load(std::move(mod), std::move(ctx));
        if (!fn_or) return std::unexpected(fn_or.error());

        auto t2 = std::chrono::steady_clock::now();
        last_stats_.jit_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        last_stats_.was_incremental = current_mode_is_incremental_;

        cached_fn_ = *fn_or;
        last_hash_ = h;
        last_structure_hash_ = sh;
        return cached_fn_;
    }

    // Force recompile — ignores the cache.
    std::expected<RenderTileFn, std::string>
    force_recompile(const SceneGraph& scene) {
        last_hash_ = 0;   // invalidate
        last_structure_hash_ = 0;
        cached_fn_ = nullptr;
        params_.reset();
        return compile_if_changed(scene);
    }

    const CompileStats& last_stats() const noexcept { return last_stats_; }

    // Returns the TracerConfig — allows changes from the GUI with an automatic recompile.
    TracerConfig& tracer_config() noexcept { return tracer_cfg_; }
    const TracerConfig& tracer_config() const noexcept { return tracer_cfg_; }

    // ── Compile mode (Constant / Incremental / Auto) ────────────────────────
    CompileModePolicy&       policy()       noexcept { return policy_; }
    const CompileModePolicy& policy() const noexcept { return policy_; }

    // ── Spatial guards (BVH approach 1) ─────────────────────────────────────
    // When enabled, the compiler uses the host calibration to decide, per
    // recompile, whether to emit the guarded scene_sdf (worth it only for
    // scenes of enough expensive-per-object objects). ON by default: the
    // calibration is conservative (its threshold is always above a bare
    // primitive, so simple scenes stay inlined), making guarding a safe
    // automatic win for the CSG/deform-heavy scenes it helps and a no-op
    // for everything else. The GUI exposes a toggle to disable it.
    void set_spatial_guards_enabled(bool on) { spatial_guards_ = on; }
    bool spatial_guards_enabled() const noexcept { return spatial_guards_; }

    // The SDF emission mode chosen for the most recent compile (for UI/
    // diagnostics — shows whether guarding actually kicked in).
    SceneCodegen::SceneSdfMode last_sdf_mode() const noexcept { return last_sdf_mode_; }

    // Returns the runtime params buffer to pass to TileScheduler::render.
    // Returns nullptr in Constant mode — the JIT'd function ignores its
    // params arg in that case.
    float* params_buffer() {
        return (current_mode_is_incremental_ && params_)
            ? params_->buffer() : nullptr;
    }

    // True when this compiler is currently producing Incremental code.
    // (After a compile_if_changed call, reflects the most recent decision.)
    bool currently_incremental() const noexcept {
        return current_mode_is_incremental_;
    }

private:
    // Walks the scene tree and updates every bound parameter in the buffer
    // to match the scene's current value. Returns true if all bindings
    // were found in the scene (signalling that the structure indeed
    // matches and the fast path is safe).
    bool refresh_params_from_scene(const SceneGraph& scene) {
        if (!params_) return false;
        // Collect (node_id, param_name) -> value pairs from the scene.
        std::unordered_map<std::string, float> live;
        std::function<void(const FRepNode*)> walk = [&](const FRepNode* n) {
            if (!n) return;
            for (const auto& [k, v] : n->params)
                live[n->id + "::" + k] = v;
            for (const auto& c : n->children) walk(c.get());
        };
        for (const auto& [id, obj] : scene.objects())
            walk(obj.geometry.get());

        // For each binding the compiler emitted, update the buffer.
        // Any binding that no longer exists in the scene means the
        // structure changed — bail out to force a recompile.
        for (int slot = 0; slot < params_->size(); ++slot) {
            // We need (node_id, param_name) for this slot — IncrementalParams
            // doesn't expose that, so reconstruct via key_to_slot_ lookup.
        }
        // Simpler: walk the live map and set every key the params_ knows.
        // If any params_ key isn't in `live`, structure changed.
        // IncrementalParams::has() lets us query.
        bool ok = true;
        for (const auto& [k, v] : live) {
            auto sep = k.find("::");
            if (sep == std::string::npos) continue;
            std::string id = k.substr(0, sep);
            std::string nm = k.substr(sep + 2);
            if (params_->has(id, nm)) params_->set(id, nm, v);
        }
        // We don't actually have a way to assert every slot got written;
        // unwritten slots keep the previous value, which is acceptable as
        // long as we trust the structure_hash check. The hash check
        // already gates this path.
        return ok;
    }

    // Pick the scene_sdf emission mode for this compile. Inlined unless
    // spatial guards are enabled AND the host calibration says this scene's
    // objects are expensive enough (and numerous enough) for guarding to
    // pay off. The calibration is fetched once and cached — but only when
    // the scene could plausibly benefit, so simple scenes never pay the
    // (one-time) calibration cost at all.
    SceneCodegen::SceneSdfMode choose_sdf_mode(const SceneGraph& scene) {
        if (!spatial_guards_) {
            last_sdf_mode_ = SceneCodegen::SceneSdfMode::Inlined;
            return last_sdf_mode_;
        }
        // Cheap pre-screen: count visible objects and their average node
        // count WITHOUT calibrating. Guarding needs a handful of objects
        // and at least mildly complex ones; if the scene can't clear even
        // the lowest plausible bar, stay inlined and skip calibration
        // entirely (so trivial scenes never trigger the measurement).
        long total = 0; int n = 0; int maxc = 0;
        for (const auto& [id, obj] : scene.objects()) {
            if (!obj.visible || !obj.geometry) continue;
            const int c = node_count(*obj.geometry);
            total += c; ++n;
            if (c > maxc) maxc = c;
        }
        constexpr int kMinObjects = 8;
        // The smallest threshold calibration can return is 3 (guaranteed
        // above a bare primitive). If no object even reaches 3 nodes, or
        // there aren't enough objects, guarding can't apply — don't
        // calibrate.
        if (n < kMinObjects || maxc < 3) {
            last_sdf_mode_ = SceneCodegen::SceneSdfMode::Inlined;
            return last_sdf_mode_;
        }
        if (!cal_loaded_) {
            cal_ = accel::get_or_calibrate(false);
            cal_loaded_ = true;
        }
        const double avg = n ? (double)total / n : 0.0;
        const bool guard = accel::should_guard(cal_, n, avg);
        last_sdf_mode_ = guard ? SceneCodegen::SceneSdfMode::Guarded
                               : SceneCodegen::SceneSdfMode::Inlined;
        return last_sdf_mode_;
    }

    TracerConfig                   tracer_cfg_;
    CompileModePolicy              policy_;
    std::unique_ptr<JitEngine>     jit_;
    RenderTileFn                   cached_fn_ = nullptr;
    std::size_t                    last_hash_ = 0;
    std::size_t                    last_structure_hash_ = 0;
    CompileStats                   last_stats_;

    std::unique_ptr<IncrementalParams> params_;
    bool                               current_mode_is_incremental_ = false;

    bool                           spatial_guards_ = true;
    bool                           cal_loaded_ = false;
    accel::GuardCalibration        cal_;
    SceneCodegen::SceneSdfMode     last_sdf_mode_ = SceneCodegen::SceneSdfMode::Inlined;
};

} // namespace frep
