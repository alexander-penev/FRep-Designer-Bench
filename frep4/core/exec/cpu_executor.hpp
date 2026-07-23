// core/exec/cpu_executor.hpp
//
// IExecutor for the CPU-IR path: codegen → LLVM IR → JIT → native, rendered
// with the tiled CPU scheduler. Renders the full frame (the scheduler is
// already internally tiled and multi-threaded) and crops the requested
// Tile out of it. Cropping rather than region-rendering keeps this simple
// and correct; a future optimisation could teach the scheduler to render a
// sub-rectangle so a frame-split actually halves CPU work.

#pragma once

#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>

namespace frep::exec {

class CpuIrExecutor : public IExecutor {
public:
    // `mode` selects Inlined/Guarded SDF emission (the spatial-guard work);
    // defaults to Inlined for a stable baseline in comparisons. `cfg` lets a
    // caller override the tracer config (e.g. disable shadows/AO to isolate
    // where two paths diverge) — both executors must use the SAME cfg for a
    // fair compare.
    explicit CpuIrExecutor(
        SceneCodegen::SceneSdfMode mode = SceneCodegen::SceneSdfMode::Inlined,
        TracerConfig cfg = {})
        : mode_(mode), cfg_(cfg) {}

    PathKind path() const override { return PathKind::CpuIr; }
    bool available() const override { return true; }  // CPU always available

    TileResult render(const SceneGraph& scene, int W, int H,
                      const Tile& tile) override {
        using clk = std::chrono::steady_clock;
        TileResult r;
        r.tile = tile;
        r.path = PathKind::CpuIr;

        // Compile (codegen + JIT), cached on the scene's structure hash so
        // repeated renders of the same topology — e.g. a calibration probe
        // followed by the real band, or successive frames — reuse the JIT'd
        // function instead of recompiling. Compile dominates wall-time, so
        // this is what makes the weighted split's calibration pass nearly
        // free. A topology change invalidates the cache.
        auto t_c = clk::now();
        const std::size_t shash = scene.structure_hash();
        if (!cached_fn_ || shash != cached_structure_hash_) {
            auto ctx = std::make_unique<llvm::LLVMContext>();
            TracerConfig cfg = cfg_;
            SceneCodegen cg(*ctx, cfg);
            try {
                if (std::getenv("FREP4_VEC_RENDER"))
                    cg.emit_render_tile_vec(scene, mode_, 8);   // Approach B (CPU packet)
                else
                    cg.emit_render_tile(scene, mode_);
            } catch (const std::exception& e) {
                r.error = std::string("codegen: ") + e.what();
                return r;
            }
            auto mod = cg.take_module();
            jit_ = std::make_unique<JitEngine>();
            auto fn_or = jit_->load(std::move(mod), std::move(ctx));
            if (!fn_or) { r.error = "jit: " + fn_or.error(); return r; }
            cached_fn_ = *fn_or;
            cached_structure_hash_ = shash;
            r.compile_ms = std::chrono::duration<double, std::milli>(clk::now() - t_c).count();
        } else {
            r.compile_ms = 0.0;  // reused cached compile
        }
        auto* fn = cached_fn_;

        // Render only the tile's region (the JIT'd render_tile is natively
        // tile-addressed; RenderParams::region restricts which tiles run, so
        // a sub-frame costs proportionally less — no whole-frame waste).
        // We still allocate a full-frame buffer so tile coordinates map
        // directly, then copy out the region; only the region is computed.
        std::vector<float> full((std::size_t)W * H * 4, 0.0f);
        RenderParams rp; rp.width = W; rp.height = H;
        rp.region_x0 = tile.x0; rp.region_y0 = tile.y0;
        rp.region_x1 = tile.x1; rp.region_y1 = tile.y1;
        if (const char* s = std::getenv("FREP4_SSAA")) rp.ssaa = std::max(1, atoi(s));
        r.threads_used = (int)std::thread::hardware_concurrency();
        auto t_r = clk::now();
        TileScheduler::render(*fn, full.data(), scene.camera(), rp);
        r.render_ms = std::chrono::duration<double, std::milli>(clk::now() - t_r).count();

        // Copy the tile region into the result buffer.
        int tw = tile.width(), th = tile.height();
        r.rgba.assign((std::size_t)tw * th * 4, 0.0f);
        for (int y = 0; y < th; ++y) {
            int sy = tile.y0 + y;
            if (sy < 0 || sy >= H) continue;
            for (int x = 0; x < tw; ++x) {
                int sx = tile.x0 + x;
                if (sx < 0 || sx >= W) continue;
                std::size_t src = ((std::size_t)sy * W + sx) * 4;
                std::size_t dst = ((std::size_t)y * tw + x) * 4;
                for (int c = 0; c < 4; ++c) r.rgba[dst + c] = full[src + c];
            }
        }

        r.peak_rss_kb = peak_rss_kb();
        r.ok = true;
        return r;
    }

private:
    SceneCodegen::SceneSdfMode mode_;
    TracerConfig cfg_;
    // Compile cache: the JIT engine (owns the executable memory) and the
    // function pointer into it, keyed on the scene structure hash. Reused
    // across renders of the same topology so calibration + real render — and
    // successive frames — don't recompile.
    std::unique_ptr<JitEngine> jit_;
    RenderTileFn               cached_fn_ = nullptr;
    std::size_t                cached_structure_hash_ = 0;

    static std::size_t peak_rss_kb() {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("VmHWM:", 0) == 0)
                return std::strtoul(line.c_str() + 6, nullptr, 10);
        return 0;
    }
};

} // namespace frep::exec
