// core/exec/gpu_executor.hpp
//
// IExecutor for the GPU-GLSL path: codegen → GLSL → glslang → SPIR-V →
// Vulkan compute. Renders a tile by setting the tile bounds in the push
// constants so the shader computes only that sub-region (dispatch is sized
// to the tile), then copies the tile pixels out of the full-frame readback
// and converts RGBA8 → float to match the CPU executor's format.
//
// Needs a real Vulkan device: available() reflects VulkanCtx::available().
// In environments without one (software-only CI) the executor reports
// unavailable and the multi-path runner skips it.
//
// The Vulkan context + compiled shader are built lazily on first render and
// cached (creation is expensive), keyed by the scene's emitted GLSL so a
// changed scene rebuilds. This makes per-tile rendering cheap after warm-up.

#pragma once

#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/compiler/scene_bindings.hpp"
#include "core/compiler/compile_sdf.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace frep::exec {

class GpuGlslExecutor : public IExecutor {
public:
    GpuGlslExecutor() = default;
    // Override the tracer config (e.g. disable shadows/AO to isolate where
    // this path diverges from CPU). Must match the other executor's cfg.
    explicit GpuGlslExecutor(TracerConfig cfg) : cfg_(cfg) {}

    // Install a placement policy to enable runtime parameters (opt-in; null =
    // every parameter baked, the previous behaviour). When set, runtime-placed
    // parameters are read from the shader's binding-3 buffer, so editing them
    // re-uploads the buffer instead of recompiling the SPIR-V. Not owned.
    void set_compile_policy(const CompilePolicy* p) { policy_ = p; }

    PathKind path() const override { return PathKind::GpuGlsl; }
    bool available() const override { return gpu::VulkanCtx::available(); }

    TileResult render(const SceneGraph& scene, int W, int H,
                      const Tile& tile) override {
        using clk = std::chrono::steady_clock;
        TileResult r;
        r.tile = tile;
        r.path = PathKind::GpuGlsl;

        // ── Build / reuse the Vulkan context (compile) ──────────────────────
        auto t_c = clk::now();
        // Shared parameter binding table (empty unless a policy is installed,
        // so the default path is bit-identical to before). When non-empty the
        // emitted shader reads runtime params from binding 3 and its source is
        // invariant to those values, making a value edit a cache hit below.
        frep::ParamBindingTable bt;
        if (policy_) bt = frep::build_bindings(scene, *policy_);
        const frep::ParamBindingTable* btp = bt.empty() ? nullptr : &bt;
        auto pbuf = bt.seed_buffer();   // current runtime values, slot order

        // Resolve an Auto cull choice with a coarse probe (needs the JIT SDF,
        // hence here rather than in the emitter). The choice depends on scene
        // topology, not on edited parameters, so it is cached and only recomputed
        // when the scene pointer changes — the per-frame path then skips both the
        // probe and, via the source cache below, the shader rebuild.
        TracerConfig cfg = cfg_;
        if (cfg_.cull_slabs > 0 && cfg_.cull_method == TracerConfig::CullMethod::Auto) {
            if (&scene != cull_resolved_scene_) {
                if (cfg_.cull_auto_timed_probe) {
                    // Future work: pick Lipschitz vs Interval by timing a few
                    // frames of each on this device and keeping the faster, then
                    // cache. Correct where cell-count probing is not (it ignores
                    // per-box cull cost), but costs real frames — enabled only via
                    // this opt-in. Not yet implemented; fall back to topology so
                    // the flag is inert rather than wrong.
                    cull_resolved_cfg_ = jit::resolve_cull_method(scene, cfg_);
                } else {
                    cull_resolved_cfg_ = jit::resolve_cull_method(scene, cfg_);
                }
                cull_resolved_scene_ = &scene;
            }
            cfg = cull_resolved_cfg_;
        }
        auto emit = gpu::GlslEmitter::emit(scene, cfg, btp);
        if (!emit) { r.error = "glsl emit: " + emit.error(); return r; }

        // Rebuild context when the shader source changes (new scene/placement).
        if (!ctx_ || emit->source != cached_glsl_) {
            auto spv = gpu::compile_glsl_to_spv_managed(emit->source);
            if (!spv) { r.error = "spirv: " + spv.error(); return r; }
            auto ctx_or = gpu::VulkanCtx::create(
                spv->path(), emit->mesh_voxels, emit->texture_pixels, pbuf);
            if (!ctx_or) { r.error = "vulkan ctx create failed"; return r; }
            ctx_ = std::move(*ctx_or);
            cached_glsl_ = emit->source;
        }
        // Push current runtime parameter values (cheap memcpy; no-op when there
        // are none). The parameter-edit fast path on GPU-GLSL: no re-emit, no
        // SPIR-V recompile, no pipeline rebuild.
        ctx_->update_params(pbuf);
        r.compile_ms = std::chrono::duration<double, std::milli>(clk::now() - t_c).count();

        // ── Render the tile region ──────────────────────────────────────────
        auto push = gpu::build_push_from_scene(scene, W, H);
        // Set tile bounds so the shader computes only this sub-region and
        // the dispatch is sized to it. (Whole frame if tile == full frame.)
        push.tile_x0 = tile.x0; push.tile_y0 = tile.y0;
        push.tile_x1 = tile.x1; push.tile_y1 = tile.y1;

        std::vector<std::uint8_t> full;   // full-frame RGBA8 readback
        auto t_r = clk::now();
        auto rr = ctx_->render(push, full);
        r.render_ms = std::chrono::duration<double, std::milli>(clk::now() - t_r).count();
        if (!rr) { r.error = "gpu render failed"; return r; }

        // Crop tile region out of the full-frame readback, RGBA8 → float.
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
                if (src + 3 < full.size())
                    for (int c = 0; c < 4; ++c)
                        r.rgba[dst + c] = full[src + c] / 255.0f;
            }
        }

        const auto& st = ctx_->stats();
        r.render_ms = st.render_ms > 0 ? st.render_ms : r.render_ms;
        r.threads_used = 0;   // GPU — not CPU threads
        r.ok = true;
        return r;
    }

private:
    std::unique_ptr<gpu::VulkanCtx> ctx_;
    std::string cached_glsl_;
    const SceneGraph*   cull_resolved_scene_ = nullptr;
    TracerConfig        cull_resolved_cfg_{};
    TracerConfig cfg_;
    const CompilePolicy* policy_ = nullptr;
};

} // namespace frep::exec
