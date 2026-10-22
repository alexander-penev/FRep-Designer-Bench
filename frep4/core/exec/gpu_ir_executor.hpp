// core/exec/gpu_ir_executor.hpp
//
// IExecutor for the GPU-IR path: codegen → LLVM IR → NVPTX → PTX → CUDA
// Driver API. This shares the IR source with the CPU-IR path (same codegen,
// same render_tile), then retargets the IR to a per-pixel CUDA kernel and
// runs it on the GPU — the "one IR, two targets" result (JIT for CPU, PTX for
// GPU), distinct from the GLSL→Vulkan path.
//
// Needs an NVIDIA device: available() reflects CudaCtx::available(). The
// compiled kernel is cached twice over: codegen is skipped while the scene's
// content_hash is unchanged, and when it does run, the PTX is rebuilt only if
// the IR text differs from the one the current CUDA module came from.

#pragma once

#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/compile_policy.hpp"
#include "core/compiler/retarget_nvptx.hpp"
#include "core/gpu/cuda_ctx.hpp"

#include <llvm/Support/TargetSelect.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace frep::exec {

class GpuIrExecutor : public IExecutor {
public:
    explicit GpuIrExecutor(
        SceneCodegen::SceneSdfMode mode = SceneCodegen::SceneSdfMode::Inlined,
        TracerConfig cfg = {})
        : mode_(mode), cfg_(cfg) {}

    // Install a placement policy to enable incremental runtime parameters
    // (opt-in; null = every parameter baked, the previous behaviour). When set,
    // runtime-placed parameters are read from the kernel's `params` buffer, so
    // editing them re-uploads the buffer instead of rebuilding the PTX. Not
    // owned; must outlive the executor.
    void set_compile_policy(const CompilePolicy* p) { policy_ = p; have_key_ = false; }

    PathKind path() const override { return PathKind::GpuIr; }
#ifdef FREP_HAS_CUDA
    bool available() const override { return gpu::CudaCtx::available(); }
#else
    bool available() const override { return false; }  // CUDA not built in
#endif

    TileResult render(const SceneGraph& scene, int W, int H,
                      const Tile& tile) override {
        using clk = std::chrono::steady_clock;
        TileResult r;
        r.tile = tile;
        r.path = PathKind::GpuIr;
#ifndef FREP_HAS_CUDA
        r.error = "CUDA GPU-IR path not built (FREP_BUILD_CUDA off)";
        return r;
#else

        auto t_c = clk::now();
        // Codegen bakes the geometry, materials, visibility and lights into
        // the kernel, so while the scene's content_hash is unchanged the
        // kernel we already have is the kernel codegen would produce: skip it.
        // Before this check, every frame re-ran codegen and printed the whole
        // module to text just to compare it with the cached text — 1–2 ms per
        // frame, more than the kernel itself at 512². Runtime parameters (a
        // policy is installed) are the exception: the bindings codegen
        // produces carry the current values, so it runs to refresh them, and
        // the IR-text compare below still spares the PTX rebuild.
        const std::size_t key = scene.content_hash();
        const bool reuse = cuda_ctx_ && have_key_ && key == cached_key_ && !policy_;
        if (!reuse) {
            // Codegen the same render_tile the CPU path JITs.
            auto ctx = std::make_unique<llvm::LLVMContext>();
            TracerConfig cfg = cfg_;
            // Opt-in incremental params (see set_compile_policy): runtime-placed
            // parameters become `params` buffer loads, so the emitted IR — and
            // thus the cached PTX — is invariant to those values. Without a
            // policy this is exactly the previous all-baked path.
            cfg.incremental_params = (policy_ != nullptr);
            SceneCodegen cg(*ctx, cfg);
            cg.set_compile_policy(policy_);
            try {
                cg.emit_gpu_kernel(scene, mode_);   // per-pixel CUDA kernel
            } catch (const std::exception& e) {
                r.error = std::string("codegen: ") + e.what();
                return r;
            }
            auto mod = cg.take_module();

            // Ensure the LLVM NVPTX backend is registered — once per process.
            static std::once_flag init_flag;
            std::call_once(init_flag, [] {
                llvm::InitializeAllTargetInfos();
                llvm::InitializeAllTargets();
                llvm::InitializeAllTargetMCs();
                llvm::InitializeAllAsmPrinters();
            });

            // Retarget IR → PTX. Cache the CUDA context (PTX JIT is expensive)
            // keyed on the IR text so a changed scene rebuilds.
            std::string ir_text;
            { llvm::raw_string_ostream os(ir_text); mod->print(os, nullptr); }

            if (!cuda_ctx_ || ir_text != cached_ir_) {
                NVPTXRetarget retarget;
                auto ptx = retarget.retarget(*mod);
                if (!ptx) { r.error = "nvptx: " + ptx.error(); return r; }
                auto ctx_or = gpu::CudaCtx::create(*ptx, "render_tile");
                if (!ctx_or) { r.error = "cuda: " + ctx_or.error(); return r; }
                cuda_ctx_ = std::move(*ctx_or);
                cached_ir_ = ir_text;
            }

            // Runtime parameter buffer in the kernel's slot order. The bindings
            // carry the values of this codegen, so with a policy installed a
            // runtime-value edit is just this refresh and the re-upload below,
            // while the PTX is rebuilt only for a placement / topology /
            // constant edit (one that changes the IR text).
            cached_params_.assign(static_cast<std::size_t>(cg.param_slot_count()), 0.0f);
            for (const auto& b : cg.param_bindings()) cached_params_[b.slot] = b.default_value;
            cached_key_ = key;
            have_key_   = true;
        }
        r.compile_ms = std::chrono::duration<double, std::milli>(clk::now() - t_c).count();

        // ── Camera basis (mirrors TileScheduler) ────────────────────────────
        const Camera& cam = scene.camera();
        auto sub = [](const auto& a, const auto& b) {
            return std::array<float,3>{a[0]-b[0], a[1]-b[1], a[2]-b[2]};
        };
        auto norm = [](std::array<float,3> v) {
            float l = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            if (l < 1e-8f) return std::array<float,3>{0,0,-1};
            return std::array<float,3>{v[0]/l, v[1]/l, v[2]/l};
        };
        auto cross = [](const auto& a, const auto& b) {
            return std::array<float,3>{a[1]*b[2]-a[2]*b[1],
                                       a[2]*b[0]-a[0]*b[2],
                                       a[0]*b[1]-a[1]*b[0]};
        };
        auto fwd = norm(sub(cam.target, cam.position));
        auto right = norm(cross(fwd, cam.up));
        auto up = cross(right, fwd);
        float view_scale = (cam.projection == Camera::Projection::Orthographic)
            ? -0.5f * cam.ortho_size
            :  std::tan(cam.fov_deg * 3.14159265f / 360.0f);

        gpu::CudaRenderArgs args;
        args.tx = tile.x0; args.ty = tile.y0;
        args.tw = tile.width(); args.th = tile.height();
        args.iw = W; args.ih = H;
        for (int i = 0; i < 3; ++i) {
            args.cam_pos[i]   = cam.position[i];
            args.cam_fwd[i]   = fwd[i];
            args.cam_right[i] = right[i];
            args.cam_up[i]    = up[i];
        }
        args.fov_scale = view_scale;

        // The context returns the tile's floats directly: no full-frame copy,
        // no clamp to RGBA8 and back. That round trip quantised every GPU-IR
        // pixel to 1/255 and cost two passes over the frame per frame.
        auto t_r = clk::now();
        auto rr = cuda_ctx_->render(args, cached_params_, r.rgba);
        r.render_ms = std::chrono::duration<double, std::milli>(clk::now() - t_r).count();
        if (!rr) { r.error = "cuda render: " + rr.error(); return r; }
        pad_clipped_tile(r.rgba, tile, W, H);
        r.threads_used = 0;
        r.ok = true;

        if (profile_enabled()) {
            const auto& st = cuda_ctx_->stats();
            std::fprintf(stderr,
                "gpu_ir-prof: %-7s compile=%.3f run=%.3f (launch %.3f copy %.3f) total=%.3f ms\n",
                reuse ? "reuse" : "codegen", r.compile_ms, r.render_ms,
                st.launch_ms, st.copy_ms,
                std::chrono::duration<double, std::milli>(clk::now() - t_c).count());
        }
        return r;
#endif  // FREP_HAS_CUDA
    }

private:
    // FREP_EXEC_PROFILE=1 prints one line per frame with the stage split.
    static bool profile_enabled() {
        static const bool on = std::getenv("FREP_EXEC_PROFILE") != nullptr;
        return on;
    }

    // CudaCtx returns the tile clipped to the frame. The scheduler never asks
    // for a tile that reaches outside it, but the interface allows one; pad it
    // with zeros so the result has the tile's own size.
    static void pad_clipped_tile(std::vector<float>& rgba, const Tile& tile, int W, int H) {
        const int x0 = std::max(tile.x0, 0), y0 = std::max(tile.y0, 0);
        const int x1 = std::min(tile.x1, W), y1 = std::min(tile.y1, H);
        const int cw = std::max(x1 - x0, 0), ch = std::max(y1 - y0, 0);
        const int tw = tile.width(), th = tile.height();
        if (cw == tw && ch == th) return;
        std::vector<float> full((std::size_t)tw * th * 4, 0.0f);
        for (int y = 0; y < ch; ++y)
            std::memcpy(&full[(((std::size_t)(y0 - tile.y0 + y)) * tw + (x0 - tile.x0)) * 4],
                        &rgba[(std::size_t)y * cw * 4],
                        (std::size_t)cw * 4 * sizeof(float));
        rgba.swap(full);
    }

    SceneCodegen::SceneSdfMode mode_;
    TracerConfig               cfg_;
#ifdef FREP_HAS_CUDA
    std::unique_ptr<gpu::CudaCtx> cuda_ctx_;
#endif
    std::string        cached_ir_;        // IR text the current PTX was built from
    std::size_t        cached_key_ = 0;   // content_hash the kernel was compiled for
    bool               have_key_   = false;
    std::vector<float> cached_params_;    // kernel params in slot order, from that codegen
    const CompilePolicy* policy_ = nullptr;
};

} // namespace frep::exec
