// core/exec/gpu_ir_executor.hpp
//
// IExecutor for the GPU-IR path: codegen → LLVM IR → llvm-spirv → SPIR-V
// kernel → OpenCL. This shares the IR source with the CPU-IR path (same
// codegen / same render_tile), then retargets the IR to a SPIR-V compute
// kernel and runs it on the GPU through OpenCL — the "one IR, two targets"
// result (JIT for CPU, OpenCL for GPU), distinct from the GLSL→Vulkan path.
//
// Needs a real OpenCL GPU device: available() reflects OpenClCtx::available().
// The OpenCL context (SPIR-V build is expensive) is created once and cached,
// keyed by the emitted IR, so per-tile rendering is cheap after warm-up.
//
// Part 1: the kernel is render_tile run as a single work-item over the tile
// (its internal loop walks the pixels). Part 2 will switch codegen to a
// per-pixel kernel + an NDRange launch for real GPU parallelism.

#pragma once

#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/compile_policy.hpp"
#include "core/compiler/retarget_nvptx.hpp"
#include "core/gpu/cuda_ctx.hpp"

#include <llvm/Support/TargetSelect.h>

#include <chrono>
#include <cmath>
#include <cstdint>
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
    void set_compile_policy(const CompilePolicy* p) { policy_ = p; }

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
        // Codegen the same render_tile the CPU path JITs.
        auto ctx = std::make_unique<llvm::LLVMContext>();
        TracerConfig cfg = cfg_;
        // Opt-in incremental params (see set_compile_policy): runtime-placed
        // parameters become `params` buffer loads, so the emitted IR — and thus
        // the cached PTX — is invariant to those values. Without a policy this
        // is exactly the previous all-baked path.
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

        // Runtime parameter buffer in the kernel's slot order. Codegen runs
        // every frame (cheap), so these carry the current values; the PTX is
        // rebuilt only when the IR text changes (a placement / topology /
        // constant edit), whereas a runtime-value edit is just this re-upload.
        std::vector<float> params(static_cast<std::size_t>(cg.param_slot_count()), 0.0f);
        for (const auto& b : cg.param_bindings()) params[b.slot] = b.default_value;
        std::vector<std::uint8_t> full;   // full-frame RGBA8
        auto t_r = clk::now();
        auto rr = cuda_ctx_->render(args, params, full);
        r.render_ms = std::chrono::duration<double, std::milli>(clk::now() - t_r).count();
        if (!rr) { r.error = "cuda render: " + rr.error(); return r; }

        // Crop tile region, RGBA8 → float.
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
        r.threads_used = 0;
        r.ok = true;
        return r;
#endif  // FREP_HAS_CUDA
    }

private:
    SceneCodegen::SceneSdfMode mode_;
    TracerConfig               cfg_;
#ifdef FREP_HAS_CUDA
    std::unique_ptr<gpu::CudaCtx> cuda_ctx_;
#endif
    std::string cached_ir_;
    const CompilePolicy* policy_ = nullptr;
};

} // namespace frep::exec
