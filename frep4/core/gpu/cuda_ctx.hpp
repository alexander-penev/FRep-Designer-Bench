// core/gpu/cuda_ctx.hpp
//
// CUDA Driver API execution context for the GPU-IR path on NVIDIA.
//
// The GPU-IR path is: codegen → LLVM IR → NVPTX backend → PTX → CUDA. The
// same render_tile the CPU JITs is retargeted to PTX (NVPTXRetarget) and
// run on the GPU through the CUDA Driver API: cuModuleLoadData JIT-compiles
// the PTX to SASS for the device, cuLaunchKernel runs it. This is the
// native NVIDIA route for executing LLVM IR on the GPU (NVIDIA's OpenCL
// can't ingest SPIR-V, so the OpenCL/SPIR-V path doesn't run here).
//
// Runtime mirror of VulkanCtx. The kernel is launched as a 2D grid with one
// thread per pixel of the requested tile; the output lives in a full-frame
// float RGBA buffer on the device and only the tile rectangle is read back,
// into a pinned host buffer the context keeps across frames.
//
// We declare the small slice of the CUDA Driver API we use rather than
// depend on the full cuda.h, so this compiles without the CUDA toolkit
// headers (it links against libcuda.so, present with the NVIDIA driver).

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace frep::gpu {

struct CudaRenderArgs {
    int   tx = 0, ty = 0, tw = 0, th = 0;
    int   iw = 0, ih = 0;
    float cam_pos[3]   = {0, 0, 0};
    float cam_fwd[3]   = {0, 0, -1};
    float cam_right[3] = {1, 0, 0};
    float cam_up[3]    = {0, 1, 0};
    float fov_scale    = 1.0f;
};

struct CudaRenderStats {
    double init_ms   = 0;   // cuInit + device + context
    double build_ms  = 0;   // cuModuleLoadData (PTX → SASS JIT)
    double render_ms = 0;   // the whole render(): upload + launch + wait + readback
    double launch_ms = 0;   // host side of cuLaunchKernel (enqueue only)
    double copy_ms   = 0;   // readback: waits for the kernel, then the DtoH copy
    int    width = 0, height = 0;
    std::string device_name;
};

class CudaCtx {
public:
    // Build a context from PTX text (as produced by NVPTXRetarget).
    static std::expected<std::unique_ptr<CudaCtx>, std::string>
    create(const std::string& ptx, const std::string& kernel_name = "render_tile");

    ~CudaCtx();
    CudaCtx(const CudaCtx&)            = delete;
    CudaCtx& operator=(const CudaCtx&) = delete;

    // Render the tile in `args` and return its pixels in out_rgba as float
    // RGBA, tightly packed row by row, clipped to the frame (so a tile that
    // reaches outside the iw×ih frame comes back smaller than requested).
    // The values are what the kernel wrote — no clamp, no quantisation.
    // `params` is the kernel's optional parameter buffer (may be empty).
    std::expected<void, std::string>
    render(const CudaRenderArgs& args,
           const std::vector<float>& params,
           std::vector<float>& out_rgba);

    const CudaRenderStats& stats() const { return stats_; }

    // True if the CUDA driver initialises and at least one device exists.
    static bool available();

private:
    CudaCtx() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    CudaRenderStats stats_;
};

} // namespace frep::gpu
