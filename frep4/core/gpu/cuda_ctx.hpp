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
// Runtime mirror of VulkanCtx / OpenClCtx. Part 1 launches render_tile as a
// single thread over the whole tile (correctness first); Part 2 switches to
// a per-pixel grid for real GPU parallelism.
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
    double render_ms = 0;   // launch + sync + readback
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

    // Render args into out_rgba (iw*ih*4 RGBA8). `params` is the kernel's
    // optional parameter buffer (may be empty).
    std::expected<void, std::string>
    render(const CudaRenderArgs& args,
           const std::vector<float>& params,
           std::vector<std::uint8_t>& out_rgba);

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
