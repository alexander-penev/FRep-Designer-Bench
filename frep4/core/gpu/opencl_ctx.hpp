// core/gpu/opencl_ctx.hpp
//
// OpenCL execution context for the GPU-IR retargeting path.
//
// The GPU-IR path is: codegen → LLVM IR → llvm-spirv → SPIR-V kernel →
// OpenCL. Unlike the GLSL path (Vulkan compute, push constants + storage
// image), the IR path's `render_tile` is a SPIR_KERNEL with the same C
// signature the CPU JIT uses: (global float* out, int tx,ty,tw,th,iw,ih,
// float cam[12], float fov, global float* params). OpenCL runs exactly
// that kind of kernel, so the IR can be executed on the GPU with no change
// to its signature — the authentic "one IR, two retarget targets" result
// (JIT for CPU, OpenCL for GPU).
//
// This is the runtime mirror of VulkanCtx. Part 1 launches render_tile as
// a single work-item covering the whole tile (correctness first); Part 2
// adds a per-pixel kernel for real GPU parallelism.

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace frep::gpu {

// Camera + tile parameters for one render_tile invocation, mirroring the
// scalar arguments the JIT'd kernel takes.
struct ClRenderArgs {
    int   tx = 0, ty = 0, tw = 0, th = 0;   // tile origin + extent
    int   iw = 0, ih = 0;                   // full image dims
    float cam_pos[3]   = {0, 0, 0};
    float cam_fwd[3]   = {0, 0, -1};
    float cam_right[3] = {1, 0, 0};
    float cam_up[3]    = {0, 1, 0};
    float fov_scale    = 1.0f;              // sign encodes projection mode
};

struct ClRenderStats {
    double init_ms       = 0;   // platform + device + context + queue
    double build_ms      = 0;   // clCreateProgramWithIL + clBuildProgram
    double render_ms     = 0;   // enqueue + finish + readback
    int    width = 0, height = 0;
    std::string device_name;
};

// Owns an OpenCL context built from a SPIR-V kernel module. Construct via
// create(), call render() to get an RGBA8 frame back.
class OpenClCtx {
public:
    // Build a context from an in-memory SPIR-V module (the bytes the
    // SPIRVRetarget produced). `params` is the optional incremental-mode
    // parameter buffer the kernel reads (may be empty).
    static std::expected<std::unique_ptr<OpenClCtx>, std::string>
    create(const std::vector<unsigned char>& spirv,
           const std::string& kernel_name = "render_tile");

    ~OpenClCtx();
    OpenClCtx(const OpenClCtx&)            = delete;
    OpenClCtx& operator=(const OpenClCtx&) = delete;

    // Render the given args into out_rgba (iw*ih*4 RGBA8). `params` is the
    // kernel's parameter buffer (may be empty for non-incremental scenes).
    std::expected<void, std::string>
    render(const ClRenderArgs& args,
           const std::vector<float>& params,
           std::vector<std::uint8_t>& out_rgba);

    const ClRenderStats& stats() const { return stats_; }

    // True if an OpenCL platform with at least one device exists.
    static bool available();

private:
    OpenClCtx() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ClRenderStats stats_;
};

} // namespace frep::gpu
