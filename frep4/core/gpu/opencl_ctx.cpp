// core/gpu/opencl_ctx.cpp

#include "core/gpu/opencl_ctx.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <chrono>
#include <cstring>

namespace frep::gpu {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct OpenClCtx::Impl {
    cl_platform_id   platform = nullptr;
    cl_device_id     device   = nullptr;
    cl_context       context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_program       program  = nullptr;
    cl_kernel        kernel   = nullptr;

    // Persistent device buffers, reallocated when sizes change.
    cl_mem out_buf    = nullptr;   // iw*ih*4 floats (kernel writes float RGBA)
    cl_mem params_buf = nullptr;   // optional params (>=1 float to keep valid)
    std::size_t out_floats = 0;
    std::size_t params_floats = 0;

    ~Impl() {
        if (out_buf)    clReleaseMemObject(out_buf);
        if (params_buf) clReleaseMemObject(params_buf);
        if (kernel)     clReleaseKernel(kernel);
        if (program)    clReleaseProgram(program);
        if (queue)      clReleaseCommandQueue(queue);
        if (context)    clReleaseContext(context);
        // device/platform are not retained, no release.
    }
};

bool OpenClCtx::available() {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0)
        return false;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    for (auto p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) == CL_SUCCESS
            && ndev > 0)
            return true;
    }
    return false;
}

std::expected<std::unique_ptr<OpenClCtx>, std::string>
OpenClCtx::create(const std::vector<unsigned char>& spirv,
                  const std::string& kernel_name) {
    auto t0 = clk::now();
    auto self = std::unique_ptr<OpenClCtx>(new OpenClCtx());
    self->impl_ = std::make_unique<Impl>();
    Impl& I = *self->impl_;

    // ── Platform + GPU device ───────────────────────────────────────────────
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0)
        return std::unexpected("no OpenCL platform");
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    cl_int err = CL_SUCCESS;
    bool found = false;
    for (auto p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) == CL_SUCCESS
            && ndev > 0) {
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &I.device, nullptr);
            I.platform = p; found = true; break;
        }
    }
    if (!found) return std::unexpected("no OpenCL GPU device");

    char name[256] = {0};
    clGetDeviceInfo(I.device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    self->stats_.device_name = name;

    I.context = clCreateContext(nullptr, 1, &I.device, nullptr, nullptr, &err);
    if (!I.context) return std::unexpected("clCreateContext failed");

    // OpenCL 2.0+ queue creation (the 1.2 clCreateCommandQueue is deprecated
    // but still works; use the properties form for forward compatibility).
    I.queue = clCreateCommandQueueWithProperties(I.context, I.device, nullptr, &err);
    if (!I.queue) return std::unexpected("clCreateCommandQueue failed");

    self->stats_.init_ms = ms_since(t0);

    // ── Program from SPIR-V IL ──────────────────────────────────────────────
    auto t1 = clk::now();
    if (spirv.empty()) return std::unexpected("empty SPIR-V");
    I.program = clCreateProgramWithIL(I.context, spirv.data(), spirv.size(), &err);
    if (!I.program || err != CL_SUCCESS)
        return std::unexpected("clCreateProgramWithIL failed (err "
                               + std::to_string(err) + ")");

    err = clBuildProgram(I.program, 1, &I.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string log(4096, '\0');
        std::size_t n = 0;
        clGetProgramBuildInfo(I.program, I.device, CL_PROGRAM_BUILD_LOG,
                              log.size(), log.data(), &n);
        log.resize(n);
        return std::unexpected("clBuildProgram failed: " + log);
    }

    I.kernel = clCreateKernel(I.program, kernel_name.c_str(), &err);
    if (!I.kernel || err != CL_SUCCESS)
        return std::unexpected("clCreateKernel('" + kernel_name + "') failed (err "
                               + std::to_string(err) + ")");

    self->stats_.build_ms = ms_since(t1);
    return self;
}

std::expected<void, std::string>
OpenClCtx::render(const ClRenderArgs& args,
                  const std::vector<float>& params,
                  std::vector<std::uint8_t>& out_rgba) {
    Impl& I = *impl_;
    auto t0 = clk::now();
    const int W = args.iw, H = args.ih;
    if (W <= 0 || H <= 0) return std::unexpected("bad image dims");

    cl_int err = CL_SUCCESS;
    const std::size_t out_floats = (std::size_t)W * H * 4;

    // (Re)allocate device buffers when sizes change.
    if (I.out_buf == nullptr || I.out_floats != out_floats) {
        if (I.out_buf) clReleaseMemObject(I.out_buf);
        I.out_buf = clCreateBuffer(I.context, CL_MEM_WRITE_ONLY,
                                   out_floats * sizeof(float), nullptr, &err);
        if (!I.out_buf) return std::unexpected("alloc out_buf failed");
        I.out_floats = out_floats;
    }
    // params buffer must be a valid (non-null) global pointer even when the
    // scene has no incremental params — allocate at least one float.
    const std::size_t pf = params.empty() ? 1 : params.size();
    if (I.params_buf == nullptr || I.params_floats != pf) {
        if (I.params_buf) clReleaseMemObject(I.params_buf);
        I.params_buf = clCreateBuffer(I.context, CL_MEM_READ_ONLY,
                                      pf * sizeof(float), nullptr, &err);
        if (!I.params_buf) return std::unexpected("alloc params_buf failed");
        I.params_floats = pf;
    }
    if (!params.empty()) {
        err = clEnqueueWriteBuffer(I.queue, I.params_buf, CL_TRUE, 0,
                                   params.size() * sizeof(float),
                                   params.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return std::unexpected("upload params failed");
    }

    // ── Set kernel arguments (matches render_tile's C signature) ────────────
    // (global float* out, int tx,ty,tw,th,iw,ih, float cam[12]+fov, global float* params)
    cl_uint a = 0;
    auto setI = [&](int v) { return clSetKernelArg(I.kernel, a++, sizeof(int), &v); };
    auto setF = [&](float v){ return clSetKernelArg(I.kernel, a++, sizeof(float), &v); };
    err  = clSetKernelArg(I.kernel, a++, sizeof(cl_mem), &I.out_buf);
    err |= setI(args.tx); err |= setI(args.ty);
    err |= setI(args.tw); err |= setI(args.th);
    err |= setI(args.iw); err |= setI(args.ih);
    err |= setF(args.cam_pos[0]);   err |= setF(args.cam_pos[1]);   err |= setF(args.cam_pos[2]);
    err |= setF(args.cam_fwd[0]);   err |= setF(args.cam_fwd[1]);   err |= setF(args.cam_fwd[2]);
    err |= setF(args.cam_right[0]); err |= setF(args.cam_right[1]); err |= setF(args.cam_right[2]);
    err |= setF(args.cam_up[0]);    err |= setF(args.cam_up[1]);    err |= setF(args.cam_up[2]);
    err |= setF(args.fov_scale);
    err |= clSetKernelArg(I.kernel, a++, sizeof(cl_mem), &I.params_buf);
    if (err != CL_SUCCESS) return std::unexpected("clSetKernelArg failed");

    // Part 1: single work-item covering the whole tile (correctness first;
    // the kernel's internal loop walks the tile pixels). Part 2 switches to
    // a per-pixel NDRange for real parallelism.
    std::size_t global = 1;
    err = clEnqueueNDRangeKernel(I.queue, I.kernel, 1, nullptr,
                                 &global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        return std::unexpected("clEnqueueNDRangeKernel failed (err "
                               + std::to_string(err) + ")");

    // Read back the float RGBA buffer and convert to 8-bit.
    std::vector<float> fbuf(out_floats);
    err = clEnqueueReadBuffer(I.queue, I.out_buf, CL_TRUE, 0,
                              out_floats * sizeof(float), fbuf.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS) return std::unexpected("readback failed");
    clFinish(I.queue);

    out_rgba.resize(out_floats);
    for (std::size_t i = 0; i < out_floats; ++i) {
        float v = fbuf[i];
        v = v < 0 ? 0 : (v > 1 ? 1 : v);
        out_rgba[i] = (std::uint8_t)(v * 255.0f + 0.5f);
    }

    stats_.width = W; stats_.height = H;
    stats_.render_ms = ms_since(t0);
    return {};
}

OpenClCtx::~OpenClCtx() = default;

} // namespace frep::gpu
