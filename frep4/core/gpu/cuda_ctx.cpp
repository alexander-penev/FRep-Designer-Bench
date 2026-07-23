// core/gpu/cuda_ctx.cpp

#include "core/gpu/cuda_ctx.hpp"

#include <chrono>
#include <cstring>

// ── Minimal CUDA Driver API declarations ────────────────────────────────────
// We declare only what we use (stable ABI) instead of depending on cuda.h,
// so this builds without the CUDA toolkit. Links against libcuda.so (shipped
// with the NVIDIA driver).
extern "C" {
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st*    CUcontext;
typedef struct CUmod_st*    CUmodule;
typedef struct CUfunc_st*   CUfunction;
typedef unsigned long long  CUdeviceptr;

#define CUDA_SUCCESS 0
// cuCtxCreate flag / cuDeviceGetName etc. constants are passed literally.

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t n);
CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t n);
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int sharedMem, void* hStream,
    void** kernelParams, void** extra);
CUresult cuCtxSynchronize(void);
CUresult cuGetErrorString(CUresult error, const char** pStr);
}

namespace frep::gpu {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static std::string cu_err(CUresult e) {
    const char* s = nullptr;
    cuGetErrorString(e, &s);
    return s ? s : ("CUresult " + std::to_string(e));
}

struct CudaCtx::Impl {
    CUcontext  context = nullptr;
    CUmodule   module  = nullptr;
    CUfunction kernel  = nullptr;
    CUdeviceptr out_buf = 0;     // iw*ih*4 floats
    CUdeviceptr params_buf = 0;
    std::size_t out_floats = 0;
    std::size_t params_floats = 0;

    ~Impl() {
        if (out_buf)    cuMemFree(out_buf);
        if (params_buf) cuMemFree(params_buf);
        if (module)     cuModuleUnload(module);
        if (context)    cuCtxDestroy(context);
    }
};

bool CudaCtx::available() {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS) return false;
    return n > 0;
}

std::expected<std::unique_ptr<CudaCtx>, std::string>
CudaCtx::create(const std::string& ptx, const std::string& kernel_name) {
    auto t0 = clk::now();
    auto self = std::unique_ptr<CudaCtx>(new CudaCtx());
    self->impl_ = std::make_unique<Impl>();
    Impl& I = *self->impl_;

    CUresult e = cuInit(0);
    if (e != CUDA_SUCCESS) return std::unexpected("cuInit: " + cu_err(e));

    int ndev = 0;
    if (cuDeviceGetCount(&ndev) != CUDA_SUCCESS || ndev == 0)
        return std::unexpected("no CUDA device");

    CUdevice dev;
    e = cuDeviceGet(&dev, 0);
    if (e != CUDA_SUCCESS) return std::unexpected("cuDeviceGet: " + cu_err(e));

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    self->stats_.device_name = name;

    e = cuCtxCreate(&I.context, 0, dev);
    if (e != CUDA_SUCCESS) return std::unexpected("cuCtxCreate: " + cu_err(e));
    self->stats_.init_ms = ms_since(t0);

    // ── PTX → module (driver JITs PTX to SASS here) ─────────────────────────
    auto t1 = clk::now();
    if (ptx.empty()) return std::unexpected("empty PTX");
    e = cuModuleLoadData(&I.module, ptx.c_str());
    if (e != CUDA_SUCCESS) return std::unexpected("cuModuleLoadData: " + cu_err(e));

    e = cuModuleGetFunction(&I.kernel, I.module, kernel_name.c_str());
    if (e != CUDA_SUCCESS)
        return std::unexpected("cuModuleGetFunction('" + kernel_name + "'): " + cu_err(e));
    self->stats_.build_ms = ms_since(t1);

    return self;
}

std::expected<void, std::string>
CudaCtx::render(const CudaRenderArgs& args,
                const std::vector<float>& params,
                std::vector<std::uint8_t>& out_rgba) {
    Impl& I = *impl_;
    auto t0 = clk::now();
    // The CUDA context is created on (and bound to) the thread that called
    // create(). The multi-view render worker spawns a fresh std::thread per
    // frame, so a later render runs on a different thread where this context is
    // not current — making it current here is what keeps gpu_ir from freezing on
    // the first frame's thread. cuCtxSetCurrent is cheap and idempotent.
    {
        CUresult ec = cuCtxSetCurrent(I.context);
        if (ec != CUDA_SUCCESS) return std::unexpected("cuCtxSetCurrent: " + cu_err(ec));
    }
    const int W = args.iw, H = args.ih;
    if (W <= 0 || H <= 0) return std::unexpected("bad image dims");

    const std::size_t out_floats = (std::size_t)W * H * 4;
    CUresult e;

    // (Re)allocate device buffers when sizes change.
    if (!I.out_buf || I.out_floats != out_floats) {
        if (I.out_buf) cuMemFree(I.out_buf);
        e = cuMemAlloc(&I.out_buf, out_floats * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAlloc out: " + cu_err(e));
        I.out_floats = out_floats;
    }
    const std::size_t pf = params.empty() ? 1 : params.size();
    if (!I.params_buf || I.params_floats != pf) {
        if (I.params_buf) cuMemFree(I.params_buf);
        e = cuMemAlloc(&I.params_buf, pf * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAlloc params: " + cu_err(e));
        I.params_floats = pf;
    }
    if (!params.empty()) {
        e = cuMemcpyHtoD(I.params_buf, params.data(), params.size() * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("upload params: " + cu_err(e));
    }

    // ── Kernel arguments (match render_tile's C signature) ──────────────────
    // (float* out, int tx,ty,tw,th,iw,ih, float cam[12]+fov, float* params)
    // cuLaunchKernel takes void** of pointers to each argument value.
    int   i_tx = args.tx, i_ty = args.ty, i_tw = args.tw, i_th = args.th;
    int   i_iw = args.iw, i_ih = args.ih;
    float fov  = args.fov_scale;
    void* kargs[] = {
        &I.out_buf,
        &i_tx, &i_ty, &i_tw, &i_th, &i_iw, &i_ih,
        (void*)&args.cam_pos[0],   (void*)&args.cam_pos[1],   (void*)&args.cam_pos[2],
        (void*)&args.cam_fwd[0],   (void*)&args.cam_fwd[1],   (void*)&args.cam_fwd[2],
        (void*)&args.cam_right[0], (void*)&args.cam_right[1], (void*)&args.cam_right[2],
        (void*)&args.cam_up[0],    (void*)&args.cam_up[1],    (void*)&args.cam_up[2],
        &fov,
        &I.params_buf,
    };

    // Per-pixel kernel: launch a 2D grid covering the tile. Each thread
    // renders one pixel (reads its pixel from the block/thread id). 16×16
    // threads per block is a good default for this kind of kernel.
    const unsigned bx = 16, by = 16;
    unsigned gx = (args.tw + bx - 1) / bx;
    unsigned gy = (args.th + by - 1) / by;
    e = cuLaunchKernel(I.kernel, gx ? gx : 1, gy ? gy : 1, 1, bx, by, 1,
                       0, nullptr, kargs, nullptr);
    if (e != CUDA_SUCCESS) return std::unexpected("cuLaunchKernel: " + cu_err(e));

    e = cuCtxSynchronize();
    if (e != CUDA_SUCCESS) return std::unexpected("cuCtxSynchronize: " + cu_err(e));

    std::vector<float> fbuf(out_floats);
    e = cuMemcpyDtoH(fbuf.data(), I.out_buf, out_floats * sizeof(float));
    if (e != CUDA_SUCCESS) return std::unexpected("readback: " + cu_err(e));

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

CudaCtx::~CudaCtx() = default;

} // namespace frep::gpu
