// core/gpu/cuda_ctx.cpp

#include "core/gpu/cuda_ctx.hpp"
#include "core/gpu/gpu_numa.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>

// ── Minimal CUDA Driver API declarations ────────────────────────────────────
// We declare only what we use (stable ABI) instead of depending on cuda.h,
// so this builds without the CUDA toolkit. Links against libcuda.so (shipped
// with the NVIDIA driver).
//
// The names are the versioned exports cuda.h resolves its macros to
// (cuMemAlloc → cuMemAlloc_v2 and so on). The unsuffixed exports are the
// pre-CUDA-3.2 entry points, which take 32-bit device pointers and sizes;
// calling those with 64-bit arguments only worked because the driver keeps
// legacy allocations below 4 GiB.
extern "C" {
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st*    CUcontext;
typedef struct CUmod_st*    CUmodule;
typedef struct CUfunc_st*   CUfunction;
typedef struct CUarray_st*  CUarray;
typedef unsigned long long  CUdeviceptr;
typedef enum { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 } CUmemorytype;

// CUDA_MEMCPY2D_v2 from cuda.h; field order and widths are the ABI.
typedef struct {
    size_t       srcXInBytes, srcY;
    CUmemorytype srcMemoryType;
    const void*  srcHost;
    CUdeviceptr  srcDevice;
    CUarray      srcArray;
    size_t       srcPitch;
    size_t       dstXInBytes, dstY;
    CUmemorytype dstMemoryType;
    void*        dstHost;
    CUdeviceptr  dstDevice;
    CUarray      dstArray;
    size_t       dstPitch;
    size_t       WidthInBytes, Height;
} CUDA_MEMCPY2D;

#define CUDA_SUCCESS 0
// cuCtxCreate flag / cuDeviceGetName etc. constants are passed literally.

CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemAllocHost_v2(void** pp, size_t bytesize);
CUresult cuMemFreeHost(void* p);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t n);
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t n);
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy);
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int sharedMem, void* hStream,
    void** kernelParams, void** extra);
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

// The pack kernel: one thread per pixel, float4 in → RGBA8 (one u32) out.
// clamp to [0,1], scale by 255, round to nearest; r in the low byte, so the
// bytes in memory are r,g,b,a like the compute path's R8G8B8A8 image. A
// hand-written PTX string rather than a second LLVM lowering: it is twenty
// instructions and never changes.
static const char* kPackPtx = R"PTX(
.version 7.0
.target sm_50
.address_size 64

.visible .entry frep_pack_rgba8(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n)
{
    .reg .pred %p<2>;
    .reg .b32  %r<16>;
    .reg .b64  %rd<10>;
    .reg .f32  %f<8>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mul.wide.u32 %rd5, %r5, 16;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6];

    max.f32 %f1, %f1, 0f00000000;
    min.f32 %f1, %f1, 0f3F800000;
    mul.f32 %f1, %f1, 0f437F0000;
    cvt.rni.u32.f32 %r6, %f1;
    max.f32 %f2, %f2, 0f00000000;
    min.f32 %f2, %f2, 0f3F800000;
    mul.f32 %f2, %f2, 0f437F0000;
    cvt.rni.u32.f32 %r7, %f2;
    max.f32 %f3, %f3, 0f00000000;
    min.f32 %f3, %f3, 0f3F800000;
    mul.f32 %f3, %f3, 0f437F0000;
    cvt.rni.u32.f32 %r8, %f3;
    max.f32 %f4, %f4, 0f00000000;
    min.f32 %f4, %f4, 0f3F800000;
    mul.f32 %f4, %f4, 0f437F0000;
    cvt.rni.u32.f32 %r9, %f4;

    shl.b32 %r7, %r7, 8;
    shl.b32 %r8, %r8, 16;
    shl.b32 %r9, %r9, 24;
    or.b32 %r10, %r6, %r7;
    or.b32 %r10, %r10, %r8;
    or.b32 %r10, %r10, %r9;

    mul.wide.u32 %rd7, %r5, 4;
    add.s64 %rd8, %rd4, %rd7;
    st.global.u32 [%rd8], %r10;

DONE:
    ret;
}
)PTX";

struct CudaCtx::Impl {
    CUcontext  context = nullptr;
    CUmodule   module  = nullptr;
    CUfunction kernel  = nullptr;
    CUmodule   pack_module = nullptr;    // kPackPtx, loaded with the scene module
    CUfunction pack_kernel = nullptr;
    CUdeviceptr out_buf = 0;     // iw*ih*4 floats
    CUdeviceptr out8_buf = 0;    // iw*ih u32 (RGBA8), written by the pack kernel
    CUdeviceptr params_buf = 0;
    std::size_t out_floats = 0;
    std::size_t out8_px = 0;
    std::size_t params_floats = 0;
    // Pinned readback buffers, grown to the largest tile seen and kept across
    // frames (see render() for why they are pinned).
    float*         host_buf    = nullptr;
    std::size_t    host_floats = 0;
    std::uint8_t*  host8_buf   = nullptr;
    std::size_t    host8_bytes = 0;

    ~Impl() {
        if (context)     cuCtxSetCurrent(context);
        if (host_buf)    cuMemFreeHost(host_buf);
        if (host8_buf)   cuMemFreeHost(host8_buf);
        if (out_buf)     cuMemFree_v2(out_buf);
        if (out8_buf)    cuMemFree_v2(out8_buf);
        if (params_buf)  cuMemFree_v2(params_buf);
        if (pack_module) cuModuleUnload(pack_module);
        if (module)      cuModuleUnload(module);
        if (context)     cuCtxDestroy_v2(context);
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

    // Pin this thread — the one that will launch and read back — to the
    // card's NUMA node before the context and its host allocations exist
    // (see gpu_numa.hpp). CUDA spells the address in upper-case hex; sysfs
    // wants lower case.
    {
        char bdf[32] = {0};
        if (cuDeviceGetPCIBusId(bdf, sizeof(bdf), dev) == CUDA_SUCCESS) {
            std::string s(bdf);
            for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            bind_thread_to_pci_device(s);
        }
    }

    e = cuCtxCreate_v2(&I.context, 0, dev);
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

    e = cuModuleLoadData(&I.pack_module, kPackPtx);
    if (e != CUDA_SUCCESS) return std::unexpected("cuModuleLoadData(pack): " + cu_err(e));
    e = cuModuleGetFunction(&I.pack_kernel, I.pack_module, "frep_pack_rgba8");
    if (e != CUDA_SUCCESS) return std::unexpected("cuModuleGetFunction(pack): " + cu_err(e));
    self->stats_.build_ms = ms_since(t1);

    return self;
}

std::expected<CudaCtx::Rect, std::string>
CudaCtx::launch(const CudaRenderArgs& args, const std::vector<float>& params) {
    Impl& I = *impl_;
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
    // The requested tile, clipped to the frame. The kernel addresses a
    // full-frame device buffer in tile coordinates; only this rectangle is
    // read back.
    Rect rc;
    rc.W = W; rc.H = H;
    rc.x0 = std::max(args.tx, 0); rc.y0 = std::max(args.ty, 0);
    const int x1 = std::min(args.tx + args.tw, W), y1 = std::min(args.ty + args.th, H);
    rc.tw = std::max(x1 - rc.x0, 0); rc.th = std::max(y1 - rc.y0, 0);

    const std::size_t out_floats = (std::size_t)W * H * 4;
    CUresult e;

    // (Re)allocate device buffers when sizes change.
    if (!I.out_buf || I.out_floats != out_floats) {
        if (I.out_buf) cuMemFree_v2(I.out_buf);
        e = cuMemAlloc_v2(&I.out_buf, out_floats * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAlloc out: " + cu_err(e));
        I.out_floats = out_floats;
    }
    const std::size_t pf = params.empty() ? 1 : params.size();
    if (!I.params_buf || I.params_floats != pf) {
        if (I.params_buf) cuMemFree_v2(I.params_buf);
        e = cuMemAlloc_v2(&I.params_buf, pf * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAlloc params: " + cu_err(e));
        I.params_floats = pf;
    }
    if (!params.empty()) {
        e = cuMemcpyHtoD_v2(I.params_buf, params.data(), params.size() * sizeof(float));
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
    auto t_launch = clk::now();
    e = cuLaunchKernel(I.kernel, gx ? gx : 1, gy ? gy : 1, 1, bx, by, 1,
                       0, nullptr, kargs, nullptr);
    if (e != CUDA_SUCCESS) return std::unexpected("cuLaunchKernel: " + cu_err(e));
    stats_.launch_ms = ms_since(t_launch);
    stats_.width = W; stats_.height = H;
    return rc;
}

std::expected<void, std::string>
CudaCtx::render(const CudaRenderArgs& args,
                const std::vector<float>& params,
                std::vector<float>& out_rgba) {
    Impl& I = *impl_;
    auto t0 = clk::now();
    auto rc = launch(args, params);
    if (!rc) return std::unexpected(rc.error());
    const int W = rc->W, x0 = rc->x0, y0 = rc->y0, tw = rc->tw, th = rc->th;
    CUresult e = CUDA_SUCCESS;

    // Read the tile back. A synchronous copy on the legacy stream waits for
    // the kernel first, so no cuCtxSynchronize is needed. The destination is
    // pinned and kept across frames: with a pageable destination the driver
    // stages the copy through its own bounce buffer at roughly half the PCIe
    // rate, and a fresh std::vector per frame would add a page-faulting
    // zero-fill of the whole frame on top of that. Only the tile rectangle
    // crosses the bus; the old full-frame readback made a 64-px tile at
    // 1080p cost the same 33 MB as the whole picture.
    auto t_copy = clk::now();
    const std::size_t tile_floats = (std::size_t)tw * th * 4;
    if (tile_floats > I.host_floats) {
        if (I.host_buf) cuMemFreeHost(I.host_buf);
        I.host_buf = nullptr; I.host_floats = 0;
        void* p = nullptr;
        e = cuMemAllocHost_v2(&p, tile_floats * sizeof(float));
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAllocHost: " + cu_err(e));
        I.host_buf = static_cast<float*>(p);
        I.host_floats = tile_floats;
    }
    if (tile_floats > 0) {
        if (tw == W && th == rc->H) {
            e = cuMemcpyDtoH_v2(I.host_buf, I.out_buf, tile_floats * sizeof(float));
        } else {
            CUDA_MEMCPY2D c{};
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            c.srcDevice     = I.out_buf;
            c.srcXInBytes   = (std::size_t)x0 * 4 * sizeof(float);
            c.srcY          = (std::size_t)y0;
            c.srcPitch      = (std::size_t)W * 4 * sizeof(float);
            c.dstMemoryType = CU_MEMORYTYPE_HOST;
            c.dstHost       = I.host_buf;
            c.dstPitch      = (std::size_t)tw * 4 * sizeof(float);
            c.WidthInBytes  = (std::size_t)tw * 4 * sizeof(float);
            c.Height        = (std::size_t)th;
            e = cuMemcpy2D_v2(&c);
        }
        if (e != CUDA_SUCCESS) return std::unexpected("readback: " + cu_err(e));
    }
    stats_.copy_ms = ms_since(t_copy);

    out_rgba.assign(I.host_buf, I.host_buf + tile_floats);
    stats_.render_ms = ms_since(t0);
    return {};
}

std::expected<void, std::string>
CudaCtx::render(const CudaRenderArgs& args,
                const std::vector<float>& params,
                std::vector<std::uint8_t>& out_rgba8) {
    Impl& I = *impl_;
    auto t0 = clk::now();
    auto rc = launch(args, params);
    if (!rc) return std::unexpected(rc.error());
    const int W = rc->W, H = rc->H, x0 = rc->x0, y0 = rc->y0, tw = rc->tw, th = rc->th;
    CUresult e = CUDA_SUCCESS;

    // Pack the whole frame to RGBA8 on the device (the kernel wrote float4;
    // the pack is one thread per pixel and runs in the kernel's shadow), then
    // read back only the tile rectangle, a quarter of the float bytes.
    auto t_copy = clk::now();
    const std::size_t px = (std::size_t)W * H;
    if (!I.out8_buf || I.out8_px != px) {
        if (I.out8_buf) cuMemFree_v2(I.out8_buf);
        e = cuMemAlloc_v2(&I.out8_buf, px * 4);
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAlloc out8: " + cu_err(e));
        I.out8_px = px;
    }
    {
        unsigned n = (unsigned)px;
        void* pargs[] = { &I.out_buf, &I.out8_buf, &n };
        const unsigned bs = 256;
        e = cuLaunchKernel(I.pack_kernel, (n + bs - 1) / bs, 1, 1, bs, 1, 1,
                           0, nullptr, pargs, nullptr);
        if (e != CUDA_SUCCESS) return std::unexpected("cuLaunchKernel(pack): " + cu_err(e));
    }
    const std::size_t tile_bytes = (std::size_t)tw * th * 4;
    if (tile_bytes > I.host8_bytes) {
        if (I.host8_buf) cuMemFreeHost(I.host8_buf);
        I.host8_buf = nullptr; I.host8_bytes = 0;
        void* p = nullptr;
        e = cuMemAllocHost_v2(&p, tile_bytes);
        if (e != CUDA_SUCCESS) return std::unexpected("cuMemAllocHost: " + cu_err(e));
        I.host8_buf = static_cast<std::uint8_t*>(p);
        I.host8_bytes = tile_bytes;
    }
    if (tile_bytes > 0) {
        if (tw == W && th == H) {
            e = cuMemcpyDtoH_v2(I.host8_buf, I.out8_buf, tile_bytes);
        } else {
            CUDA_MEMCPY2D c{};
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            c.srcDevice     = I.out8_buf;
            c.srcXInBytes   = (std::size_t)x0 * 4;
            c.srcY          = (std::size_t)y0;
            c.srcPitch      = (std::size_t)W * 4;
            c.dstMemoryType = CU_MEMORYTYPE_HOST;
            c.dstHost       = I.host8_buf;
            c.dstPitch      = (std::size_t)tw * 4;
            c.WidthInBytes  = (std::size_t)tw * 4;
            c.Height        = (std::size_t)th;
            e = cuMemcpy2D_v2(&c);
        }
        if (e != CUDA_SUCCESS) return std::unexpected("readback: " + cu_err(e));
    }
    stats_.copy_ms = ms_since(t_copy);

    out_rgba8.assign(I.host8_buf, I.host8_buf + tile_bytes);
    stats_.render_ms = ms_since(t0);
    return {};
}

CudaCtx::~CudaCtx() = default;

} // namespace frep::gpu
