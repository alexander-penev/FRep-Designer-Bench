// tests/test_nvptx.cpp
//
// Tests for the NVPTX retarget (GPU-IR path on NVIDIA). These check PTX
// *generation* from the per-pixel kernel — they don't run on the GPU
// (no CUDA device in CI; that's validated on hardware).

#include <gtest/gtest.h>

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/retarget_nvptx.hpp"
#include "core/exec/parity_scenes.hpp"

#include <llvm/Support/TargetSelect.h>
#include <memory>
#include <mutex>

using namespace frep;

namespace {
void init_targets_once() {
    static std::once_flag f;
    std::call_once(f, [] {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
    });
}

SceneGraph one_sphere() {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    s.camera().position = {4, 3, 4};
    s.camera().target = {0, 0, 0};
    return s;
}
} // namespace

TEST(NVPTX, EmitsKernelPTXFromGpuKernel) {
    init_targets_once();
    SceneGraph s = one_sphere();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_gpu_kernel(s);
    auto mod = cg.take_module();

    NVPTXRetarget r;
    auto ptx = r.retarget(*mod);
    ASSERT_TRUE(ptx) << r.last_error;
    EXPECT_GT(ptx->size(), 1000u);
    // The kernel entry must be present, and it must read the thread id —
    // i.e. it really is a per-pixel kernel, not the single-thread loop.
    EXPECT_NE(ptx->find(".entry render_tile"), std::string::npos)
        << "no kernel entry in PTX";
    EXPECT_NE(ptx->find("%tid.x"), std::string::npos)
        << "kernel doesn't read thread id (not per-pixel)";
}

TEST(NVPTX, GpuKernelMatchesCpuRenderTileSignature) {
    // The GPU kernel must keep render_tile's name + arg count so CudaCtx can
    // launch it with the same argument list the CPU path uses.
    init_targets_once();
    SceneGraph s = one_sphere();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;
    SceneCodegen cg(*ctx, cfg);
    auto* k = cg.emit_gpu_kernel(s);
    ASSERT_NE(k, nullptr);
    EXPECT_EQ(k->getName(), "render_tile");
    // out + 6 ints + 13 floats + params = 21 args.
    EXPECT_EQ(k->arg_size(), 21u);
}

TEST(NVPTX, RetargetReportsErrorWithoutRenderTile) {
    // An empty module (no render_tile) must fail cleanly, not crash.
    init_targets_once();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("empty", *ctx);
    NVPTXRetarget r;
    auto ptx = r.retarget(*mod);
    EXPECT_FALSE(ptx);
}

// Transcendental nodes (sin/cos) must lower to an inline polynomial in IR;
// without that the NVPTX backend fails with "Cannot select: fsin", and naive
// libdevice externs make the CUDA JIT fail (unresolved __nv_*). The PTX must be
// self-contained: no extern function declarations, no unselectable intrinsic.
// Regression for the bend/twist/rotate crash on the GPU-IR path.
TEST(NVPTX, LowersTranscendentalsInline) {
    init_targets_once();
    for (const char* name : {"twist", "bend", "rotate"}) {
        SceneGraph s;
        for (auto& ns : parity::all_scenes())
            if (std::string(ns.name) == name) s = ns.make();
        auto ctx = std::make_unique<llvm::LLVMContext>();
        TracerConfig cfg;
        SceneCodegen cg(*ctx, cfg);
        cg.emit_gpu_kernel(s);
        auto mod = cg.take_module();
        NVPTXRetarget r;
        auto ptx = r.retarget(*mod);
        ASSERT_TRUE(ptx) << name << ": " << r.last_error;
        // Self-contained PTX: the polynomial leaves no external symbol to
        // resolve at JIT time (which is what broke cuModuleLoadData before).
        EXPECT_EQ(ptx->find(".extern .func"), std::string::npos)
            << name << ": PTX still has an extern func (JIT will fail)";
    }
}
