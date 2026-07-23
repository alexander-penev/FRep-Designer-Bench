#pragma once
// core/compiler/jit_engine.hpp
//
// LLJIT wrapper.
// Takes an llvm::Module + LLVMContext (together), optimizes with the O3
// PassManager, loads into the JIT and returns a function pointer to render_tile.

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <expected>
#include <memory>
#include <string>

namespace frep {

// Signature — must match emit_tracer() exactly.
using RenderTileFn = void(*)(
    float* out_rgba,
    int tx, int ty, int tw, int th, int iw, int ih,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float rx, float ry, float rz,
    float ux, float uy, float uz,
    float fov_scale,
    float* params              // incremental-mode parameter buffer; may be null
);

// Signature of scene_pick — matches emit_scene_pick().
// Casts a ray (ox,oy,oz)+(dx,dy,dz), returns the hit object index or -1.
using ScenePickFn = int(*)(
    float ox, float oy, float oz,
    float dx, float dy, float dz
);

class JitEngine {
public:
    JitEngine() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        auto j = llvm::orc::LLJITBuilder().create();
        if (!j) {
            llvm::errs() << "LLJIT create: "
                         << llvm::toString(j.takeError()) << "\n";
            std::abort();
        }
        jit_ = std::move(*j);
    }

    // Loads the module. The module and the context are merged into a ThreadSafeModule.
    // IMPORTANT: the context must be the one the module was created in.
    std::expected<RenderTileFn, std::string>
    load(std::unique_ptr<llvm::Module>      mod,
         std::unique_ptr<llvm::LLVMContext> ctx)
    {
        optimize(*mod, opt_level_);

        // ThreadSafeModule takes ownership of both the module and the context.
        auto tsm = llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx));

        if (auto err = jit_->addIRModule(std::move(tsm)))
            return std::unexpected(llvm::toString(std::move(err)));

        auto sym = jit_->lookup("render_tile");
        if (!sym)
            return std::unexpected(llvm::toString(sym.takeError()));

        return sym->toPtr<RenderTileFn>();
    }

    // Set the optimization level applied in optimize(). Default is O3
    // (full inlining + vectorization — best render throughput, but the
    // O3 pipeline dominates JIT compile time on large unrolled scenes).
    // Lower levels (O0/O1/O2) trade render speed for far faster compile;
    // the benchmark sweeps them to quantify that trade-off. Accepts the
    // LLVM OptimizationLevel directly.
    void set_opt_level(llvm::OptimizationLevel lvl) { opt_level_ = lvl; }
    llvm::OptimizationLevel opt_level() const { return opt_level_; }

    // Generic variant — loads the module and resolves a symbol by name.
    // Allows JIT-ing functions other than render_tile (e.g. scene_pick).
    template <typename FnPtr>
    std::expected<FnPtr, std::string>
    load_as(std::unique_ptr<llvm::Module>      mod,
            std::unique_ptr<llvm::LLVMContext> ctx,
            const std::string&                 symbol)
    {
        optimize(*mod, opt_level_);

        auto tsm = llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx));
        if (auto err = jit_->addIRModule(std::move(tsm)))
            return std::unexpected(llvm::toString(std::move(err)));

        auto sym = jit_->lookup(symbol);
        if (!sym)
            return std::unexpected(llvm::toString(sym.takeError()));

        return sym->toPtr<FnPtr>();
    }

    static void dump(llvm::Module& m) { m.print(llvm::errs(), nullptr); }

private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    llvm::OptimizationLevel opt_level_ = llvm::OptimizationLevel::O3;

    static void optimize(llvm::Module& mod, llvm::OptimizationLevel lvl) {
        llvm::PassBuilder             pb;
        llvm::LoopAnalysisManager     lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager    cgam;
        llvm::ModuleAnalysisManager   mam;

        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        // O0 has no module pipeline in the new PM — buildPerModuleDefault
        // pipeline asserts on O0 — so route it through the O0-specific
        // builder. O1/O2/O3 use the standard per-module pipeline (O3 adds
        // aggressive inlining + vectorization; the cost the sweep measures).
        if (lvl == llvm::OptimizationLevel::O0)
            pb.buildO0DefaultPipeline(lvl).run(mod, mam);
        else
            pb.buildPerModuleDefaultPipeline(lvl).run(mod, mam);
    }
};

} // namespace frep
