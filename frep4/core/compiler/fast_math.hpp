#pragma once
// core/compiler/fast_math.hpp
//
// The relaxed floating-point flags both JIT paths apply before optimisation.
//
// Without any flags LLVM keeps every FP expression in source order: no fused
// multiply-add, no reassociation, no reciprocal, min/max kept as the IEEE
// select sequences the node emitters wrote. Bit-exactness across paths is not
// worth that here: the paths are compared at a tolerance of a few 1e-4 per
// channel (frep4_scene_parity), far above what reassociation moves, so the
// flags that buy speed and stay well inside the tolerance are on.
//
//   contract  — fuse a*b+c into an FMA (one rounding instead of two).
//   nsz       — ignore the sign of zero.
//   arcp      — allow x/y as x*(1/y).
//   reassoc   — allow reassociation / commutation of a chain.
//
// Not set, deliberately: nnan/ninf (the emitters use large finite sentinels
// and compare against them, and a NaN out of sqrt(negative) must still
// propagate to a visible pixel rather than be assumed away) and afn (the
// backend would then replace sqrt/div with approximate forms whose error is
// visible). FREP_FAST_MATH=0 turns the flags off for an A/B.
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <cstdlib>

namespace frep {

inline bool fast_math_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("FREP_FAST_MATH");
        return !(s && s[0] == '0');
    }();
    return on;
}

inline void apply_fast_math(llvm::Module& m) {
    if (!fast_math_enabled()) return;
    llvm::FastMathFlags f;
    f.setAllowContract(true);
    f.setNoSignedZeros(true);
    f.setAllowReciprocal(true);
    f.setAllowReassoc(true);
    for (auto& fn : m)
        for (auto& bb : fn)
            for (auto& inst : bb)
                if (llvm::isa<llvm::FPMathOperator>(inst))
                    inst.setFastMathFlags(f);
}

} // namespace frep
