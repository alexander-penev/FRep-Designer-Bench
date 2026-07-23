// core/frep/node.cpp
//
// Default implementation of codegen_grad — finite-difference fallback.
// Overridden in the primitives for direct (symbolic) AD.

#include "node.hpp"

#include <llvm/IR/Intrinsics.h>

namespace frep {

FRepNode::DualVal FRepNode::codegen_grad(CgCtx&  ctx,
                                          DualVal x,
                                          DualVal y,
                                          DualVal z) const
{
    // Finite difference approximation:
    //   dot = (f(x+h*x.dot, y+h*y.dot, z+h*z.dot) - f(x-...))/(2h)
    //
    // For small h this is an approximate directional derivative along (x.dot, y.dot, z.dot).
    auto& b = ctx.b;
    const float h = 1e-3f;
    auto hc = ctx.fc(h);

    auto pxp = b.CreateFAdd(x.val, b.CreateFMul(hc, x.dot));
    auto pyp = b.CreateFAdd(y.val, b.CreateFMul(hc, y.dot));
    auto pzp = b.CreateFAdd(z.val, b.CreateFMul(hc, z.dot));

    auto pxm = b.CreateFSub(x.val, b.CreateFMul(hc, x.dot));
    auto pym = b.CreateFSub(y.val, b.CreateFMul(hc, y.dot));
    auto pzm = b.CreateFSub(z.val, b.CreateFMul(hc, z.dot));

    auto val_p = codegen(ctx, pxp, pyp, pzp);
    auto val_m = codegen(ctx, pxm, pym, pzm);
    auto val   = codegen(ctx, x.val, y.val, z.val);

    auto diff   = b.CreateFSub(val_p, val_m);
    auto two_h  = ctx.fc(2 * h);
    auto dot    = b.CreateFDiv(diff, two_h);

    return {val, dot};
}

} // namespace frep
