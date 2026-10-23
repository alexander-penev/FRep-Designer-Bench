// core/frep/custom_expr.cpp
//
// CustomExprNode back-ends. All three (LLVM IR, CPU eval, GLSL emit)
// walk the shared frep::expr::Node AST produced by frep::expr::parse().

#include "custom_expr.hpp"
#include "template_fn.hpp"
#include "core/compiler/llvm_compat.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#ifdef FREP_HAS_ADIR
#include <adir/Differentiate.h>   // reverseGradient, for GradMethod::Adir
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace frep {

// ═════════════════════════════════════════════════════════════════════════════
// Back-end #1: LLVM IR codegen
// ═════════════════════════════════════════════════════════════════════════════

llvm::Value* CustomExprCompiler::fc(float v) {
    return llvm::ConstantFP::get(f32(), v);
}

llvm::Value* CustomExprCompiler::gen(const expr::Node& n) {
    using Kind = expr::Node::Kind;
    if (auto it = memo_.find(&n); it != memo_.end()) return it->second;  // shared subtree
    llvm::Value* out = nullptr;
    switch (n.kind) {
        case Kind::Number: return fc(n.num);

        case Kind::Var:
            if (n.ident == "x") return vx_;
            if (n.ident == "y") return vy_;
            if (n.ident == "z") return vz_;
            if (auto it = params_.find(n.ident); it != params_.end())
                return it->second;   // template scalar parameter
            fail("unknown variable '" + n.ident + "'");
            return nullptr;

        case Kind::Const:
            if (n.ident == "pi") return fc(std::numbers::pi_v<float>);
            if (n.ident == "e")  return fc(std::numbers::e_v<float>);
            fail("unknown constant '" + n.ident + "'");
            return nullptr;

        case Kind::UnaryNeg: {
            auto v = gen(*n.children[0]);
            if (!v) return nullptr;
            out = b_->CreateFNeg(v);
            break;
        }

        case Kind::BinOp: {
            auto l = gen(*n.children[0]);
            auto r = gen(*n.children[1]);
            if (!l || !r) return nullptr;
            switch (n.bop) {
                case expr::Op::Add: out = b_->CreateFAdd(l, r); break;
                case expr::Op::Sub: out = b_->CreateFSub(l, r); break;
                case expr::Op::Mul: out = b_->CreateFMul(l, r); break;
                case expr::Op::Div: out = b_->CreateFDiv(l, r); break;
            }
            break;
        }

        case Kind::Call: out = gen_call(n); break;
    }
    if (out) memo_[&n] = out;   // cache shared subtrees (BinOp/Neg/Call)
    return out;
}

llvm::Value* CustomExprCompiler::gen_call(const expr::Node& n) {
    using llvm::Intrinsic::ID;
    auto& b = *b_;

    // Pre-emit arg LLVM values.
    std::vector<llvm::Value*> args;
    args.reserve(n.children.size());
    for (const auto& a : n.children) {
        auto v = gen(*a);
        if (!v) return nullptr;
        args.push_back(v);
    }

    auto unary = [&](ID id) {
        return frep::llvm_compat::unary_intrinsic(b, id, args[0]);
    };
    auto binary = [&](ID id) {
        return frep::llvm_compat::binary_intrinsic(b, id, args[0], args[1]);
    };

    const auto& name = n.ident;
    if (name == "sqrt")  return unary(llvm::Intrinsic::sqrt);
    if (name == "abs")   return unary(llvm::Intrinsic::fabs);
    if (name == "sin")   return unary(llvm::Intrinsic::sin);
    if (name == "cos")   return unary(llvm::Intrinsic::cos);
    if (name == "exp")   return unary(llvm::Intrinsic::exp);
    if (name == "log")   return unary(llvm::Intrinsic::log);
    if (name == "floor") return unary(llvm::Intrinsic::floor);
    if (name == "ceil")  return unary(llvm::Intrinsic::ceil);
    if (name == "min")   return binary(llvm::Intrinsic::minnum);
    if (name == "max")   return binary(llvm::Intrinsic::maxnum);
    //  Domain-safe power: copysign(|a|^e, a). Raw IEEE pow(a<0, non-integer e)
    //  is NaN, and the converted gear/architecture/hello_world scenes raise a
    //  signed field to fractional powers (via nth_root b=0.3/1.5), so the naive
    //  form NaNs over the whole volume. The odd extension keeps them finite.
    //  For a >= 0 it is identical to raw pow, so canonical scenes (blend h^k,
    //  h >= 0) are unaffected. NOTE: libfive does NOT do this — it returns NaN
    //  there — so enabling this makes frep4 DIVERGE from libfive on those
    //  scenes rather than agree. Gated on FREP4_SAFE_POW so the default stays
    //  bit-identical to libfive for honest cross-system parity.
    auto pow_safe = [&](llvm::Value* base, llvm::Value* e) -> llvm::Value* {
        auto* P = llvm::Intrinsic::getOrInsertDeclaration(
                mod_, llvm::Intrinsic::pow, {b.getFloatTy()});
        if (!std::getenv("FREP4_SAFE_POW"))
            return b.CreateCall(P, {base, e});
        auto* mag = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, base);
        auto* p   = b.CreateCall(P, {mag, e});
        return b.CreateBinaryIntrinsic(llvm::Intrinsic::copysign, p, base);
    };
    if (name == "pow")   return pow_safe(args[0], args[1]);
    if (name == "nth_root") {   // a^(1/b)
        auto* one = llvm::ConstantFP::get(b.getFloatTy(), 1.0);
        return pow_safe(args[0], b.CreateFDiv(one, args[1]));
    }
    // Inverse-trig / atan2 / mod: no LLVM intrinsics -> libm calls.
    // CPU-JIT resolves them from the process; GPU_IR (NVPTX) needs
    // self-contained transcendentals, so these are CPU-only for now.
    auto libm1 = [&](const char* fn) {
        auto* fty = llvm::FunctionType::get(b.getFloatTy(), {b.getFloatTy()}, false);
        return b.CreateCall(mod_->getOrInsertFunction(fn, fty), {args[0]});
    };
    auto libm2 = [&](const char* fn) {
        auto* fty = llvm::FunctionType::get(b.getFloatTy(), {b.getFloatTy(), b.getFloatTy()}, false);
        return b.CreateCall(mod_->getOrInsertFunction(fn, fty), {args[0], args[1]});
    };
    if (name == "asin")  return libm1("asinf");
    if (name == "acos")  return libm1("acosf");
    if (name == "atan")  return libm1("atanf");
    if (name == "atan2") return libm2("atan2f");
    if (name == "mod")   return libm2("fmodf");
    if (name == "tan") {
        // No tan intrinsic — build sin/cos.
        auto s = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sin, args[0]);
        auto c = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::cos, args[0]);
        return b.CreateFDiv(s, c, "tan");
    }
    // User template call: frep_tmpl_<name>(args..., x, y, z). The template's
    // function must already be in the module (emit_templates ran first).
    if (reg_ && reg_->find(name)) {
        auto* fn = mod_->getFunction("frep_tmpl_" + name);
        if (!fn) { fail("template '" + name + "' not emitted into module"); return nullptr; }
        args.push_back(vx_); args.push_back(vy_); args.push_back(vz_);
        return b.CreateCall(fn, args);
    }
    fail("unknown function '" + name + "'");
    return nullptr;
}

// ── SIMD twin ────────────────────────────────────────────────────────────────

static llvm::VectorType* vty(llvm::LLVMContext& c, unsigned w) {
    return llvm::VectorType::get(llvm::Type::getFloatTy(c), w, false);
}

llvm::Value* CustomExprCompiler::gen_vec(const expr::Node& n) {
    using Kind = expr::Node::Kind;
    if (auto it = vmemo_.find(&n); it != vmemo_.end()) return it->second;
    auto& b = *b_;
    auto splat = [&](float v) { return b.CreateVectorSplat(vw_, fc(v)); };
    llvm::Value* out = nullptr;
    switch (n.kind) {
        case Kind::Number: return splat(n.num);
        case Kind::Const:
            return splat(n.ident == "pi" ? std::numbers::pi_v<float>
                                         : std::numbers::e_v<float>);
        case Kind::Var:
            if (n.ident == "x") return vx_;   // already <W x float> args
            if (n.ident == "y") return vy_;
            if (n.ident == "z") return vz_;
            if (auto it = vparams_.find(n.ident); it != vparams_.end())
                return it->second;            // template scalar param (broadcast)
            fail("unknown variable '" + n.ident + "'"); return nullptr;
        case Kind::UnaryNeg: {
            auto v = gen_vec(*n.children[0]); if (!v) return nullptr;
            out = b.CreateFNeg(v); break;
        }
        case Kind::BinOp: {
            auto l = gen_vec(*n.children[0]);
            auto r = gen_vec(*n.children[1]);
            if (!l || !r) return nullptr;
            switch (n.bop) {
                case expr::Op::Add: out = b.CreateFAdd(l, r); break;
                case expr::Op::Sub: out = b.CreateFSub(l, r); break;
                case expr::Op::Mul: out = b.CreateFMul(l, r); break;
                case expr::Op::Div: out = b.CreateFDiv(l, r); break;
            }
            break;
        }
        case Kind::Call: out = gen_call_vec(n); break;
    }
    if (out) vmemo_[&n] = out;
    return out;
}

// ── Interval twin ────────────────────────────────────────────────────────────
// {lo,hi} arithmetic on the same AST for octree region pruning.

std::pair<llvm::Value*,llvm::Value*>
CustomExprCompiler::gen_ival(const expr::Node& n) {
    using Kind = expr::Node::Kind;
    if (auto it = imemo_.find(&n); it != imemo_.end()) return it->second;
    auto& b = *b_;
    auto k = [&](float v){ return fc(v); };
    auto mn = [&](llvm::Value* a, llvm::Value* c){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::minnum,{b.getFloatTy()}),{a,c}); };
    auto mx = [&](llvm::Value* a, llvm::Value* c){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::maxnum,{b.getFloatTy()}),{a,c}); };
    std::pair<llvm::Value*,llvm::Value*> out{nullptr,nullptr};
    switch (n.kind) {
        case Kind::Number: out = {k(n.num), k(n.num)}; break;
        case Kind::Const:  { float v = n.ident=="pi"?std::numbers::pi_v<float>:std::numbers::e_v<float>;
                             out = {k(v),k(v)}; break; }
        case Kind::Var:
            if (n.ident=="x") out={xlo_,xhi_};
            else if (n.ident=="y") out={ylo_,yhi_};
            else if (n.ident=="z") out={zlo_,zhi_};
            else if (auto it=iparams_.find(n.ident); it!=iparams_.end()) out=it->second;
            else { fail("unknown var '"+n.ident+"'"); return {}; }
            break;
        case Kind::UnaryNeg: {
            auto [l,h]=gen_ival(*n.children[0]); if(!l) return {};
            out={b.CreateFNeg(h), b.CreateFNeg(l)}; break;
        }
        case Kind::BinOp: {
            auto [al,ah]=gen_ival(*n.children[0]);
            auto [cl,ch]=gen_ival(*n.children[1]); if(!al||!cl) return {};
            switch (n.bop) {
                case expr::Op::Add: out={b.CreateFAdd(al,cl), b.CreateFAdd(ah,ch)}; break;
                case expr::Op::Sub: out={b.CreateFSub(al,ch), b.CreateFSub(ah,cl)}; break;
                case expr::Op::Mul: {
                    auto& c0=*n.children[0]; auto& c1=*n.children[1];
                    bool sq = (&c0==&c1) ||
                              (c0.kind==expr::Node::Kind::Var &&
                               c1.kind==expr::Node::Kind::Var && c0.ident==c1.ident);
                    if (sq) { // x*x: true square, min is 0 when interval spans 0
                        auto l2=b.CreateFMul(al,al), h2=b.CreateFMul(ah,ah);
                        auto spans0=b.CreateAnd(b.CreateFCmpOLE(al,fc(0.0f)),b.CreateFCmpOGE(ah,fc(0.0f)));
                        out={b.CreateSelect(spans0,fc(0.0f),mn(l2,h2)), mx(l2,h2)}; break;
                    }
                    auto p1=b.CreateFMul(al,cl),p2=b.CreateFMul(al,ch),
                         p3=b.CreateFMul(ah,cl),p4=b.CreateFMul(ah,ch);
                    out={mn(mn(p1,p2),mn(p3,p4)), mx(mx(p1,p2),mx(p3,p4))}; break; }
                case expr::Op::Div: { // assumes divisor interval excludes 0 (scenes hold constants)
                                      auto q1=b.CreateFDiv(al,cl),q2=b.CreateFDiv(al,ch),
                                           q3=b.CreateFDiv(ah,cl),q4=b.CreateFDiv(ah,ch);
                                      out={mn(mn(q1,q2),mn(q3,q4)), mx(mx(q1,q2),mx(q3,q4))}; break; }
            }
            break;
        }
        case Kind::Call: out = gen_call_ival(n); break;
    }
    if (out.first) imemo_[&n]=out;
    return out;
}

std::pair<llvm::Value*,llvm::Value*>
CustomExprCompiler::gen_call_ival(const expr::Node& n) {
    auto& b = *b_;
    const auto& nm = n.ident;
    auto I=[&](llvm::Intrinsic::ID id,llvm::Value* v){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(mod_,id,{b.getFloatTy()}),{v}); };
    auto mn=[&](llvm::Value* a,llvm::Value* c){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::minnum,{b.getFloatTy()}),{a,c}); };
    auto mx=[&](llvm::Value* a,llvm::Value* c){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::maxnum,{b.getFloatTy()}),{a,c}); };
    std::vector<std::pair<llvm::Value*,llvm::Value*>> a;
    for (auto& c:n.children){ auto p=gen_ival(*c); if(!p.first) return {}; a.push_back(p); }
    if (nm=="sqrt"){ auto lo=mx(a[0].first, fc(0.0f));
                     return {I(llvm::Intrinsic::sqrt,lo), I(llvm::Intrinsic::sqrt,mx(a[0].second,fc(0.0f)))}; }
    if (nm=="abs"){ auto&[l,h]=a[0];
        auto al=I(llvm::Intrinsic::fabs,l), ah=I(llvm::Intrinsic::fabs,h);
        auto spans0=b.CreateAnd(b.CreateFCmpOLE(l,fc(0.0f)), b.CreateFCmpOGE(h,fc(0.0f)));
        return {b.CreateSelect(spans0, fc(0.0f), mn(al,ah)), mx(al,ah)}; }
    if (nm=="min") return {mn(a[0].first,a[1].first), mn(a[0].second,a[1].second)};
    if (nm=="max") return {mx(a[0].first,a[1].first), mx(a[0].second,a[1].second)};
    if (nm=="pow"){ // constant exponent, non-negative base (holds for the blend h^k)
        auto*P=llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::pow,{b.getFloatTy()});
        auto lo=mx(a[0].first,fc(0.0f)), hi=mx(a[0].second,fc(0.0f));
        return {b.CreateCall(P,{lo,a[1].first}), b.CreateCall(P,{hi,a[1].second})}; }
    if (nm=="nth_root"){ // a^(1/b), same clamped-base convention as pow above
        auto*P=llvm::Intrinsic::getOrInsertDeclaration(mod_,llvm::Intrinsic::pow,{b.getFloatTy()});
        auto lo=mx(a[0].first,fc(0.0f)), hi=mx(a[0].second,fc(0.0f));
        auto elo=b.CreateFDiv(fc(1.0f),a[1].first), ehi=b.CreateFDiv(fc(1.0f),a[1].second);
        return {b.CreateCall(P,{lo,elo}), b.CreateCall(P,{hi,ehi})}; }
    // trig etc.: not yet interval-supported
    auto Fc=[&](float v){ return fc(v); };
    auto Sin=[&](llvm::Value* v){ return I(llvm::Intrinsic::sin,v); };
    auto Cos=[&](llvm::Value* v){ return I(llvm::Intrinsic::cos,v); };
    // sin/cos over [lo,hi]: start from endpoint values, then pull the interval to
    // [-1,1] toward any extremum the range crosses. Extremum of sin at pi/2+k*pi
    // (cos there = -+1); of cos at k*pi. We test inclusion via floor counting and
    // widen with select (branchless, scalar interval fn).
    // exists k with lo <= phase + k*pi <= hi, i.e. floor((hi-phase)/pi) >= ceil((lo-phase)/pi)
    auto crosses_pi=[&](llvm::Value* lo, llvm::Value* hi, llvm::Value* phase){
        auto PI=Fc(3.14159265359f);
        auto a1=b.CreateFDiv(b.CreateFSub(hi,phase),PI);
        auto c1=b.CreateFDiv(b.CreateFSub(lo,phase),PI);
        auto fa=I(llvm::Intrinsic::floor,a1);
        auto cc=b.CreateFNeg(I(llvm::Intrinsic::floor,b.CreateFNeg(c1)));  // ceil
        return b.CreateFCmpOGE(fa,cc);
    };
    auto trig=[&](llvm::Value* lo, llvm::Value* hi, bool isSin)
        -> std::pair<llvm::Value*,llvm::Value*> {
        auto el=isSin?Sin(lo):Cos(lo), eh=isSin?Sin(hi):Cos(hi);
        llvm::Value* rlo=mn(el,eh); llvm::Value* rhi=mx(el,eh);
        auto PI=Fc(3.14159265359f), HALF=Fc(1.57079632679f);
        auto crosses=[&](llvm::Value* phase){ return crosses_pi(lo,hi,phase); };
        // max at phase where fn=+1, min where fn=-1
        auto maxPhase=isSin?HALF:Fc(0.0f);
        auto minPhase=isSin?Fc(-1.57079632679f):PI;
        rhi=b.CreateSelect(crosses(maxPhase), Fc(1.0f), rhi);
        rlo=b.CreateSelect(crosses(minPhase), Fc(-1.0f), rlo);
        return {rlo,rhi};
    };
    if (nm=="sin") return trig(a[0].first,a[0].second,true);
    if (nm=="cos") return trig(a[0].first,a[0].second,false);
    if (nm=="tan"){
        // tan has a pole at pi/2 + k*pi. If the argument range crosses one, the
        // true range is all of R; only otherwise is tan monotone on [lo,hi] and
        // bounded by its endpoints.
        auto lo=a[0].first, hi=a[0].second;
        auto t0=b.CreateFDiv(Sin(lo),Cos(lo));
        auto t1=b.CreateFDiv(Sin(hi),Cos(hi));
        auto pole=crosses_pi(lo,hi,Fc(1.57079632679f));
        const float BIG=3.4e38f;
        return {b.CreateSelect(pole,Fc(-BIG),mn(t0,t1)),
                b.CreateSelect(pole,Fc( BIG),mx(t0,t1))};
    }
    // no LLVM asin/acos/atan intrinsics -> reuse scalar libm at interval endpoints
    auto libm=[&](const char* fn, llvm::Value* v){
        auto*t=llvm::FunctionType::get(b.getFloatTy(),{b.getFloatTy()},false);
        return b.CreateCall(mod_->getOrInsertFunction(fn,t),{v}); };
    if (nm=="asin") return {libm("asinf",a[0].first), libm("asinf",a[0].second)};
    if (nm=="acos") return {libm("acosf",a[0].second), libm("acosf",a[0].first)}; // decreasing
    if (nm=="atan") return {libm("atanf",a[0].first), libm("atanf",a[0].second)};
    if (nm=="atan2"){
        // atan2 is the polar angle: its level sets are rays from the origin, so
        // over a box that contains neither the origin nor the branch cut (the
        // negative x-axis) the extremes sit at box corners. The cut/origin is
        // inside exactly when xlo <= 0 and the y-range straddles 0; then the
        // range is the full circle.
        auto ylo=a[0].first, yhi=a[0].second, xlo=a[1].first, xhi=a[1].second;
        auto lm2=[&](llvm::Value* y, llvm::Value* x){
            auto*t=llvm::FunctionType::get(b.getFloatTy(),{b.getFloatTy(),b.getFloatTy()},false);
            return b.CreateCall(mod_->getOrInsertFunction("atan2f",t),{y,x}); };
        auto c00=lm2(ylo,xlo), c01=lm2(ylo,xhi), c10=lm2(yhi,xlo), c11=lm2(yhi,xhi);
        auto cmin=mn(mn(c00,c01),mn(c10,c11));
        auto cmax=mx(mx(c00,c01),mx(c10,c11));
        auto cut=b.CreateAnd(b.CreateFCmpOLE(xlo,Fc(0.0f)),
                 b.CreateAnd(b.CreateFCmpOLE(ylo,Fc(0.0f)),
                             b.CreateFCmpOGE(yhi,Fc(0.0f))));
        return {b.CreateSelect(cut,Fc(-3.14159265359f),cmin),
                b.CreateSelect(cut,Fc( 3.14159265359f),cmax)};
    }
    // User template call: inline the body with the argument intervals bound to
    // the template's parameters (fresh memo scope per call, as in gen_call_vec).
    if (reg_) {
        if (const TemplateFn* t = reg_->find(nm)) {
            auto saved_params = iparams_;
            auto saved_memo   = std::move(imemo_); imemo_.clear();
            for (std::size_t i = 0; i < t->params.size(); ++i)
                iparams_[t->params[i]] = a[i];
            auto r = gen_ival(*t->body);
            iparams_ = std::move(saved_params);
            imemo_   = std::move(saved_memo);
            return r;
        }
    }
    fail("interval unsupported fn '"+nm+"'"); return {};
}

llvm::Function* CustomExprCompiler::compile_interval(llvm::Module&        mod,
                                                     llvm::LLVMContext&   ctx,
                                                     const std::string&   fn_name,
                                                     const expr::NodePtr& ast) {
    error_.clear();
    if (!ast){ fail("null AST"); return nullptr; }
    if (mod.getFunction(fn_name)){ fail("fn exists"); return nullptr; }
    ctx_=&ctx; mod_=&mod; imemo_.clear();
    auto* pf=llvm::PointerType::getUnqual(ctx);
    auto* fty=llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),{pf,pf},false);
    auto* fn=llvm::Function::Create(fty,llvm::Function::ExternalLinkage,fn_name,mod);
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    auto it=fn->arg_begin(); llvm::Value* B=&*it++; llvm::Value* O=&*it++;
    auto* bb=llvm::BasicBlock::Create(ctx,"entry",fn);
    llvm::IRBuilder<> b(bb); b_=&b;
    auto* f32=b.getFloatTy();
    auto ld=[&](int i){ return b.CreateAlignedLoad(f32,
        b.CreateConstInBoundsGEP1_32(f32,B,i), llvm::MaybeAlign(4)); };
    xlo_=ld(0); xhi_=ld(1); ylo_=ld(2); yhi_=ld(3); zlo_=ld(4); zhi_=ld(5);
    auto [lo,hi]=gen_ival(*ast);
    if(!lo||!error_.empty()){ fn->eraseFromParent(); return nullptr; }
    b.CreateAlignedStore(lo, b.CreateConstInBoundsGEP1_32(f32,O,0), llvm::MaybeAlign(4));
    b.CreateAlignedStore(hi, b.CreateConstInBoundsGEP1_32(f32,O,1), llvm::MaybeAlign(4));
    b.CreateRetVoid();
    std::string vfy; llvm::raw_string_ostream es(vfy);
    if(llvm::verifyFunction(*fn,&es)){ fail("verify: "+vfy); fn->eraseFromParent(); return nullptr; }
    return fn;
}

// ── Vectorizable polynomial approximations ──────────────────────────────────
// Range-reduced minimax; f32 accuracy ~1e-6 (sin/cos), ~2e-6 (asin/atan).
// All ops are lane-parallel (no libm, no per-lane extract), so they lower to
// real SIMD on any target — unlike llvm.sin.vNf32 which scalarizes here.
namespace {
struct VPoly {
    llvm::IRBuilder<>& b; llvm::VectorType* vt; unsigned W;
    llvm::Value* C(float v){ return b.CreateVectorSplat(W, llvm::ConstantFP::get(b.getFloatTy(), v)); }
    llvm::Value* fma(llvm::Value* a, llvm::Value* x, llvm::Value* c){
        return b.CreateFAdd(b.CreateFMul(a, x), c); }
    llvm::Value* intr(llvm::Intrinsic::ID id, llvm::Value* x){
        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
            b.GetInsertBlock()->getModule(), id, {vt}), {x}); }
    // sin/cos via reduction to [-pi/4,pi/4] + quadrant swap (accurate to ~1e-6).
    void sincos(llvm::Value* x, llvm::Value** so, llvm::Value** co){
        auto TWO_OVER_PI=C(0.63661977236f), HALFPI=C(1.57079632679f);
        auto k=intr(llvm::Intrinsic::round, b.CreateFMul(x, TWO_OVER_PI)); // nearest quadrant
        auto r=b.CreateFSub(x, b.CreateFMul(k, HALFPI));                    // r in [-pi/4,pi/4]
        auto r2=b.CreateFMul(r,r);
        // base sin(r), cos(r) on the small interval
        auto sp=fma(C(-1.0f/5040), r2, C(1.0f/120));
        sp=fma(sp, r2, C(-1.0f/6)); sp=fma(sp, r2, C(1.0f)); auto sr=b.CreateFMul(r, sp);
        auto cp=fma(C(1.0f/40320), r2, C(-1.0f/720));
        cp=fma(cp, r2, C(1.0f/24)); cp=fma(cp, r2, C(-1.0f/2)); cp=fma(cp, r2, C(1.0f)); auto cr=cp;
        // quadrant q = k mod 4 -> select/sign
        auto ki=b.CreateFPToSI(k, llvm::VectorType::get(b.getInt32Ty(), W, false));
        auto q=b.CreateAnd(ki, b.CreateVectorSplat(W, b.getInt32(3)));
        auto is=[&](int v){ return b.CreateICmpEQ(q, b.CreateVectorSplat(W, b.getInt32(v))); };
        auto neg=[&](llvm::Value* v){ return b.CreateFNeg(v); };
        if (so) { // q0:sr q1:cr q2:-sr q3:-cr
            auto s=b.CreateSelect(is(0), sr,
                    b.CreateSelect(is(1), cr,
                    b.CreateSelect(is(2), neg(sr), neg(cr))));
            *so=s;
        }
        if (co) { // q0:cr q1:-sr q2:-cr q3:sr
            auto c=b.CreateSelect(is(0), cr,
                    b.CreateSelect(is(1), neg(sr),
                    b.CreateSelect(is(2), neg(cr), sr)));
            *co=c;
        }
    }
    llvm::Value* sin(llvm::Value* x){ llvm::Value* s; sincos(x,&s,nullptr); return s; }
    llvm::Value* cos(llvm::Value* x){ llvm::Value* c; sincos(x,nullptr,&c); return c; }
    // atan on full range via minimax on |x|<=1 + reciprocal reduction.
    llvm::Value* atan(llvm::Value* x){
        auto ax=intr(llvm::Intrinsic::fabs, x);
        auto gt=b.CreateFCmpOGT(ax, C(1.0f));
        auto inv=b.CreateFDiv(C(1.0f), ax);
        auto z=b.CreateSelect(gt, inv, ax);                    // z in [0,1]
        auto z2=b.CreateFMul(z,z);
        // minimax degree-9 odd poly for atan on [0,1]
        auto p=fma(C(0.0208351f), z2, C(-0.0851330f));
        p=fma(p, z2, C(0.1801410f)); p=fma(p, z2, C(-0.3302995f));
        p=fma(p, z2, C(0.9998660f)); p=b.CreateFMul(p, z);
        auto HALFPI=C(1.57079632679f);
        auto big=b.CreateFSub(HALFPI, p);
        auto r=b.CreateSelect(gt, big, p);
        // restore sign of x
        auto neg=b.CreateFCmpOLT(x, C(0.0f));
        return b.CreateSelect(neg, b.CreateFNeg(r), r);
    }
    llvm::Value* asin(llvm::Value* x){
        // asin(x)=atan(x/sqrt(1-x^2)); clamp domain
        auto one=C(1.0f);
        auto d=intr(llvm::Intrinsic::sqrt, b.CreateFSub(one, b.CreateFMul(x,x)));
        return atan(b.CreateFDiv(x, d));
    }
    llvm::Value* acos(llvm::Value* x){ return b.CreateFSub(C(1.57079632679f), asin(x)); }
    // atan2(y,x) via atan + quadrant fixup.
    llvm::Value* atan2(llvm::Value* y, llvm::Value* x){
        auto PI=C(3.14159265359f), z=C(0.0f);
        auto a=atan(b.CreateFDiv(y, x));
        auto xlt=b.CreateFCmpOLT(x, z);
        auto yge=b.CreateFCmpOGE(y, z);
        auto add=b.CreateSelect(yge, PI, b.CreateFNeg(PI));
        return b.CreateSelect(xlt, b.CreateFAdd(a, add), a);
    }
};
} // namespace

llvm::Value* CustomExprCompiler::gen_call_vec(const expr::Node& n) {
    auto& b = *b_;
    std::vector<llvm::Value*> a;
    for (const auto& c : n.children) {
        auto v = gen_vec(*c); if (!v) return nullptr;
        a.push_back(v);
    }
    auto* vt = vty(*ctx_, vw_);
    auto vintr = [&](llvm::Intrinsic::ID id) {
        auto* f = llvm::Intrinsic::getOrInsertDeclaration(mod_, id, {vt});
        return a.size() == 2 ? b.CreateCall(f, {a[0], a[1]})
                             : b.CreateCall(f, {a[0]});
    };
    const auto& nm = n.ident;
    VPoly vp{b, vt, vw_};
    if (nm == "sqrt")  return vintr(llvm::Intrinsic::sqrt);
    if (nm == "abs")   return vintr(llvm::Intrinsic::fabs);
    if (nm == "sin")   return vp.sin(a[0]);
    if (nm == "cos")   return vp.cos(a[0]);
    if (nm == "exp")   return vintr(llvm::Intrinsic::exp);
    if (nm == "log")   return vintr(llvm::Intrinsic::log);
    if (nm == "floor") return vintr(llvm::Intrinsic::floor);
    if (nm == "ceil")  return vintr(llvm::Intrinsic::ceil);
    if (nm == "pow")   return vintr(llvm::Intrinsic::pow);
    if (nm == "nth_root") {   // a^(1/b)
        auto* P = llvm::Intrinsic::getOrInsertDeclaration(mod_, llvm::Intrinsic::pow, {vt});
        auto* one = llvm::ConstantFP::get(vt, 1.0);
        return b.CreateCall(P, {a[0], b.CreateFDiv(one, a[1])});
    }
    if (nm == "min")   return vintr(llvm::Intrinsic::minnum);
    if (nm == "max")   return vintr(llvm::Intrinsic::maxnum);
    if (nm == "tan") { llvm::Value* s; llvm::Value* c; vp.sincos(a[0], &s, &c);
                       return b.CreateFDiv(s, c, "tan"); }
    if (nm == "asin")  return vp.asin(a[0]);
    if (nm == "acos")  return vp.acos(a[0]);
    if (nm == "atan")  return vp.atan(a[0]);
    if (nm == "atan2") return vp.atan2(a[0], a[1]);
    if (nm == "mod") {  // x - floor(x/y)*y  (matches fmod for y>0; SIMD-friendly)
        auto* ff = llvm::Intrinsic::getOrInsertDeclaration(mod_, llvm::Intrinsic::floor, {vt});
        auto flq = b.CreateCall(ff, {b.CreateFDiv(a[0], a[1])});
        return b.CreateFSub(a[0], b.CreateFMul(flq, a[1]));
    }
    // User template call: the SIMD path inlines the body with the (already
    // vector) argument values bound to the template's parameters. A fresh memo
    // scope keeps two calls of the same template with different args distinct.
    if (reg_) {
        if (const TemplateFn* t = reg_->find(nm)) {
            auto saved_params = vparams_;
            auto saved_memo   = std::move(vmemo_); vmemo_.clear();
            for (std::size_t i = 0; i < t->params.size(); ++i)
                vparams_[t->params[i]] = a[i];
            llvm::Value* r = gen_vec(*t->body);
            vparams_ = std::move(saved_params);
            vmemo_   = std::move(saved_memo);
            return r;
        }
    }
    fail("unknown function '" + nm + "'"); return nullptr;
}

llvm::Value* CustomExprCompiler::gen_vec_inline(
        llvm::Module& mod, llvm::LLVMContext& ctx, llvm::IRBuilder<>& b,
        const expr::NodePtr& ast, llvm::Value* x, llvm::Value* y, llvm::Value* z,
        unsigned width) {
    ctx_=&ctx; mod_=&mod; b_=&b; vw_=width; vmemo_.clear();
    vx_=x; vy_=y; vz_=z;
    auto r = gen_vec(*ast);
    return r ? r : llvm::ConstantFP::get(vty(ctx,width), 0.0);
}

llvm::Function* CustomExprCompiler::compile_vec(llvm::Module&        mod,
                                                llvm::LLVMContext&   ctx,
                                                const std::string&   fn_name,
                                                const expr::NodePtr& ast,
                                                unsigned             width) {
    error_.clear();
    if (!ast) { fail("null AST"); return nullptr; }
    if (mod.getFunction(fn_name)) { fail("fn exists: " + fn_name); return nullptr; }
    ctx_ = &ctx; mod_ = &mod; vw_ = width; vmemo_.clear();

    auto* pf  = llvm::PointerType::getUnqual(ctx);
    auto* fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                        {pf, pf, pf, pf}, false);
    auto* fn = llvm::Function::Create(fty, llvm::Function::ExternalLinkage,
                                      fn_name, mod);
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    auto it = fn->arg_begin();
    llvm::Value* pX = &*it++; llvm::Value* pY = &*it++;
    llvm::Value* pZ = &*it++; llvm::Value* pO = &*it++;

    auto* bb = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> b(bb); b_ = &b;
    auto* vt = vty(ctx, width);
    auto load = [&](llvm::Value* p) {
        return b.CreateAlignedLoad(vt, p, llvm::MaybeAlign(4));
    };
    vx_ = load(pX); vy_ = load(pY); vz_ = load(pZ);

    auto* r = gen_vec(*ast);
    if (!r || !error_.empty()) { fn->eraseFromParent(); return nullptr; }
    b.CreateAlignedStore(r, pO, llvm::MaybeAlign(4));
    b.CreateRetVoid();

    std::string vfy; llvm::raw_string_ostream es(vfy);
    if (llvm::verifyFunction(*fn, &es)) {
        fail("verify: " + vfy); fn->eraseFromParent(); return nullptr;
    }
    return fn;
}

llvm::Function* CustomExprCompiler::compile(llvm::Module&        mod,
                                            llvm::LLVMContext&   ctx,
                                            const std::string&   fn_name,
                                            const expr::NodePtr& ast) {
    error_.clear();
    if (!ast) { fail("null AST passed to compile()"); return nullptr; }

    ctx_ = &ctx;
    mod_ = &mod;
    memo_.clear();   // Values are per-function; don't reuse across compiles

    auto* fty = llvm::FunctionType::get(
        llvm::Type::getFloatTy(ctx),
        {llvm::Type::getFloatTy(ctx),
         llvm::Type::getFloatTy(ctx),
         llvm::Type::getFloatTy(ctx)},
        false);

    if (mod.getFunction(fn_name)) {
        fail("function " + fn_name + " already exists in module");
        return nullptr;
    }

    auto* fn = llvm::Function::Create(
        fty, llvm::Function::PrivateLinkage, fn_name, mod);
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto it = fn->arg_begin();
    vx_ = &*it++; vx_->setName("x");
    vy_ = &*it++; vy_->setName("y");
    vz_ = &*it++; vz_->setName("z");

    auto* bb = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> b(bb);
    b_ = &b;

    auto* result = gen(*ast);
    if (!result || !error_.empty()) {
        fn->eraseFromParent();
        return nullptr;
    }
    b.CreateRet(result);

    std::string vfy;
    llvm::raw_string_ostream es(vfy);
    if (llvm::verifyFunction(*fn, &es)) {
        fail("verify error: " + vfy);
        fn->eraseFromParent();
        return nullptr;
    }
    return fn;
}

// Convenience overload — parse then compile.
llvm::Function* CustomExprCompiler::compile(llvm::Module&       mod,
                                            llvm::LLVMContext&  ctx,
                                            const std::string&  fn_name,
                                            const std::string&  expr_src) {
    expr::NodePtr ast;
    try {
        ast = expr::parse(expr_src);
    } catch (const expr::ParseError& e) {
        fail(std::string("parse error: ") + e.what());
        return nullptr;
    }
    return compile(mod, ctx, fn_name, ast);
}

// ── User-defined template functions ──────────────────────────────────────────

llvm::Function* CustomExprCompiler::emit_template_fn(const TemplateFn& t) {
    std::string fn_name = "frep_tmpl_" + t.name;
    if (auto* ex = mod_->getFunction(fn_name)) return ex;   // already emitted

    // float frep_tmpl_<name>(float p0, ..., float pN, float x, float y, float z)
    std::vector<llvm::Type*> arg_tys(t.params.size() + 3, f32());
    auto* fty = llvm::FunctionType::get(f32(), arg_tys, false);
    auto* fn = llvm::Function::Create(fty, llvm::Function::PrivateLinkage,
                                      fn_name, *mod_);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    // Save the state of any in-flight emission and start fresh for this body.
    auto* saved_b   = b_;
    auto  saved_memo   = std::move(memo_);   memo_.clear();
    auto  saved_params = std::move(params_); params_.clear();
    auto* sx = vx_; auto* sy = vy_; auto* sz = vz_;

    auto* bb = llvm::BasicBlock::Create(*ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);
    b_ = &b;
    auto it = fn->arg_begin();
    for (const auto& p : t.params) { it->setName(p); params_[p] = &*it; ++it; }
    vx_ = &*it++; vx_->setName("x");
    vy_ = &*it++; vy_->setName("y");
    vz_ = &*it++; vz_->setName("z");

    llvm::Value* result = gen(*t.body);
    bool ok = result && error_.empty();
    if (ok) b.CreateRet(result);

    // Restore the outer emission state.
    memo_   = std::move(saved_memo);
    params_ = std::move(saved_params);
    vx_ = sx; vy_ = sy; vz_ = sz; b_ = saved_b;

    if (!ok) { fn->eraseFromParent(); return nullptr; }
    std::string vfy; llvm::raw_string_ostream es(vfy);
    if (llvm::verifyFunction(*fn, &es)) {
        fail("template '" + t.name + "' verify: " + vfy);
        fn->eraseFromParent();
        return nullptr;
    }
    return fn;
}

bool CustomExprCompiler::emit_templates(llvm::Module& mod, llvm::LLVMContext& ctx) {
    if (!reg_) return true;
    ctx_ = &ctx; mod_ = &mod;
    for (const auto& t : reg_->all())
        if (!emit_template_fn(t)) return false;
    return true;
}


// ═════════════════════════════════════════════════════════════════════════════
// Back-end #2: CPU evaluation (direct AST interpretation)
// ═════════════════════════════════════════════════════════════════════════════

float CustomExprNode::eval_ast(const expr::Node& n, float x, float y, float z) {
    return eval_ast(n, x, y, z, nullptr, nullptr);
}

float CustomExprNode::eval_ast(const expr::Node& n, float x, float y, float z,
                               const std::unordered_map<std::string, float>* params,
                               const TemplateRegistry* reg) {
    using Kind = expr::Node::Kind;
    switch (n.kind) {
        case Kind::Number: return n.num;
        case Kind::Var:
            if (n.ident == "x") return x;
            if (n.ident == "y") return y;
            if (n.ident == "z") return z;
            if (params) {
                auto it = params->find(n.ident);
                if (it != params->end()) return it->second;
            }
            throw std::runtime_error("unknown variable '" + n.ident + "'");
        case Kind::Const:
            if (n.ident == "pi") return std::numbers::pi_v<float>;
            if (n.ident == "e")  return std::numbers::e_v<float>;
            throw std::runtime_error("unknown constant '" + n.ident + "'");
        case Kind::UnaryNeg:
            return -eval_ast(*n.children[0], x, y, z, params, reg);
        case Kind::BinOp: {
            float l = eval_ast(*n.children[0], x, y, z, params, reg);
            float r = eval_ast(*n.children[1], x, y, z, params, reg);
            switch (n.bop) {
                case expr::Op::Add: return l + r;
                case expr::Op::Sub: return l - r;
                case expr::Op::Mul: return l * r;
                case expr::Op::Div: return l / r;
            }
            throw std::runtime_error("unhandled BinOp");
        }
        case Kind::Call: {
            const auto& name = n.ident;
            // User template call: bind evaluated args to the template's params,
            // then evaluate its body at the ambient (x,y,z). Checked before the
            // builtins (whose names never appear in the registry).
            if (reg) {
                if (const TemplateFn* t = reg->find(name)) {
                    std::unordered_map<std::string, float> np;
                    for (std::size_t i = 0; i < t->params.size(); ++i)
                        np[t->params[i]] =
                            eval_ast(*n.children[i], x, y, z, params, reg);
                    return eval_ast(*t->body, x, y, z, &np, reg);
                }
            }
            // Builtins.
            float a0 = eval_ast(*n.children[0], x, y, z, params, reg);
            if (name == "sin")   return std::sin(a0);
            if (name == "cos")   return std::cos(a0);
            if (name == "tan")   return std::tan(a0);
            if (name == "sqrt")  return std::sqrt(a0);
            if (name == "abs")   return std::abs(a0);
            if (name == "exp")   return std::exp(a0);
            if (name == "log")   return std::log(a0);
            if (name == "floor") return std::floor(a0);
            if (name == "asin")  return std::asin(a0);
            if (name == "acos")  return std::acos(a0);
            if (name == "atan")  return std::atan(a0);
            if (name == "ceil")  return std::ceil(a0);
            float a1 = eval_ast(*n.children[1], x, y, z, params, reg);
            if (name == "pow")   return std::pow(a0, a1);
            if (name == "nth_root") return std::pow(a0, 1.0f / a1);
            if (name == "min")   return std::fmin(a0, a1);
            if (name == "max")   return std::fmax(a0, a1);
            if (name == "atan2") return std::atan2(a0, a1);
            if (name == "mod")   return std::fmod(a0, a1);
            throw std::runtime_error("unknown function '" + name + "'");
        }
    }
    throw std::runtime_error("unhandled AST kind in eval");
}


// ═════════════════════════════════════════════════════════════════════════════
// Back-end #3: GLSL emission
// ═════════════════════════════════════════════════════════════════════════════

void CustomExprNode::emit_glsl_ast(std::ostream& out, const expr::Node& n) {
    emit_glsl_ast(out, n, nullptr);
}

void CustomExprNode::emit_templates_glsl(std::ostream& out,
                                         const TemplateRegistry& reg) {
    for (const auto& t : reg.all()) {
        out << "float frep_tmpl_" << t.name << "(";
        for (const auto& p : t.params) out << "float " << p << ", ";
        out << "float x, float y, float z) {\n    return ";
        emit_glsl_ast(out, *t.body, &reg);
        out << ";\n}\n";
    }
}

void CustomExprNode::emit_glsl_ast(std::ostream& out, const expr::Node& n,
                                   const TemplateRegistry* reg) {
    using Kind = expr::Node::Kind;
    switch (n.kind) {
        case Kind::Number: {
            // GLSL prefers explicit float literal syntax — `1` would be
            // an int, which in float context auto-promotes but reads
            // strangely. Emit with one decimal place at minimum.
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.7g", n.num);
            std::string s(buf);
            if (s.find('.') == std::string::npos &&
                s.find('e') == std::string::npos)
                s += ".0";
            out << s;
            return;
        }
        case Kind::Var:
            out << n.ident;  // x, y, z are in scope at the call site
            return;
        case Kind::Const:
            // GLSL has no built-in pi/e; emit numeric value.
            if (n.ident == "pi") out << "3.14159265358979";
            else if (n.ident == "e") out << "2.71828182845905";
            else out << "0.0";
            return;
        case Kind::UnaryNeg:
            out << "(-";
            emit_glsl_ast(out, *n.children[0], reg);
            out << ")";
            return;
        case Kind::BinOp: {
            const char* op = "+";
            switch (n.bop) {
                case expr::Op::Add: op = "+"; break;
                case expr::Op::Sub: op = "-"; break;
                case expr::Op::Mul: op = "*"; break;
                case expr::Op::Div: op = "/"; break;
            }
            // Parenthesize liberally — GLSL parser still folds these
            // out, and we don't need to track precedence carefully.
            out << "(";
            emit_glsl_ast(out, *n.children[0], reg);
            out << " " << op << " ";
            emit_glsl_ast(out, *n.children[1], reg);
            out << ")";
            return;
        }
        case Kind::Call: {
            // User template call -> frep_tmpl_<name>(args..., x, y, z).
            if (reg && reg->find(n.ident)) {
                out << "frep_tmpl_" << n.ident << "(";
                for (const auto& c : n.children) {
                    emit_glsl_ast(out, *c, reg);
                    out << ", ";
                }
                out << "x, y, z)";
                return;
            }
            // GLSL has no nth_root; expand it to the pow() it is defined as.
            if (n.ident == std::string("nth_root")) {
                out << "pow(";
                emit_glsl_ast(out, *n.children[0], reg);
                out << ", 1.0 / (";
                emit_glsl_ast(out, *n.children[1], reg);
                out << "))";
                return;
            }
            // All supported functions have matching names in GLSL.
            // `abs` works for floats; `pow/min/max` accept two scalars.
            // GLSL dialect: atan2(y,x) -> atan(y,x); mod() is GLSL mod.
            out << (n.ident == std::string("atan2") ? "atan" : n.ident.c_str()) << "(";
            for (std::size_t i = 0; i < n.children.size(); ++i) {
                if (i) out << ", ";
                emit_glsl_ast(out, *n.children[i], reg);
            }
            out << ")";
            return;
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// CustomExprNode::codegen — wires up the cached fn lookup + LLVM compile.
// ═════════════════════════════════════════════════════════════════════════════
llvm::Value* CustomExprNode::codegen(CgCtx& c, llvm::Value* x,
                                     llvm::Value* y, llvm::Value* z) const {
    ensure_parsed();

    if (c.width > 1) {  // SIMD broadcast: inline the vector AST (no scalar helper)
        CustomExprCompiler comp;
        comp.set_templates(reg_);   // template calls inline into the vector AST
        return comp.gen_vec_inline(c.mod, c.lc, c.b, ast_, x, y, z, c.width);
    }

    if (cached_fn_name_.empty())
        cached_fn_name_ = "frep_expr_" + std::to_string(structural_hash());

    auto* fn = c.mod.getFunction(cached_fn_name_);
    if (!fn) {
        CustomExprCompiler comp;
        comp.set_templates(reg_);
        auto saved = c.b.saveIP();
        // Emit the (shared, callable) template functions into this module once;
        // idempotent, so multiple CustomExpr nodes sharing a registry are fine.
        if (!comp.emit_templates(c.mod, c.lc)) {
            c.b.restoreIP(saved);
            llvm::errs() << "CustomExprNode templates: " << comp.last_error()
                         << " (expression: " << expr_ << ")\n";
            return c.fc(0.0f);
        }
        fn = comp.compile(c.mod, c.lc, cached_fn_name_, ast_);
        c.b.restoreIP(saved);
        if (!fn) {
            llvm::errs() << "CustomExprNode: " << comp.last_error()
                         << " (expression: " << expr_ << ")\n";
            return c.fc(0.0f);
        }
    }
    return c.b.CreateCall(fn, {x, y, z}, "expr_v");
}

// ═════════════════════════════════════════════════════════════════════════════
// CustomExprNode::codegen_grad — forward-mode AD in LLVM IR over the AST.
//
// One pass over the expression yields the value and the directional derivative
// along the seed (x.dot, y.dot, z.dot); the tracer seeds the three axes, so a
// normal is three passes instead of the finite-difference default's six full
// field evaluations (plus the value). Every builtin of expr_ast.hpp has a rule;
// a user template, or a libm function in the SIMD path (no vector form), makes
// the whole node fall back to the default so correctness never depends on
// coverage. Values follow the scalar emitter (raw IEEE pow, minnum-style
// min/max as a select); only the derivative is new.
// ═════════════════════════════════════════════════════════════════════════════
FRepNode::DualVal CustomExprNode::codegen_grad(CgCtx& c, DualVal x, DualVal y,
                                               DualVal z) const {
    ensure_parsed();
    // FREP_CUSTOMEXPR_FD=1 keeps the finite-difference default, for an A/B of
    // the analytic gradient against it (speed and parity).
    static const bool force_fd = [] {
        const char* s = std::getenv("FREP_CUSTOMEXPR_FD");
        return s && s[0] == '1';
    }();
    if (force_fd) return FRepNode::codegen_grad(c, x, y, z);
    // TracerConfig::normal_method, carried in the context.
    if (c.grad_method == GradMethod::FiniteDiff) return FRepNode::codegen_grad(c, x, y, z);
    if (c.grad_method == GradMethod::Adir && c.width == 1) {
        if (auto r = codegen_grad_adir(c, x, y, z)) return *r;
        // ADIR not built in, or it refused the function: the analytic form.
    }
    auto& b = c.b;
    llvm::Type* ty = x.val->getType();               // float or <W x float>
    auto cst = [&](double v) -> llvm::Value* { return llvm::ConstantFP::get(ty, v); };
    using Kind = expr::Node::Kind;
    bool ok = true;
    std::unordered_map<const expr::Node*, DualVal> memo;

    std::function<bool(const expr::Node&)> varies = [&](const expr::Node& n) -> bool {
        if (n.kind == Kind::Var) return true;
        for (const auto& ch : n.children) if (varies(*ch)) return true;
        return false;
    };
    auto un = [&](llvm::Intrinsic::ID id, llvm::Value* v) {
        return frep::llvm_compat::unary_intrinsic(b, id, v);
    };
    auto libm1 = [&](const char* fn, llvm::Value* a) -> llvm::Value* {
        auto* fty = llvm::FunctionType::get(b.getFloatTy(), {b.getFloatTy()}, false);
        return b.CreateCall(c.mod.getOrInsertFunction(fn, fty), {a});
    };
    auto libm2 = [&](const char* fn, llvm::Value* a0, llvm::Value* a1) -> llvm::Value* {
        auto* fty = llvm::FunctionType::get(b.getFloatTy(), {b.getFloatTy(), b.getFloatTy()}, false);
        return b.CreateCall(c.mod.getOrInsertFunction(fn, fty), {a0, a1});
    };

    std::function<DualVal(const expr::Node&)> gen = [&](const expr::Node& n) -> DualVal {
        DualVal r{cst(0.0), cst(0.0)};
        if (!ok) return r;
        if (auto it = memo.find(&n); it != memo.end()) return it->second;
        switch (n.kind) {
            case Kind::Number: r = {cst(n.num), cst(0.0)}; break;
            case Kind::Var:
                if      (n.ident == "x") r = x;
                else if (n.ident == "y") r = y;
                else if (n.ident == "z") r = z;
                else ok = false;                      // a template's parameter
                break;
            case Kind::Const:
                r = {cst(n.ident == "pi" ? std::numbers::pi
                       : n.ident == "e"  ? std::numbers::e : 0.0), cst(0.0)};
                break;
            case Kind::UnaryNeg: {
                auto a = gen(*n.children[0]);
                r = {b.CreateFNeg(a.val), b.CreateFNeg(a.dot)};
                break;
            }
            case Kind::BinOp: {
                auto l = gen(*n.children[0]);
                auto rr = gen(*n.children[1]);
                switch (n.bop) {
                    case expr::Op::Add:
                        r = {b.CreateFAdd(l.val, rr.val), b.CreateFAdd(l.dot, rr.dot)}; break;
                    case expr::Op::Sub:
                        r = {b.CreateFSub(l.val, rr.val), b.CreateFSub(l.dot, rr.dot)}; break;
                    case expr::Op::Mul:
                        r = {b.CreateFMul(l.val, rr.val),
                             b.CreateFAdd(b.CreateFMul(l.val, rr.dot), b.CreateFMul(rr.val, l.dot))};
                        break;
                    case expr::Op::Div: {
                        auto* q = b.CreateFDiv(l.val, rr.val);
                        // (l' - q r') / r
                        r = {q, b.CreateFDiv(b.CreateFSub(l.dot, b.CreateFMul(q, rr.dot)), rr.val)};
                        break;
                    }
                }
                break;
            }
            case Kind::Call: {
                const auto& nm = n.ident;
                if (reg_ && reg_->find(nm)) { ok = false; break; }
                std::vector<DualVal> a;
                for (const auto& ch : n.children) a.push_back(gen(*ch));
                if (!ok || a.empty()) { ok = false; break; }
                const auto& a0 = a[0];
                if (nm == "sqrt") {
                    auto* s  = un(llvm::Intrinsic::sqrt, a0.val);
                    auto* sg = un(llvm::Intrinsic::sqrt,
                                  frep::llvm_compat::max_num(b, a0.val, cst(1e-12)));
                    r = {s, b.CreateFDiv(b.CreateFMul(a0.dot, cst(0.5)), sg)};
                } else if (nm == "abs") {
                    auto* neg = b.CreateFCmpOLT(a0.val, cst(0.0));
                    auto* sgn = b.CreateSelect(neg, cst(-1.0), cst(1.0));
                    r = {un(llvm::Intrinsic::fabs, a0.val), b.CreateFMul(a0.dot, sgn)};
                } else if (nm == "sin") {
                    r = {un(llvm::Intrinsic::sin, a0.val),
                         b.CreateFMul(un(llvm::Intrinsic::cos, a0.val), a0.dot)};
                } else if (nm == "cos") {
                    r = {un(llvm::Intrinsic::cos, a0.val),
                         b.CreateFNeg(b.CreateFMul(un(llvm::Intrinsic::sin, a0.val), a0.dot))};
                } else if (nm == "tan") {
                    auto* t = b.CreateFDiv(un(llvm::Intrinsic::sin, a0.val),
                                           un(llvm::Intrinsic::cos, a0.val));
                    r = {t, b.CreateFMul(b.CreateFAdd(cst(1.0), b.CreateFMul(t, t)), a0.dot)};
                } else if (nm == "exp") {
                    auto* e = un(llvm::Intrinsic::exp, a0.val);
                    r = {e, b.CreateFMul(e, a0.dot)};
                } else if (nm == "log") {
                    r = {un(llvm::Intrinsic::log, a0.val), b.CreateFDiv(a0.dot, a0.val)};
                } else if (nm == "floor") {
                    r = {un(llvm::Intrinsic::floor, a0.val), cst(0.0)};
                } else if (nm == "ceil") {
                    r = {un(llvm::Intrinsic::ceil, a0.val), cst(0.0)};
                } else if ((nm == "min" || nm == "max") && a.size() == 2) {
                    auto* pick = nm == "min" ? b.CreateFCmpOLT(a0.val, a[1].val)
                                             : b.CreateFCmpOGT(a0.val, a[1].val);
                    r = {b.CreateSelect(pick, a0.val, a[1].val),
                         b.CreateSelect(pick, a0.dot, a[1].dot)};
                } else if ((nm == "pow" || nm == "nth_root") && a.size() == 2) {
                    DualVal e = a[1];
                    const bool const_e = !varies(*n.children[1]);
                    if (nm == "nth_root") {           // a^(1/b)
                        auto* inv = b.CreateFDiv(cst(1.0), e.val);
                        e = {inv, b.CreateFNeg(b.CreateFDiv(e.dot, b.CreateFMul(e.val, e.val)))};
                    }
                    auto* P = llvm::Intrinsic::getOrInsertDeclaration(&c.mod, llvm::Intrinsic::pow, {ty});
                    auto* p   = b.CreateCall(P, {a0.val, e.val});
                    auto* pm1 = b.CreateCall(P, {a0.val, b.CreateFSub(e.val, cst(1.0))});
                    auto* d = b.CreateFMul(b.CreateFMul(e.val, pm1), a0.dot);   // e a^(e-1) a'
                    if (!const_e) {                   // + a^e ln(a) e', for a > 0
                        auto* pos = b.CreateFCmpOGT(a0.val, cst(0.0));
                        auto* lg  = un(llvm::Intrinsic::log, b.CreateSelect(pos, a0.val, cst(1.0)));
                        d = b.CreateFAdd(d, b.CreateSelect(pos, b.CreateFMul(b.CreateFMul(p, lg), e.dot), cst(0.0)));
                    }
                    r = {p, d};
                } else if (nm == "asin" || nm == "acos" || nm == "atan" || nm == "atan2" || nm == "mod") {
                    if (c.width > 1) { ok = false; break; }      // libm has no vector form
                    if (nm == "asin" || nm == "acos") {
                        auto* one_m = b.CreateFSub(cst(1.0), b.CreateFMul(a0.val, a0.val));
                        auto* den = un(llvm::Intrinsic::sqrt, frep::llvm_compat::max_num(b, one_m, cst(1e-12)));
                        auto* d = b.CreateFDiv(a0.dot, den);
                        r = {libm1(nm == "asin" ? "asinf" : "acosf", a0.val),
                             nm == "asin" ? d : b.CreateFNeg(d)};
                    } else if (nm == "atan") {
                        r = {libm1("atanf", a0.val),
                             b.CreateFDiv(a0.dot, b.CreateFAdd(cst(1.0), b.CreateFMul(a0.val, a0.val)))};
                    } else if (nm == "atan2" && a.size() == 2) {
                        // d atan2(y, x) = (x y' - y x') / (x² + y²)
                        const auto& yy = a0; const auto& xx = a[1];
                        auto* den = frep::llvm_compat::max_num(
                            b, b.CreateFAdd(b.CreateFMul(xx.val, xx.val), b.CreateFMul(yy.val, yy.val)), cst(1e-12));
                        r = {libm2("atan2f", yy.val, xx.val),
                             b.CreateFDiv(b.CreateFSub(b.CreateFMul(xx.val, yy.dot), b.CreateFMul(yy.val, xx.dot)), den)};
                    } else if (nm == "mod" && a.size() == 2) {
                        // fmod(a, b) = a - trunc(a/b) b  →  a' - trunc(a/b) b'
                        auto* q = un(llvm::Intrinsic::trunc, b.CreateFDiv(a0.val, a[1].val));
                        r = {libm2("fmodf", a0.val, a[1].val),
                             b.CreateFSub(a0.dot, b.CreateFMul(q, a[1].dot))};
                    } else ok = false;
                } else {
                    ok = false;
                }
                break;
            }
        }
        if (ok) memo[&n] = r;
        return r;
    };

    DualVal out = gen(*ast_);
    if (!ok) return FRepNode::codegen_grad(c, x, y, z);   // the universal default
    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// CustomExprNode::codegen_grad_adir — the ADIR library's reverse-mode gradient
// of the expression's compiled scalar function.
//
// ADIR differentiates frep_expr_<hash>(x, y, z) in the module and adds
// frep_expr_<hash>_grad returning {∂x, ∂y, ∂z}; the directional derivative is
// that gradient dotted with the seed. The tracer seeds the three axes with
// identical arguments, and the gradient function is pure, so O3 folds the three
// calls into one. Compared with the node's own forward-mode codegen this is an
// external, general-purpose derivative of the *compiled* code rather than of the
// AST — the comparison TracerConfig::NormalMethod exists for.
// ═════════════════════════════════════════════════════════════════════════════
std::optional<FRepNode::DualVal>
CustomExprNode::codegen_grad_adir(CgCtx& c, DualVal x, DualVal y, DualVal z) const {
#ifndef FREP_HAS_ADIR
    (void)c; (void)x; (void)y; (void)z;
    return std::nullopt;
#else
    // The scalar function first: codegen() materialises frep_expr_<hash> in the
    // module (templates included) and returns a call to it — the value.
    llvm::Value* val = codegen(c, x.val, y.val, z.val);
    llvm::Function* f = c.mod.getFunction(cached_fn_name_);
    if (!f || f->isDeclaration()) return std::nullopt;

    const std::string gname = cached_fn_name_ + "_grad";
    llvm::Function* g = c.mod.getFunction(gname);
    if (!g) {
        auto saved = c.b.saveIP();
        adir::GradientOptions opts;
        opts.activeArgs   = {0u, 1u, 2u};
        opts.verifyResult = false;
        auto res = adir::reverseGradient(c.mod, f, opts);
        c.b.restoreIP(saved);
        if (!res) {
            static bool warned = false;
            if (!warned) {
                llvm::errs() << "ADIR: " << res.error().message
                             << " (expression: " << expr_ << "); using the analytic gradient\n";
                warned = true;
            }
            return std::nullopt;
        }
        g = *res;
        if (g->getName() != gname) g->setName(gname);
    }
    if (!g->getReturnType()->isStructTy() || g->getReturnType()->getStructNumElements() != 3)
        return std::nullopt;

    llvm::Value* gv = c.b.CreateCall(g, {x.val, y.val, z.val}, "expr_grad");
    llvm::Value* gx = c.b.CreateExtractValue(gv, 0);
    llvm::Value* gy = c.b.CreateExtractValue(gv, 1);
    llvm::Value* gz = c.b.CreateExtractValue(gv, 2);
    llvm::Value* dot = c.b.CreateFAdd(
        c.b.CreateFAdd(c.b.CreateFMul(gx, x.dot), c.b.CreateFMul(gy, y.dot)),
        c.b.CreateFMul(gz, z.dot));
    return DualVal{val, dot};
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
// Dual-number GLSL of the expression (see emit_glsl_dual in the header).
// ═════════════════════════════════════════════════════════════════════════════
bool CustomExprNode::emit_glsl_dual_ast(std::ostream& out, const expr::Node& n,
                                        const TemplateRegistry* reg,
                                        const std::string& dx, const std::string& dy,
                                        const std::string& dz) {
    using Kind = expr::Node::Kind;
    auto lit = [](float v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.9g", v);
        std::string s(buf);
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
        return s;
    };
    switch (n.kind) {
        case Kind::Number:
            out << "d_const(" << lit(n.num) << ")";
            return true;
        case Kind::Var:
            if (n.ident == "x") { out << dx; return true; }
            if (n.ident == "y") { out << dy; return true; }
            if (n.ident == "z") { out << dz; return true; }
            return false;                              // a template's parameter
        case Kind::Const:
            out << "d_const(" << (n.ident == "pi" ? "3.14159265358979"
                                : n.ident == "e"  ? "2.71828182845905" : "0.0") << ")";
            return true;
        case Kind::UnaryNeg:
            out << "d_neg(";
            if (!emit_glsl_dual_ast(out, *n.children[0], reg, dx, dy, dz)) return false;
            out << ")";
            return true;
        case Kind::BinOp: {
            const char* fn = "d_add";
            switch (n.bop) {
                case expr::Op::Add: fn = "d_add"; break;
                case expr::Op::Sub: fn = "d_sub"; break;
                case expr::Op::Mul: fn = "d_mul"; break;
                case expr::Op::Div: fn = "d_div"; break;
            }
            out << fn << "(";
            if (!emit_glsl_dual_ast(out, *n.children[0], reg, dx, dy, dz)) return false;
            out << ", ";
            if (!emit_glsl_dual_ast(out, *n.children[1], reg, dx, dy, dz)) return false;
            out << ")";
            return true;
        }
        case Kind::Call: {
            if (reg && reg->find(n.ident)) return false;
            static const char* const kUnary[]  = {"sin", "cos", "tan", "sqrt", "abs", "exp",
                                                  "log", "floor", "ceil", "atan", "asin", "acos"};
            static const char* const kBinary[] = {"pow", "nth_root", "atan2", "mod", "min", "max"};
            bool known = false;
            for (auto* u : kUnary)  if (n.ident == u && n.children.size() == 1) known = true;
            for (auto* bn : kBinary) if (n.ident == bn && n.children.size() == 2) known = true;
            if (!known) return false;
            out << "d_" << n.ident << "(";
            for (std::size_t i = 0; i < n.children.size(); ++i) {
                if (i) out << ", ";
                if (!emit_glsl_dual_ast(out, *n.children[i], reg, dx, dy, dz)) return false;
            }
            out << ")";
            return true;
        }
    }
    return false;
}

// ═════════════════════════════════════════════════════════════════════════════
// Bounds by interval arithmetic over the AST (see aabb() in the header).
// ═════════════════════════════════════════════════════════════════════════════
namespace {

struct Iv { float lo, hi; };
constexpr float kIvBig = 1e30f;   // "unknown": never lets a cell be discarded
inline Iv iv_big() { return {-kIvBig, kIvBig}; }

// sin/cos over an interval: the endpoints, widened to ±1 where an extremum
// of the function lies inside the interval.
Iv iv_trig(bool is_sin, const Iv& a) {
    constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
    if (a.hi - a.lo >= two_pi) return {-1.0f, 1.0f};
    float u = is_sin ? std::sin(a.lo) : std::cos(a.lo);
    float v = is_sin ? std::sin(a.hi) : std::cos(a.hi);
    Iv r{std::min(u, v), std::max(u, v)};
    auto contains = [&](float base) {   // is base + 2πk in [lo, hi] for some k?
        float k = std::ceil((a.lo - base) / two_pi);
        return base + k * two_pi <= a.hi;
    };
    const float half_pi = 0.5f * std::numbers::pi_v<float>;
    if (contains(is_sin ? half_pi : 0.0f))                       r.hi = 1.0f;
    if (contains(is_sin ? -half_pi : std::numbers::pi_v<float>)) r.lo = -1.0f;
    return r;
}

std::optional<Iv> ival(const expr::Node& n, const Iv& X, const Iv& Y, const Iv& Z,
                       const TemplateRegistry* reg) {
    using Kind = expr::Node::Kind;
    auto mono = [](float (*f)(float), const Iv& a) {
        float u = f(a.lo), v = f(a.hi);
        return Iv{std::min(u, v), std::max(u, v)};
    };
    switch (n.kind) {
        case Kind::Number: return Iv{n.num, n.num};
        case Kind::Var:
            if (n.ident == "x") return X;
            if (n.ident == "y") return Y;
            if (n.ident == "z") return Z;
            return std::nullopt;
        case Kind::Const: {
            float v = n.ident == "pi" ? std::numbers::pi_v<float>
                    : n.ident == "e"  ? std::numbers::e_v<float> : 0.0f;
            return Iv{v, v};
        }
        case Kind::UnaryNeg: {
            auto a = ival(*n.children[0], X, Y, Z, reg);
            if (!a) return std::nullopt;
            return Iv{-a->hi, -a->lo};
        }
        case Kind::BinOp: {
            auto l = ival(*n.children[0], X, Y, Z, reg);
            auto r = ival(*n.children[1], X, Y, Z, reg);
            if (!l || !r) return std::nullopt;
            switch (n.bop) {
                case expr::Op::Add: return Iv{l->lo + r->lo, l->hi + r->hi};
                case expr::Op::Sub: return Iv{l->lo - r->hi, l->hi - r->lo};
                case expr::Op::Mul: {
                    float p[4] = {l->lo * r->lo, l->lo * r->hi, l->hi * r->lo, l->hi * r->hi};
                    return Iv{std::min({p[0], p[1], p[2], p[3]}), std::max({p[0], p[1], p[2], p[3]})};
                }
                case expr::Op::Div: {
                    if (r->lo <= 0.0f && r->hi >= 0.0f) return iv_big();
                    Iv inv{1.0f / r->hi, 1.0f / r->lo};
                    float p[4] = {l->lo * inv.lo, l->lo * inv.hi, l->hi * inv.lo, l->hi * inv.hi};
                    return Iv{std::min({p[0], p[1], p[2], p[3]}), std::max({p[0], p[1], p[2], p[3]})};
                }
            }
            return iv_big();
        }
        case Kind::Call: {
            const auto& nm = n.ident;
            if (reg && reg->find(nm)) return std::nullopt;
            std::vector<Iv> a;
            for (const auto& ch : n.children) {
                auto v = ival(*ch, X, Y, Z, reg);
                if (!v) return std::nullopt;
                a.push_back(*v);
            }
            if (a.empty()) return std::nullopt;
            const Iv& a0 = a[0];
            if (nm == "sqrt")  return Iv{std::sqrt(std::max(a0.lo, 0.0f)), std::sqrt(std::max(a0.hi, 0.0f))};
            if (nm == "abs") {
                if (a0.lo >= 0.0f) return a0;
                if (a0.hi <= 0.0f) return Iv{-a0.hi, -a0.lo};
                return Iv{0.0f, std::max(-a0.lo, a0.hi)};
            }
            if (nm == "sin")   return iv_trig(true, a0);
            if (nm == "cos")   return iv_trig(false, a0);
            if (nm == "tan")   return iv_big();
            if (nm == "exp")   return mono([](float v) { return std::exp(v); }, a0);
            if (nm == "log") {
                if (a0.hi <= 0.0f) return iv_big();
                return Iv{a0.lo > 0.0f ? std::log(a0.lo) : -kIvBig, std::log(a0.hi)};
            }
            if (nm == "floor") return mono([](float v) { return std::floor(v); }, a0);
            if (nm == "ceil")  return mono([](float v) { return std::ceil(v); }, a0);
            if (nm == "atan")  return mono([](float v) { return std::atan(v); }, a0);
            if (nm == "asin" || nm == "acos") {
                Iv c{std::clamp(a0.lo, -1.0f, 1.0f), std::clamp(a0.hi, -1.0f, 1.0f)};
                return nm == "asin" ? mono([](float v) { return std::asin(v); }, c)
                                    : mono([](float v) { return std::acos(v); }, c);
            }
            if (a.size() != 2) return iv_big();
            const Iv& a1 = a[1];
            if (nm == "min") return Iv{std::min(a0.lo, a1.lo), std::min(a0.hi, a1.hi)};
            if (nm == "max") return Iv{std::max(a0.lo, a1.lo), std::max(a0.hi, a1.hi)};
            if (nm == "atan2") return Iv{-std::numbers::pi_v<float>, std::numbers::pi_v<float>};
            if (nm == "mod") {
                if (a1.lo == a1.hi && a1.lo > 0.0f) return Iv{-a1.lo, a1.lo};
                return iv_big();
            }
            if (nm == "pow" || nm == "nth_root") {
                if (a1.lo != a1.hi) return iv_big();          // only a constant exponent
                float e = nm == "pow" ? a1.lo : (a1.lo != 0.0f ? 1.0f / a1.lo : 0.0f);
                if (a0.lo >= 0.0f) {
                    float u = std::pow(a0.lo, e), v = std::pow(a0.hi, e);
                    return Iv{std::min(u, v), std::max(u, v)};
                }
                const bool integer = std::floor(e) == e;
                if (integer && std::fmod(e, 2.0f) == 0.0f) {   // even power: symmetric
                    float m = std::max(std::fabs(a0.lo), std::fabs(a0.hi));
                    float lo = (a0.lo <= 0.0f && a0.hi >= 0.0f)
                             ? 0.0f : std::pow(std::min(std::fabs(a0.lo), std::fabs(a0.hi)), e);
                    return Iv{lo, std::pow(m, e)};
                }
                if (integer) {                                  // odd power: monotone
                    float u = std::pow(a0.lo, e), v = std::pow(a0.hi, e);
                    return Iv{std::min(u, v), std::max(u, v)};
                }
                return iv_big();                                // fractional power of a signed base
            }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

} // namespace

FRepNode::AABB CustomExprNode::infer_bounds() const {
    ensure_parsed();
    constexpr float R = 64.0f;      // search box: [-R, R]³
    constexpr int   kMaxDepth = 8;  // cells of 0.5 units at the surface
    const Iv root{-R, R};
    if (!ival(*ast_, root, root, root, reg_)) return AABB::infinite();

    float bx0 = kIvBig, by0 = kIvBig, bz0 = kIvBig, bx1 = -kIvBig, by1 = -kIvBig, bz1 = -kIvBig;
    bool any = false, edge = false;
    // Work bound. A cell that touches the search box makes the result infinite
    // whatever else is found, so the walk stops at the first one; and a field
    // whose surface fills the box (a floor slab under a 400-node expression
    // took four seconds here before these two cuts) stops at a cell budget.
    constexpr long kCellBudget = 200000;
    long cells = 0;
    std::function<void(float, float, float, float, int)> rec =
        [&](float x0, float y0, float z0, float s, int depth) {
            if (edge || cells > kCellBudget) return;
            ++cells;
            auto iv = ival(*ast_, {x0, x0 + s}, {y0, y0 + s}, {z0, z0 + s}, reg_);
            if (iv && iv->lo > 0.0f) return;                 // positive throughout: outside
            const bool inside = iv && iv->hi < 0.0f;         // negative throughout: no surface
            if (!inside && depth < kMaxDepth) {
                const float h = 0.5f * s;
                for (int i = 0; i < 8; ++i)
                    rec(x0 + ((i & 1) ? h : 0.0f), y0 + ((i & 2) ? h : 0.0f),
                        z0 + ((i & 4) ? h : 0.0f), h, depth + 1);
                return;
            }
            any = true;
            bx0 = std::min(bx0, x0); by0 = std::min(by0, y0); bz0 = std::min(bz0, z0);
            bx1 = std::max(bx1, x0 + s); by1 = std::max(by1, y0 + s); bz1 = std::max(bz1, z0 + s);
            if (x0 <= -R || y0 <= -R || z0 <= -R || x0 + s >= R || y0 + s >= R || z0 + s >= R)
                edge = true;                                   // the field reaches the box
        };
    rec(-R, -R, -R, 2.0f * R, 0);
    const bool gave_up = cells > kCellBudget;
    const AABB box = (!any || edge || gave_up) ? AABB::infinite()
                                              : AABB{bx0, by0, bz0, bx1, by1, bz1};
    if (std::getenv("FREP_DEBUG_BOUNDS")) {
        if (!any || edge || gave_up)
            std::fprintf(stderr, "custom-expr bounds: unbounded (%s, %ld cells) for %s\n",
                         gave_up ? "cell budget" : !any ? "no cell kept" : "reaches the search box",
                         cells, expr_.c_str());
        else
            std::fprintf(stderr, "custom-expr bounds: [%g %g %g]..[%g %g %g] for %s\n",
                         box.min_x, box.min_y, box.min_z, box.max_x, box.max_y, box.max_z,
                         expr_.c_str());
    }
    return box;
}

FRepNode::AABB CustomExprNode::aabb() const {
    if (has_bounds_) return bounds_;
    if (!inferred_) {
        inferred_box_ = infer_bounds();   // idempotent; a concurrent caller recomputes the same
        inferred_ = true;
    }
    return inferred_box_;
}

} // namespace frep
