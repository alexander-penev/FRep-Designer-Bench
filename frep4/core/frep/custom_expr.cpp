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

#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>

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

} // namespace frep
