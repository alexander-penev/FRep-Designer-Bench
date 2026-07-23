// core/compiler/codegen.cpp
//
// Generates a single LLVM IR module:
//   scene_sdf → scene_normal → shade_pixel → render_tile
// Everything is AlwaysInline → after O3, render_tile contains the inlined SDF directly.

#include "codegen.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/compiler/node_interval_ir.hpp"

#include "core/compiler/bvh.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

namespace frep {

// ─────────────────────────────────────────────────────────────────────────────
SceneCodegen::SceneCodegen(llvm::LLVMContext& ctx, TracerConfig cfg, std::string name)
    : ctx_(ctx), cfg_(cfg)
    , mod_(std::make_unique<llvm::Module>(name, ctx))
{}

// ─────────────────────────────────────────────────────────────────────────────
// Alloca in the function entry block (required by the mem2reg pass).
llvm::AllocaInst* SceneCodegen::entry_alloca(llvm::Function* fn, const std::string& name) {
    auto& entry = fn->getEntryBlock();
    llvm::IRBuilder<> tmp(&entry, entry.begin());
    return tmp.CreateAlloca(f32(), nullptr, name);
}

// ─────────────────────────────────────────────────────────────────────────────
// verify_fn — throws an exception on an invalid function.
static void verify_fn(llvm::Function* fn) {
    std::string err;
    llvm::raw_string_ostream es(err);
    if (llvm::verifyFunction(*fn, &es))
        throw std::runtime_error(std::string(fn->getName()) + ": " + err);
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot allocation: returns existing slot for (node_id, param_name), or
// allocates a new one. Records the default value the first time we see
// a key — the host uses this list to seed the runtime buffer.
int SceneCodegen::acquire_param_slot(const std::string& node_id,
                                     const std::string& param_name,
                                     float default_value,
                                     int param_class)
{
    // Consult the compile policy: a Constant placement returns -1, which
    // makes param_value() bake the constant instead of allocating a slot.
    if (policy_) {
        auto placement = policy_->decide(node_id, param_name,
                                         static_cast<ParamClass>(param_class));
        if (placement == ParamPlacement::Constant) return -1;
    }
    const std::string key = node_id + "::" + param_name;
    auto it = param_slot_by_key_.find(key);
    if (it != param_slot_by_key_.end()) return it->second;
    int slot = static_cast<int>(param_bindings_.size());
    param_slot_by_key_[key] = slot;
    param_bindings_.push_back({node_id, param_name, slot, default_value});
    return slot;
}

llvm::Value* SceneCodegen::acquire_instance_fn(const FRepNode* target,
                                               llvm::IRBuilder<>& caller,
                                               llvm::Value* x, llvm::Value* y,
                                               llvm::Value* z, llvm::Value* params)
{
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), fptr()}, false);

    auto it = instance_fn_by_target_.find(target);
    llvm::Function* fn = (it != instance_fn_by_target_.end()) ? it->second : nullptr;
    if (!fn) {
        // Create the shared function and emit the target's geometry into it once.
        fn = llvm::Function::Create(
            fty, llvm::Function::InternalLinkage,
            "inst_geom_" + std::to_string(instance_fn_by_target_.size()),
            mod_.get());
        fn->addFnAttr(llvm::Attribute::NoInline);   // keep it a real shared call
        fn->addFnAttr(llvm::Attribute::NoUnwind);
        fn->addFnAttr(llvm::Attribute::WillReturn);
        // Register before emitting the body so a self/cyclic reference resolves
        // to a call rather than recursing forever.
        instance_fn_by_target_[target] = fn;

        auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
        llvm::IRBuilder<> body(bb);
        auto a = fn->arg_begin();
        auto* ax = &*a++; auto* ay = &*a++; auto* az = &*a++; auto* apb = &*a++;
        // A fresh CgCtx bound to the shared function's own builder. It also gets
        // the instance_call callback so nested instances share too.
        CgCtx sub = make_cgctx(body, apb);
        body.CreateRet(target->codegen(sub, ax, ay, az));
        verify_fn(fn);
    }
    // Emit the call at the caller's site.
    return caller.CreateCall(fn, {x, y, z, params});
}

bool SceneCodegen::acquire_instance_grad_fn(const FRepNode* target,
                                            llvm::IRBuilder<>& caller,
                                            llvm::Value* xv, llvm::Value* xd,
                                            llvm::Value* yv, llvm::Value* yd,
                                            llvm::Value* zv, llvm::Value* zd,
                                            llvm::Value* params,
                                            llvm::Value*& out_val, llvm::Value*& out_dot)
{
    // Return a { value, derivative } pair as a literal struct.
    auto* ret_ty = llvm::StructType::get(ctx_, {f32(), f32()});
    auto* fty = llvm::FunctionType::get(ret_ty,
        {f32(), f32(), f32(), f32(), f32(), f32(), fptr()}, false);

    auto it = instance_grad_fn_by_target_.find(target);
    llvm::Function* fn = (it != instance_grad_fn_by_target_.end()) ? it->second : nullptr;
    if (!fn) {
        fn = llvm::Function::Create(
            fty, llvm::Function::InternalLinkage,
            "inst_grad_" + std::to_string(instance_grad_fn_by_target_.size()),
            mod_.get());
        fn->addFnAttr(llvm::Attribute::NoInline);
        fn->addFnAttr(llvm::Attribute::NoUnwind);
        fn->addFnAttr(llvm::Attribute::WillReturn);
        instance_grad_fn_by_target_[target] = fn;

        auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
        llvm::IRBuilder<> body(bb);
        auto a = fn->arg_begin();
        auto* axv=&*a++; auto* axd=&*a++; auto* ayv=&*a++; auto* ayd=&*a++;
        auto* azv=&*a++; auto* azd=&*a++; auto* apb=&*a++;
        CgCtx sub = make_cgctx(body, apb);
        FRepNode::DualVal dx{axv, axd}, dy{ayv, ayd}, dz{azv, azd};
        FRepNode::DualVal r = target->codegen_grad(sub, dx, dy, dz);
        llvm::Value* agg = llvm::UndefValue::get(ret_ty);
        agg = body.CreateInsertValue(agg, r.val, {0});
        agg = body.CreateInsertValue(agg, r.dot, {1});
        body.CreateRet(agg);
        verify_fn(fn);
    }
    auto* call = caller.CreateCall(fn, {xv, xd, yv, yd, zv, zd, params});
    out_val = caller.CreateExtractValue(call, {0});
    out_dot = caller.CreateExtractValue(call, {1});
    return true;
}

CgCtx SceneCodegen::make_cgctx(llvm::IRBuilder<>& b, llvm::Value* params_buffer) {
    CgCtx c{ctx_, *mod_, b};
    if (cfg_.incremental_params && params_buffer) {
        c.incremental_params = true;
        c.params_buffer      = params_buffer;
        // Bind the slot allocator. Capture `this` so the table is shared
        // across every CgCtx the SceneCodegen creates.
        c.slot_for_param = [this](const std::string& node_id,
                                  const std::string& param_name,
                                  float default_value,
                                  int param_class) {
            return this->acquire_param_slot(node_id, param_name,
                                            default_value, param_class);
        };
    }
    // Instancing Level 2: bind the shared-subprogram callback. Capture `this`
    // (for the memo table + emitter) and the params buffer so the shared body
    // can read runtime params; the call site uses the caller's builder `b`.
    if (cfg_.instance_shared_subprograms) {
        llvm::IRBuilder<>* caller = &b;
        llvm::Value* pbuf = params_buffer;
        c.instance_call = [this, caller, pbuf](const FRepNode* target,
                                               llvm::Value* x, llvm::Value* y,
                                               llvm::Value* z) -> llvm::Value* {
            return this->acquire_instance_fn(target, *caller, x, y, z, pbuf);
        };
        c.instance_grad_call = [this, caller, pbuf](
                const FRepNode* target,
                llvm::Value* xv, llvm::Value* xd, llvm::Value* yv, llvm::Value* yd,
                llvm::Value* zv, llvm::Value* zd,
                llvm::Value*& ov, llvm::Value*& od) -> bool {
            return this->acquire_instance_grad_fn(target, *caller,
                xv, xd, yv, yd, zv, zd, pbuf, ov, od);
        };
    }
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_scene_sdf
// float scene_sdf(float x, float y, float z)
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_sdf(const FRepNode& root) {
    // signature: (x, y, z, params)
    // params is a float* used in Incremental mode; in Constant mode it
    // is unused and DCE-ed by O3.
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), fptr()}, false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "scene_sdf", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr(llvm::Attribute::WillReturn);

    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);

    auto it = fn->arg_begin();
    auto* x  = &*it++; x->setName("x");
    auto* y  = &*it++; y->setName("y");
    auto* z  = &*it++; z->setName("z");
    auto* pb = &*it++; pb->setName("params");

    CgCtx cctx = make_cgctx(b, pb);
    auto* result = root.codegen(cctx, x, y, z);
    b.CreateRet(result);

    verify_fn(fn);
    return fn;
}

// Diagnostic / scalability variant — see header. Each object becomes a
// standalone non-inlined function obj_sdf_N(x,y,z,params)->float; scene_sdf
// calls them in sequence and folds with min(). The point is that LLVM
// optimises N small functions in ~linear total time, whereas inlining all
// N into one body makes a single huge function whose register allocation
// and scheduling cost grows super-linearly.
llvm::Function* SceneCodegen::emit_scene_sdf_split(
    const std::vector<FRepNode::Ptr>& objects)
{
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), fptr()}, false);

    // One non-inlined function per object.
    std::vector<llvm::Function*> obj_fns;
    obj_fns.reserve(objects.size());
    for (std::size_t i = 0; i < objects.size(); ++i) {
        auto* of = llvm::Function::Create(
            fty, llvm::Function::InternalLinkage,
            "obj_sdf_" + std::to_string(i), mod_.get());
        // Deliberately NOT AlwaysInline — keep them separate so the
        // optimiser never merges them back into one giant body. NoInline
        // makes that explicit and is what gives the linear compile cost.
        of->addFnAttr(llvm::Attribute::NoInline);
        of->addFnAttr(llvm::Attribute::NoUnwind);
        of->addFnAttr(llvm::Attribute::WillReturn);

        auto* bb = llvm::BasicBlock::Create(ctx_, "entry", of);
        llvm::IRBuilder<> b(bb);
        auto it = of->arg_begin();
        auto* x = &*it++; x->setName("x");
        auto* y = &*it++; y->setName("y");
        auto* z = &*it++; z->setName("z");
        auto* pb = &*it++; pb->setName("params");
        CgCtx cctx = make_cgctx(b, pb);
        b.CreateRet(objects[i]->codegen(cctx, x, y, z));
        verify_fn(of);
        obj_fns.push_back(of);
    }

    // scene_sdf: call each obj fn, fold with min().
    auto* fn = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "scene_sdf", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr(llvm::Attribute::WillReturn);
    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);
    auto it = fn->arg_begin();
    auto* x = &*it++; auto* y = &*it++; auto* z = &*it++; auto* pb = &*it++;

    llvm::Value* acc = nullptr;
    for (auto* of : obj_fns) {
        auto* d = b.CreateCall(of, {x, y, z, pb});
        acc = acc ? frep::llvm_compat::binary_intrinsic(
                        b, llvm::Intrinsic::minnum, acc, d)
                  : d;
    }
    if (!acc) acc = fc(b, 1e30f);   // empty scene → far away
    b.CreateRet(acc);
    verify_fn(fn);
    return fn;
}

// Build-time spatial guard variant (BVH approach 1 prototype). Each object
// becomes a non-inlined obj_sdf_N, and scene_sdf evaluates it only when the
// point is within the running best distance of the object's AABB:
//
//     best = +inf
//     d0 = aabb_distance(box_i, p);  if (d0 < best) best = min(best, obj_i(p));
//     ... repeated per object ...
//
// The AABB-distance test is cheap (a few sub/max/mul + one sqrt) and skips
// the (potentially expensive) object SDF when the point can't be nearer
// than what we already have. This is the flat prune from the BVH prototype
// materialised directly in the generated code — no hierarchy, no runtime
// stack, so it works on both the JIT and (later) GLSL paths. Objects are
// taken in the given order; the caller may pre-sort them spatially so
// nearby objects cluster and `best` tightens sooner.
//
// Correctness matches brute force for the same reason the BVH does: AABB
// distance is a lower bound on the object's exterior distance. Unlike the
// BVH we don't disable the guard for interior points — instead the guard
// uses `<` against `best` which, once `best` goes negative, is false for
// every non-negative AABB distance, so all remaining objects ARE still
// evaluated (no wrongful skip). That's the safe behaviour, just without
// the BVH's nearer-first ordering.
llvm::Function* SceneCodegen::emit_scene_sdf_guarded(
    const std::vector<FRepNode::Ptr>& objects)
{
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), fptr()}, false);

    // Per-object non-inlined functions (so the guard actually gates a
    // call rather than being optimised into the inlined body).
    std::vector<llvm::Function*> obj_fns;
    std::vector<FRepNode::AABB>  boxes;
    obj_fns.reserve(objects.size());
    boxes.reserve(objects.size());
    for (std::size_t i = 0; i < objects.size(); ++i) {
        auto* of = llvm::Function::Create(
            fty, llvm::Function::InternalLinkage,
            "obj_sdf_" + std::to_string(i), mod_.get());
        of->addFnAttr(llvm::Attribute::NoInline);
        of->addFnAttr(llvm::Attribute::NoUnwind);
        of->addFnAttr(llvm::Attribute::WillReturn);
        auto* bb = llvm::BasicBlock::Create(ctx_, "entry", of);
        llvm::IRBuilder<> bb_(bb);
        auto it = of->arg_begin();
        auto* x = &*it++; auto* y = &*it++; auto* z = &*it++; auto* pb = &*it++;
        CgCtx cctx = make_cgctx(bb_, pb);
        bb_.CreateRet(objects[i]->codegen(cctx, x, y, z));
        verify_fn(of);
        obj_fns.push_back(of);
        boxes.push_back(objects[i]->aabb());
    }

    auto* fn = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "scene_sdf", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr(llvm::Attribute::WillReturn);
    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(entry);
    auto it = fn->arg_begin();
    auto* x = &*it++; auto* y = &*it++; auto* z = &*it++; auto* pb = &*it++;

    // Mutable `best`, threaded through the per-object guarded blocks via a
    // stack slot (simplest correct SSA; mem2reg promotes it in O1+).
    auto* best = b.CreateAlloca(f32(), nullptr, "best");
    b.CreateStore(fc(b, 1e30f), best);

    auto aabb_dist = [&](const FRepNode::AABB& box) -> llvm::Value* {
        // max(min_i - p_i, 0, p_i - max_i) per axis, then length.
        auto axis = [&](llvm::Value* p, float lo, float hi) {
            auto* a = b.CreateFSub(fc(b, lo), p);
            auto* c = b.CreateFSub(p, fc(b, hi));
            auto* m = frep::llvm_compat::max_num(b, a,
                          frep::llvm_compat::max_num(b, c, fc(b, 0.0f)));
            return m;
        };
        auto* dx = axis(x, box.min_x, box.max_x);
        auto* dy = axis(y, box.min_y, box.max_y);
        auto* dz = axis(z, box.min_z, box.max_z);
        auto* s = b.CreateFAdd(b.CreateFMul(dx, dx),
                   b.CreateFAdd(b.CreateFMul(dy, dy), b.CreateFMul(dz, dz)));
        return frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, s);
    };

    for (std::size_t i = 0; i < obj_fns.size(); ++i) {
        // Unbounded objects (infinite AABB) are always evaluated — guard
        // would never skip them anyway, so emit them unconditionally to
        // avoid a pointless sqrt+branch.
        const auto& box = boxes[i];
        const bool infinite =
            box.min_x < -1e8f || box.max_x > 1e8f ||
            box.min_y < -1e8f || box.max_y > 1e8f ||
            box.min_z < -1e8f || box.max_z > 1e8f;

        if (infinite) {
            auto* cur = b.CreateLoad(f32(), best);
            auto* d   = b.CreateCall(obj_fns[i], {x, y, z, pb});
            b.CreateStore(frep::llvm_compat::binary_intrinsic(
                b, llvm::Intrinsic::minnum, cur, d), best);
            continue;
        }

        auto* cur  = b.CreateLoad(f32(), best);
        auto* dbox = aabb_dist(box);
        auto* take = b.CreateFCmpOLT(dbox, cur, "guard");
        auto* then_bb = llvm::BasicBlock::Create(ctx_, "eval", fn);
        auto* cont_bb = llvm::BasicBlock::Create(ctx_, "skip", fn);
        b.CreateCondBr(take, then_bb, cont_bb);

        b.SetInsertPoint(then_bb);
        auto* d = b.CreateCall(obj_fns[i], {x, y, z, pb});
        b.CreateStore(frep::llvm_compat::binary_intrinsic(
            b, llvm::Intrinsic::minnum, cur, d), best);
        b.CreateBr(cont_bb);

        b.SetInsertPoint(cont_bb);
    }

    b.CreateRet(b.CreateLoad(f32(), best));
    verify_fn(fn);
    return fn;
}
//
// float scene_sdf_grad(float xv, float xd, float yv, float yd,
//                      float zv, float zd, float* out_dot)
//
// Returns the SDF value; writes the directional derivative along (xd,yd,zd) to out_dot.
// Called 3 times by scene_normal with tangent vectors (1,0,0),(0,1,0),(0,0,1).
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_sdf_grad(const FRepNode& root) {
    // signature: (x_val, x_dot, y_val, y_dot, z_val, z_dot, out_dot, params)
    // The trailing `params` is a float* used in Incremental mode to load
    // node parameters at runtime; in Constant mode it is unused (and DCE'd
    // by O3 since no instruction references it).
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), f32(), f32(), f32(), fptr(), fptr()}, false);
    auto* fn = llvm::Function::Create(fty,
                   llvm::Function::ExternalLinkage, "scene_sdf_grad", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);

    auto it = fn->arg_begin();
    auto* xv = &*it++; xv->setName("x_val");
    auto* xd = &*it++; xd->setName("x_dot");
    auto* yv = &*it++; yv->setName("y_val");
    auto* yd = &*it++; yd->setName("y_dot");
    auto* zv = &*it++; zv->setName("z_val");
    auto* zd = &*it++; zd->setName("z_dot");
    auto* od = &*it++; od->setName("out_dot");
    auto* pb = &*it++; pb->setName("params");

    CgCtx cctx = make_cgctx(b, pb);
    FRepNode::DualVal x{xv, xd}, y{yv, yd}, z{zv, zd};
    auto result = root.codegen_grad(cctx, x, y, z);

    b.CreateStore(result.dot, od);
    b.CreateRet(result.val);

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_scene_normal — normal via forward-mode AD.
// void scene_normal(float x, float y, float z, float* nx, float* ny, float* nz)
//
// Calls scene_sdf_grad 3 times with tangents (1,0,0), (0,1,0), (0,0,1).
// Each call yields df/d(axis). This is mathematically exact — unlike
// finite-diff, which is an approximation and noisy near the surface.
// 3 calls vs 6 (finite-diff) → ~2x fewer SDF evaluations.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_normal(llvm::Function* sdf_grad_fn) {
    // signature: (x, y, z, out_nx, out_ny, out_nz, params)
    auto* fty = llvm::FunctionType::get(vd(),
        {f32(), f32(), f32(), fptr(), fptr(), fptr(), fptr()}, false);
    auto* fn = llvm::Function::Create(fty,
                   llvm::Function::ExternalLinkage, "scene_normal", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);

    auto it = fn->arg_begin();
    auto* px  = &*it++; px->setName("x");
    auto* py  = &*it++; py->setName("y");
    auto* pz  = &*it++; pz->setName("z");
    auto* onx = &*it++; onx->setName("out_nx");
    auto* ony = &*it++; ony->setName("out_ny");
    auto* onz = &*it++; onz->setName("out_nz");
    auto* pb  = &*it++; pb->setName("params");

    auto zero = llvm::ConstantFP::get(f32(), 0.0f);
    auto one  = llvm::ConstantFP::get(f32(), 1.0f);

    // out_dot scratch — alloca in the entry block
    auto* scratch = b.CreateAlloca(f32(), nullptr, "grad_dot");

    // 3 calls with different tangent vectors.
    // call(x, x_dot=1, y, 0, z, 0) → ∂f/∂x
    auto grad = [&](llvm::Value* xd, llvm::Value* yd, llvm::Value* zd) -> llvm::Value* {
        b.CreateCall(sdf_grad_fn, {px, xd, py, yd, pz, zd, scratch, pb});
        return b.CreateLoad(f32(), scratch, "df");
    };

    auto dnx = grad(one,  zero, zero);
    auto dny = grad(zero, one,  zero);
    auto dnz = grad(zero, zero, one);

    // Normalization
    auto len2 = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(dnx, dnx), b.CreateFMul(dny, dny)),
        b.CreateFMul(dnz, dnz), "len2");
    auto len = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, len2, "len");

    auto tiny    = llvm::ConstantFP::get(f32(), 1e-6f);
    auto is_tiny = b.CreateFCmpOLT(len, tiny, "is_tiny");
    auto safe    = b.CreateSelect(is_tiny, one, len, "safe");

    auto nx_n = b.CreateFDiv(dnx, safe, "nx_n");
    auto ny_n = b.CreateSelect(is_tiny, one, b.CreateFDiv(dny, safe), "ny_n");
    auto nz_n = b.CreateFDiv(dnz, safe, "nz_n");

    b.CreateStore(nx_n, onx);
    b.CreateStore(ny_n, ony);
    b.CreateStore(nz_n, onz);
    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_shader
// void shade_pixel(float nx,ny,nz, float lx,ly,lz, float vx,vy,vz,
//                  float albR,albG,albB, float roughness, float metallic,
//                  float* r, float* g, float* b_out)
//
// Single-light shader. Dispatches between two models based on TracerConfig:
//
//   BlinnPhong:    fast Phong-style highlight with metallic tinting.
//                  Output = albedo*(ambient + (1-metal)*ndotl) + spec_tint*spec.
//
//   CookTorrance:  physically-based microfacet BRDF.
//                  D = GGX, G = Smith-GGX, F = Schlick.
//                  f = (kd * albedo/pi + D*G*F / (4*ndotl*ndotv)) * ndotl.
//                  No ambient term — caller must add it if desired.
//
// In both cases the caller multiplies by per-light shadow + colour + intensity
// outside the shader. This keeps the shader a stateless per-light call so
// the same emitted function serves any number of lights.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_shader() {
    // args: nx,ny,nz, lx,ly,lz, vx,vy,vz, albR,albG,albB, roughness, metallic,
    //       *r, *g, *b
    std::vector<llvm::Type*> params(14, f32());
    params.push_back(fptr()); params.push_back(fptr()); params.push_back(fptr());
    auto* fty = llvm::FunctionType::get(vd(), params, false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "shade_pixel", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);

    auto it = fn->arg_begin();
    auto* nx = &*it++; auto* ny = &*it++; auto* nz = &*it++;
    auto* lx = &*it++; auto* ly = &*it++; auto* lz = &*it++;
    auto* vx = &*it++; auto* vy = &*it++; auto* vz = &*it++;
    auto* ar = &*it++; auto* ag = &*it++; auto* ab_alb = &*it++;
    auto* rough = &*it++; auto* metal = &*it++;
    auto* or_ = &*it++; auto* og = &*it++; auto* ob = &*it++;

    auto fc = [&](float v) { return llvm::ConstantFP::get(f32(), v); };

    // ── Shared geometry: dot products and half vector ────────────────────────
    auto dot_nl = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(nx, lx), b.CreateFMul(ny, ly)),
        b.CreateFMul(nz, lz), "ndotl");
    auto ndotl = frep::llvm_compat::max_num(b, dot_nl, fc(0.0f), "cndotl");

    auto hx = b.CreateFAdd(lx, vx);
    auto hy = b.CreateFAdd(ly, vy);
    auto hz = b.CreateFAdd(lz, vz);
    auto hlen2 = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(hx, hx), b.CreateFMul(hy, hy)),
        b.CreateFMul(hz, hz));
    auto hlen2_safe = b.CreateFAdd(hlen2, fc(1e-8f));
    auto hlen = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, hlen2_safe);
    auto inv_hlen = b.CreateFDiv(fc(1.0f), hlen);
    auto hxn = b.CreateFMul(hx, inv_hlen);
    auto hyn = b.CreateFMul(hy, inv_hlen);
    auto hzn = b.CreateFMul(hz, inv_hlen);

    auto dot_nh = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(nx, hxn), b.CreateFMul(ny, hyn)),
        b.CreateFMul(nz, hzn));
    auto ndoth = frep::llvm_compat::max_num(b, dot_nh, fc(0.0f));

    auto front = b.CreateFCmpOGT(ndotl, fc(0.0f));

    if (cfg_.shading_model == TracerConfig::ShadingModel::BlinnPhong) {
        // ── Blinn-Phong path ─────────────────────────────────────────────────
        // We mirror the GPU GLSL emitter's Blinn-Phong: a single
        // specular term using half-vector pow(ndoth, shininess)
        // blended by a metallic factor, plus a Lambertian diffuse
        // term — no Lambert /pi, no ambient. The caller
        // (`emit_pixel_shader`) adds the ambient term once per
        // pixel, multiplied by AO, exactly matching GPU's
        //   Lo = albedo * 0.08 * ao_v;             // ambient
        //   Lo += (diffuse + specular) * shadow * lI;   // per-light
        // formula.
        auto r2 = b.CreateFMul(rough, rough);
        auto r2_safe = frep::llvm_compat::max_num(b, r2, fc(1e-3f));
        auto shin_raw = b.CreateFSub(b.CreateFDiv(fc(2.0f), r2_safe), fc(2.0f));
        auto shin = frep::llvm_compat::min_num(b,
                        frep::llvm_compat::max_num(b, shin_raw, fc(2.0f)),
                        fc(2048.0f));
        auto spec_raw = frep::llvm_compat::binary_intrinsic(b,
                            llvm::Intrinsic::pow, ndoth, shin, "spec");
        auto spec  = b.CreateSelect(front, spec_raw, fc(0.0f));

        auto k_dielectric = fc(0.04f);
        auto one_minus_m  = b.CreateFSub(fc(1.0f), metal);

        auto shade = [&](llvm::Value* alb, llvm::Value* out) {
            auto spec_tint = b.CreateFAdd(
                b.CreateFMul(metal, alb),
                b.CreateFMul(one_minus_m, fc(1.0f)));
            auto spec_coef = b.CreateFAdd(
                b.CreateFMul(metal, fc(1.0f)),
                b.CreateFMul(one_minus_m, k_dielectric));
            auto spec_contrib = b.CreateFMul(
                b.CreateFMul(spec_tint, spec_coef), spec);
            // Diffuse: alb * (1 - metal) * ndotl. No ambient here.
            auto diffuse = b.CreateFMul(
                b.CreateFMul(alb, one_minus_m), ndotl);
            b.CreateStore(b.CreateFAdd(diffuse, spec_contrib), out);
        };
        shade(ar,     or_);
        shade(ag,     og);
        shade(ab_alb, ob);
    } else {
        // ── Cook-Torrance (GGX microfacet) path ──────────────────────────────
        // n . v (clamped, used in geometry term and BRDF denominator)
        auto dot_nv = b.CreateFAdd(
            b.CreateFAdd(b.CreateFMul(nx, vx), b.CreateFMul(ny, vy)),
            b.CreateFMul(nz, vz));
        auto ndotv = frep::llvm_compat::max_num(b, dot_nv, fc(0.0f));

        // v . h (Fresnel parameter; equals l . h for the half vector)
        auto dot_vh = b.CreateFAdd(
            b.CreateFAdd(b.CreateFMul(vx, hxn), b.CreateFMul(vy, hyn)),
            b.CreateFMul(vz, hzn));
        auto vdoth = frep::llvm_compat::max_num(b, dot_vh, fc(0.0f));

        // alpha = roughness^2  (Disney convention — more perceptual)
        auto alpha = b.CreateFMul(rough, rough);
        auto alpha2 = b.CreateFMul(alpha, alpha);

        // D(h) = alpha^2 / (pi * (ndoth^2 * (alpha^2 - 1) + 1)^2)
        auto ndoth2 = b.CreateFMul(ndoth, ndoth);
        auto a2_m1  = b.CreateFSub(alpha2, fc(1.0f));
        auto inner  = b.CreateFAdd(b.CreateFMul(ndoth2, a2_m1), fc(1.0f));
        auto inner2 = b.CreateFMul(inner, inner);
        auto inner2_safe = frep::llvm_compat::max_num(b, inner2, fc(1e-7f));
        auto D = b.CreateFDiv(alpha2,
                              b.CreateFMul(fc(3.14159265f), inner2_safe));

        // Smith-GGX (height-correlated approximation, separable form):
        //   G_l = ndotl / (ndotl * (1 - k) + k),   k = (alpha + 1)^2 / 8
        //   G_v = ndotv / (ndotv * (1 - k) + k)
        //   G   = G_l * G_v
        auto k_g_raw = b.CreateFAdd(rough, fc(1.0f));
        auto k_g = b.CreateFDiv(b.CreateFMul(k_g_raw, k_g_raw), fc(8.0f));
        auto one_m_k = b.CreateFSub(fc(1.0f), k_g);
        auto gv_denom = b.CreateFAdd(b.CreateFMul(ndotv, one_m_k), k_g);
        auto gl_denom = b.CreateFAdd(b.CreateFMul(ndotl, one_m_k), k_g);
        auto gv_safe = frep::llvm_compat::max_num(b, gv_denom, fc(1e-5f));
        auto gl_safe = frep::llvm_compat::max_num(b, gl_denom, fc(1e-5f));
        auto G_v = b.CreateFDiv(ndotv, gv_safe);
        auto G_l = b.CreateFDiv(ndotl, gl_safe);
        // (G = G_v · G_l is folded into the visibility term below, where
        //  the ndotv·ndotl factors cancel against the BRDF denominator.)
        (void)G_v; (void)G_l;

        // Fresnel-Schlick: F = F0 + (1 - F0) * (1 - vdoth)^5
        // F0 = mix(0.04, albedo, metallic) — done per channel below.
        auto one_minus_vh = b.CreateFSub(fc(1.0f), vdoth);
        auto vh2 = b.CreateFMul(one_minus_vh, one_minus_vh);
        auto vh4 = b.CreateFMul(vh2, vh2);
        auto vh5 = b.CreateFMul(vh4, one_minus_vh);

        // Combined specular visibility term.
        //
        // The naive form is  D * G / (4·ndotv·ndotl), with
        //   G = G_v · G_l,  G_v = ndotv/gv_safe,  G_l = ndotl/gl_safe.
        // Substituting and cancelling the ndotv·ndotl that appears in
        // both G and the denominator gives the algebraically identical
        //   D / (4 · gv_safe · gl_safe).
        // This cancellation is the key: in the naive form ndotv → 0 at
        // grazing silhouette angles blows the quotient up into a bright
        // white Fresnel rim (the thin light outline tracing every
        // object against a dark background). Cancelling removes the
        // 1/ndotv·ndotl singularity entirely, so the term stays bounded
        // and the rim disappears — without the heavy-handed firefly
        // clamp that would also dim legitimate highlights. (We keep a
        // generous clamp below purely as a NaN/overflow backstop.)
        auto vis_denom = b.CreateFMul(
            b.CreateFMul(fc(4.0f), gv_safe), gl_safe);
        auto vis_denom_safe = frep::llvm_compat::max_num(b, vis_denom, fc(1e-5f));
        auto dg_over_d = b.CreateFDiv(D, vis_denom_safe);

        auto one_minus_m = b.CreateFSub(fc(1.0f), metal);
        // PI compensation factor for specular. The GPU emitter
        // applies `specular * PI` to compensate for the lighting
        // convention that bakes PI into the per-light intensity (see
        // Frostbite "Moving Frostbite to PBR" PDF). We mirror it
        // here so CPU and GPU produce visually identical results.
        auto pi          = fc(3.14159265358979f);

        auto shade = [&](llvm::Value* alb, llvm::Value* out) {
            // F0 = mix(0.04, alb, metal)
            auto F0 = b.CreateFAdd(
                b.CreateFMul(one_minus_m, fc(0.04f)),
                b.CreateFMul(metal,        alb));
            auto F = b.CreateFAdd(F0,
                b.CreateFMul(b.CreateFSub(fc(1.0f), F0), vh5));

            // Specular term: F * D * G / (4 * ndotl * ndotv) * PI.
            // The PI factor matches the GPU GLSL emitter's
            // `specular * PI` term — both paths compensate for the
            // engine's per-light intensity convention. Without this,
            // the CPU output's specular highlights were noticeably
            // dimmer than the GPU output's.
            auto spec_brdf_raw = b.CreateFMul(
                b.CreateFMul(F, dg_over_d), pi);
            // Firefly clamp — mirror the GPU emitter's
            // `specular = min(specular, 8.0)` (applied there before the
            // ×PI factor, so the cap here is 8·PI). At grazing
            // silhouette angles the 1/(4·ndotl·ndotv) denominator blows
            // the specular up into a white outline fringe; capping it
            // removes the fringe while leaving real highlights intact.
            auto spec_cap      = fc(8.0f * 3.14159265358979f);
            auto spec_clamped  = frep::llvm_compat::min_num(b, spec_brdf_raw, spec_cap);
            auto spec_brdf_sel = b.CreateSelect(front, spec_clamped, fc(0.0f));
            // Diagnostic: drop the specular term entirely when disabled, to
            // isolate whether a CPU-vs-GLSL divergence is in specular or
            // diffuse. Must match the GLSL emitter's enable_specular gate.
            auto spec_brdf     = cfg_.enable_specular ? spec_brdf_sel : fc(0.0f);

            // Energy conservation: diffuse fraction = (1 - F) * (1 - metal).
            // NOTE: we deliberately drop the `/ pi` Lambert normalisation
            // here so the CPU matches the GPU emitter's
            // `kd * albedo * ndotl * shadow * lI` (no 1/pi). Strict PBR
            // would divide by pi; the GPU convention (and now CPU
            // convention) bakes pi back into the per-light intensity,
            // which is what most authored scenes assume.
            auto kd = b.CreateFMul(b.CreateFSub(fc(1.0f), F), one_minus_m);
            auto diff_brdf = b.CreateFMul(kd, alb);

            // Final per-light contribution = (diff + spec) * ndotl.
            // Ambient is added in `emit_pixel_shader` AFTER this
            // function returns (and multiplied by AO there), so the
            // shading model matches the GPU emitter's
            //   Lo = albedo * 0.08 * ao;            // ambient
            //   Lo += (kd*albedo + spec*PI) * ndotl * shadow * lI;
            // exactly. No ambient term written here — the caller
            // handles it.
            auto contrib = b.CreateFMul(
                b.CreateFAdd(diff_brdf, spec_brdf), ndotl);
            b.CreateStore(contrib, out);
        };
        shade(ar,     or_);
        shade(ag,     og);
        shade(ab_alb, ob);
    }

    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_shadow
// float shadow(float ox,oy,oz, float dx,dy,dz)
//
// Soft shadow ray (Inigo Quilez formula):
//   t = epsilon (offset to avoid self-intersection)
//   k_min = 1.0
//   for step in 0..N:
//     h = sdf(o + t*d)
//     if h < tiny:  return 0.0  (full shadow)
//     k_min = min(k_min, softness * h / t)
//     t += h * safety
//     if t > max_dist: break
//   return clamp(k_min, 0, 1)
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_shadow(llvm::Function* sdf_fn) {
    // signature: (ox, oy, oz, dx, dy, dz, max_t, params)
    // max_t is the distance to the light: the shadow ray must STOP there, not
    // march on to max_dist. Marching past the light let the CPU ray hit
    // geometry behind the light and report a false shadow — this was the
    // entire twist divergence (the GLSL emitter breaks at max_t correctly).
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), f32(), f32(), f32(), f32(), fptr()}, false);
    auto* fn = llvm::Function::Create(fty,
                   llvm::Function::ExternalLinkage, "shadow_ray", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto it = fn->arg_begin();
    auto* OX = &*it++; auto* OY = &*it++; auto* OZ = &*it++;
    auto* DX = &*it++; auto* DY = &*it++; auto* DZ = &*it++;
    auto* MAXT = &*it++; MAXT->setName("max_t");
    auto* PB = &*it++; PB->setName("params");

    auto* entry_bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    auto* cond_bb  = llvm::BasicBlock::Create(ctx_, "cond",  fn);
    auto* body_bb  = llvm::BasicBlock::Create(ctx_, "body",  fn);
    auto* dark_bb  = llvm::BasicBlock::Create(ctx_, "dark",  fn);
    auto* exit_bb  = llvm::BasicBlock::Create(ctx_, "exit",  fn);

    llvm::IRBuilder<> b(entry_bb);
    b.CreateBr(cond_bb);

    b.SetInsertPoint(cond_bb);
    auto* step_phi = b.CreatePHI(i32(), 2, "s_step");
    auto* t_phi    = b.CreatePHI(f32(), 2, "s_t");
    auto* k_phi    = b.CreatePHI(f32(), 2, "s_k");
    step_phi->addIncoming(b.getInt32(0),                                 entry_bb);
    // Ray start t = 0.05 along the light direction (matches GLSL); combined
    // with the 0.01 origin normal offset this reproduces the GLSL standoff.
    t_phi   ->addIncoming(llvm::ConstantFP::get(f32(), 0.05f),            entry_bb);
    k_phi   ->addIncoming(llvm::ConstantFP::get(f32(), 1.0f),              entry_bb);

    auto* step_ok = b.CreateICmpSLT(step_phi, b.getInt32(cfg_.shadow_steps));
    // Stop at the light (max_t), matching the GLSL emitter — not at max_dist.
    auto* dist_ok = b.CreateFCmpOLT(t_phi, MAXT);
    b.CreateCondBr(b.CreateAnd(step_ok, dist_ok), body_bb, exit_bb);

    b.SetInsertPoint(body_bb);
    auto* px = b.CreateFAdd(OX, b.CreateFMul(t_phi, DX));
    auto* py = b.CreateFAdd(OY, b.CreateFMul(t_phi, DY));
    auto* pz = b.CreateFAdd(OZ, b.CreateFMul(t_phi, DZ));
    auto* h  = b.CreateCall(sdf_fn, {px, py, pz, PB}, "h");

    // if h < 0.002 → full shadow (matches GLSL shadow-ray epsilon).
    auto* in_obj = b.CreateFCmpOLT(h,
                       llvm::ConstantFP::get(f32(), 0.002f));

    // k_new = min(k_phi, softness * h / t).
    // When multi-sampling (shadow_samples > 1) we want a HARD binary
    // ray — the softness comes from averaging many area-light samples
    // in the pixel shader, not from the IQ penumbra term. So we freeze
    // k at 1.0 and rely purely on the in-object early-out for occlusion,
    // matching the GLSL emitter's behaviour and making the sample count
    // / light radius controls actually visible.
    llvm::Value* k_next;
    if (cfg_.shadow_samples > 1) {
        k_next = k_phi;   // stays 1.0 — hard ray
    } else {
        auto* ratio   = b.CreateFDiv(
                            b.CreateFMul(llvm::ConstantFP::get(f32(), cfg_.shadow_softness), h),
                            t_phi);
        auto* k_clamp = frep::llvm_compat::max_num(b, ratio, llvm::ConstantFP::get(f32(), 0.0f));
        k_next        = frep::llvm_compat::min_num(b, k_phi, k_clamp);
    }

    auto* t_step  = b.CreateFMul(h, llvm::ConstantFP::get(f32(), cfg_.safety_factor));
    auto* t_next  = b.CreateFAdd(t_phi, t_step);
    auto* s_next  = b.CreateAdd(step_phi, b.getInt32(1));

    step_phi->addIncoming(s_next, body_bb);
    t_phi   ->addIncoming(t_next, body_bb);
    k_phi   ->addIncoming(k_next, body_bb);

    b.CreateCondBr(in_obj, dark_bb, cond_bb);

    b.SetInsertPoint(dark_bb);
    b.CreateRet(llvm::ConstantFP::get(f32(), 0.0f));

    b.SetInsertPoint(exit_bb);
    // clamp(k_phi, 0, 1)
    auto* clamped = frep::llvm_compat::min_num(b,
        frep::llvm_compat::max_num(b, k_phi, llvm::ConstantFP::get(f32(), 0.0f)),
        llvm::ConstantFP::get(f32(), 1.0f));
    b.CreateRet(clamped);

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_ao
// float ao(float px,py,pz, float nx,ny,nz)
//
// SDF-based ambient occlusion (cheap approximation):
//   occlusion = 0
//   for i in 1..N:
//     dist = i * step
//     h = sdf(p + n * dist)
//     occlusion += (dist - h) * weight(i)
//   return 1 - strength * clamp(occlusion, 0, 1)
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_ao(llvm::Function* sdf_fn) {
    // signature: (px, py, pz, nx, ny, nz, params)
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), f32(), f32(), f32(), fptr()}, false);
    auto* fn = llvm::Function::Create(fty,
                   llvm::Function::ExternalLinkage, "ambient_occlusion", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto it = fn->arg_begin();
    auto* PX = &*it++; auto* PY = &*it++; auto* PZ = &*it++;
    auto* NX = &*it++; auto* NY = &*it++; auto* NZ = &*it++;
    auto* PB = &*it++; PB->setName("params");

    auto* bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(bb);

    // Unrolled loop — N is small (~5), the optimizer will unroll it.
    llvm::Value* total = llvm::ConstantFP::get(f32(), 0.0f);
    llvm::Value* w     = llvm::ConstantFP::get(f32(), 1.0f);

    for (int i = 1; i <= cfg_.ao_samples; ++i) {
        float dist = i * cfg_.ao_step;
        auto* d_c  = llvm::ConstantFP::get(f32(), dist);
        // p + n*dist
        auto* sx = b.CreateFAdd(PX, b.CreateFMul(NX, d_c));
        auto* sy = b.CreateFAdd(PY, b.CreateFMul(NY, d_c));
        auto* sz = b.CreateFAdd(PZ, b.CreateFMul(NZ, d_c));
        auto* h  = b.CreateCall(sdf_fn, {sx, sy, sz, PB});
        // (dist - h) — if large, there is nearby geometry in front of us (occlusion)
        auto* diff = b.CreateFSub(d_c, h);
        // weighted contribution
        auto* contrib = b.CreateFMul(diff, w);
        total = b.CreateFAdd(total, contrib);
        // half weight each time
        w = b.CreateFMul(w, llvm::ConstantFP::get(f32(), 0.5f));
    }

    // clamp(total, 0, 1)
    auto* cl = frep::llvm_compat::min_num(b,
        frep::llvm_compat::max_num(b, total, llvm::ConstantFP::get(f32(), 0.0f)),
        llvm::ConstantFP::get(f32(), 1.0f));
    // 1 - strength * cl
    auto* ao_v = b.CreateFSub(
        llvm::ConstantFP::get(f32(), 1.0f),
        b.CreateFMul(cl, llvm::ConstantFP::get(f32(), cfg_.ao_strength)));
    b.CreateRet(ao_v);

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_tracer
//
// void render_tile(float* out_rgba,
//                  int tx, int ty, int tw, int th, int iw, int ih,
//                  float camOx, camOy, camOz,
//                  float camDx, camDy, camDz,   // forward (normalized)
//                  float camRx, camRy, camRz,   // right   (normalized)
//                  float camUx, camUy, camUz,   // up      (normalized)
//                  float fovScale)              // tan(fov/2)
//
// Algorithm: sphere tracing for each pixel in the tile.
// On hit: compute the normal, shade.
// On miss: sky gradient.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_shade_hit(const SceneGraph& scene,
                                             llvm::Function* normal_fn,
                                             llvm::Function* shader_fn,
                                             llvm::Function* mat_fn,
                                             llvm::Function* shadow_fn,
                                             llvm::Function* ao_fn) {
    // signature: (hx,hy,hz, vx,vy,vz, out_r,out_g,out_b, params) -> void
    auto* fty = llvm::FunctionType::get(vd(),
        {f32(), f32(), f32(),          // hit point
         f32(), f32(), f32(),          // view direction (toward eye)
         fptr(), fptr(), fptr(),       // out r, g, b
         fptr()},                      // params
        false);
    auto* fn = llvm::Function::Create(fty,
                   llvm::Function::ExternalLinkage, "shade_hit", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto it = fn->arg_begin();
    auto* hx = &*it++; hx->setName("hx");
    auto* hy = &*it++; hy->setName("hy");
    auto* hz = &*it++; hz->setName("hz");
    auto* vx = &*it++; vx->setName("vx");
    auto* vy = &*it++; vy->setName("vy");
    auto* vz = &*it++; vz->setName("vz");
    auto* o_r = &*it++; o_r->setName("out_r");
    auto* o_g = &*it++; o_g->setName("out_g");
    auto* o_b = &*it++; o_b->setName("out_b");
    auto* PARAMS = &*it++; PARAMS->setName("params");

    auto* entry_bb = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(entry_bb);
    auto fc = [&](float v){ return llvm::ConstantFP::get(f32(), v); };

    // Local allocas (entry block for SSA correctness).
    auto* a_nx = b.CreateAlloca(f32(), nullptr, "a_nx");
    auto* a_ny = b.CreateAlloca(f32(), nullptr, "a_ny");
    auto* a_nz = b.CreateAlloca(f32(), nullptr, "a_nz");
    auto* a_cr = b.CreateAlloca(f32(), nullptr, "a_cr");
    auto* a_cg = b.CreateAlloca(f32(), nullptr, "a_cg");
    auto* a_cb = b.CreateAlloca(f32(), nullptr, "a_cb");
    auto* a_ar = b.CreateAlloca(f32(), nullptr, "a_ar");
    auto* a_ag = b.CreateAlloca(f32(), nullptr, "a_ag");
    auto* a_ab = b.CreateAlloca(f32(), nullptr, "a_ab");
    auto* a_rough = b.CreateAlloca(f32(), nullptr, "a_rough");
    auto* a_metal = b.CreateAlloca(f32(), nullptr, "a_metal");
    auto* a_lr = b.CreateAlloca(f32(), nullptr, "a_lr");
    auto* a_lg = b.CreateAlloca(f32(), nullptr, "a_lg");
    auto* a_lb = b.CreateAlloca(f32(), nullptr, "a_lb");

    // Normal at the hit point.
    b.CreateCall(normal_fn, {hx, hy, hz, a_nx, a_ny, a_nz, PARAMS});
    auto* nx_v = b.CreateLoad(f32(), a_nx, "nx_v");
    auto* ny_v = b.CreateLoad(f32(), a_ny, "ny_v");
    auto* nz_v = b.CreateLoad(f32(), a_nz, "nz_v");

    // Material at the hit point.
    b.CreateCall(mat_fn, {hx, hy, hz, a_ar, a_ag, a_ab, a_rough, a_metal, PARAMS});
    auto* alb_r = b.CreateLoad(f32(), a_ar, "alb_r");
    auto* alb_g = b.CreateLoad(f32(), a_ag, "alb_g");
    auto* alb_b = b.CreateLoad(f32(), a_ab, "alb_b");
    auto* rough_v = b.CreateLoad(f32(), a_rough, "rough");
    auto* metal_v = b.CreateLoad(f32(), a_metal, "metal");

    // Shadow ray origin: offset off the surface to avoid self-shadow.
    // Match the GLSL emitter's fixed 0.01 normal offset (it then starts the
    // ray at t=0.05 along the light direction). The combined standoff decides
    // how close to the surface self-shadowing is sampled — a mismatch here
    // was the twist divergence (twist self-shadows through its concavities,
    // so the standoff distance changes the result; smooth convex shapes are
    // insensitive to it, which is why only twist diverged).
    auto offset = fc(0.01f);
    auto* sx = b.CreateFAdd(hx, b.CreateFMul(nx_v, offset));
    auto* sy = b.CreateFAdd(hy, b.CreateFMul(ny_v, offset));
    auto* sz = b.CreateFAdd(hz, b.CreateFMul(nz_v, offset));

    // Ambient occlusion factor.
    llvm::Value* ao_v = fc(1.0f);
    if (ao_fn) {
        ao_v = b.CreateCall(ao_fn, {hx, hy, hz, nx_v, ny_v, nz_v, PARAMS}, "ao");
    }

    b.CreateStore(fc(0.0f), a_cr);
    b.CreateStore(fc(0.0f), a_cg);
    b.CreateStore(fc(0.0f), a_cb);

    const auto* lights_ptr = &scene.lights();
    static const std::vector<PointLight> kFallbackLights = {
        PointLight{{5.0f, 10.0f, 5.0f}, {1.0f, 1.0f, 1.0f}, 1.0f}
    };
    if (lights_ptr->empty()) lights_ptr = &kFallbackLights;

    for (const auto& L : *lights_ptr) {
        auto* dxl = b.CreateFSub(fc(L.pos[0]), hx);
        auto* dyl = b.CreateFSub(fc(L.pos[1]), hy);
        auto* dzl = b.CreateFSub(fc(L.pos[2]), hz);
        auto* len2_l = b.CreateFAdd(
            b.CreateFAdd(b.CreateFMul(dxl, dxl), b.CreateFMul(dyl, dyl)),
            b.CreateFMul(dzl, dzl));
        auto* len_l = frep::llvm_compat::unary_intrinsic(b,
                          llvm::Intrinsic::sqrt, b.CreateFAdd(len2_l, fc(1e-8f)));
        auto* inv_l = b.CreateFDiv(fc(1.0f), len_l);
        auto* lxn   = b.CreateFMul(dxl, inv_l);
        auto* lyn   = b.CreateFMul(dyl, inv_l);
        auto* lzn   = b.CreateFMul(dzl, inv_l);

        llvm::Value* shadow_v = fc(1.0f);
        if (shadow_fn) {
            if (cfg_.shadow_samples > 1) {
                const int   N = cfg_.shadow_samples;
                const float R = cfg_.shadow_light_radius;
                llvm::Value* sum = fc(0.0f);
                for (int s = 0; s < N; ++s) {
                    float fi  = (static_cast<float>(s) + 0.5f) / static_cast<float>(N);
                    float ang = static_cast<float>(s) * 2.39996323f;
                    float rad = std::sqrt(fi) * R;
                    float ox  = std::cos(ang) * rad;
                    float oy  = std::sin(ang) * rad;
                    float oz  = (fi - 0.5f) * rad;
                    auto* jx = b.CreateFSub(fc(L.pos[0] + ox), hx);
                    auto* jy = b.CreateFSub(fc(L.pos[1] + oy), hy);
                    auto* jz = b.CreateFSub(fc(L.pos[2] + oz), hz);
                    auto* jl2 = b.CreateFAdd(
                        b.CreateFAdd(b.CreateFMul(jx, jx), b.CreateFMul(jy, jy)),
                        b.CreateFMul(jz, jz));
                    auto* jlen = frep::llvm_compat::unary_intrinsic(b,
                        llvm::Intrinsic::sqrt, b.CreateFAdd(jl2, fc(1e-8f)));
                    auto* jinv = b.CreateFDiv(fc(1.0f), jlen);
                    auto* one = b.CreateCall(shadow_fn,
                        {sx, sy, sz,
                         b.CreateFMul(jx, jinv), b.CreateFMul(jy, jinv),
                         b.CreateFMul(jz, jinv), jlen, PARAMS}, "ss");
                    sum = b.CreateFAdd(sum, one);
                }
                shadow_v = b.CreateFMul(sum, fc(1.0f / static_cast<float>(N)));
            } else {
                shadow_v = b.CreateCall(shadow_fn,
                    {sx, sy, sz, lxn, lyn, lzn, len_l, PARAMS}, "shadow");
            }
        }

        b.CreateCall(shader_fn, {nx_v, ny_v, nz_v, lxn, lyn, lzn,
                                  vx, vy, vz,
                                  alb_r, alb_g, alb_b, rough_v, metal_v,
                                  a_lr, a_lg, a_lb});
        auto* lr = b.CreateLoad(f32(), a_lr);
        auto* lg = b.CreateLoad(f32(), a_lg);
        auto* lb = b.CreateLoad(f32(), a_lb);

        auto* k = b.CreateFMul(shadow_v, fc(L.intensity));
        auto* cr = b.CreateFMul(b.CreateFMul(lr, k), fc(L.color[0]));
        auto* cg = b.CreateFMul(b.CreateFMul(lg, k), fc(L.color[1]));
        auto* cb = b.CreateFMul(b.CreateFMul(lb, k), fc(L.color[2]));
        b.CreateStore(b.CreateFAdd(b.CreateLoad(f32(), a_cr), cr), a_cr);
        b.CreateStore(b.CreateFAdd(b.CreateLoad(f32(), a_cg), cg), a_cg);
        b.CreateStore(b.CreateFAdd(b.CreateLoad(f32(), a_cb), cb), a_cb);
    }

    auto* raw_r = b.CreateLoad(f32(), a_cr);
    auto* raw_g = b.CreateLoad(f32(), a_cg);
    auto* raw_b = b.CreateLoad(f32(), a_cb);

    // GPU-style ambient = albedo * 0.08 * ao_v (added once).
    auto* amb_factor = b.CreateFMul(fc(0.08f), ao_v, "amb_factor");
    auto* out_rv = b.CreateFAdd(raw_r, b.CreateFMul(alb_r, amb_factor));
    auto* out_gv = b.CreateFAdd(raw_g, b.CreateFMul(alb_g, amb_factor));
    auto* out_bv = b.CreateFAdd(raw_b, b.CreateFMul(alb_b, amb_factor));

    b.CreateStore(out_rv, o_r);
    b.CreateStore(out_gv, o_g);
    b.CreateStore(out_bv, o_b);
    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

llvm::Function* SceneCodegen::emit_tracer(const SceneGraph& scene,
                                          llvm::Function* sdf_fn,
                                          llvm::Function* normal_fn,
                                          llvm::Function* shader_fn,
                                          llvm::Function* mat_fn) {
    // Scene bounding box (union of visible object AABBs) for the optional ray-box
    // near/far clip below. If any visible object is unbounded (planes, unknown
    // implicits report an infinite AABB), the clip is disabled — an infinite box
    // can't narrow anything. Computed once here; the march uses it per ray.
    bool scene_bounded = false;
    FRepNode::AABB scene_box{};
    {
        auto finite = [](float v){ return std::isfinite(v); };
        for (auto& [id, obj] : scene.objects()) {
            if (!obj.visible || !obj.geometry) continue;
            FRepNode::AABB b = obj.geometry->aabb();
            bool ok = finite(b.min_x)&&finite(b.min_y)&&finite(b.min_z)&&
                      finite(b.max_x)&&finite(b.max_y)&&finite(b.max_z);
            if (!ok) { scene_bounded = false; break; }
            if (!scene_bounded) { scene_box = b; scene_bounded = true; }
            else {
                scene_box.min_x = std::min(scene_box.min_x, b.min_x);
                scene_box.min_y = std::min(scene_box.min_y, b.min_y);
                scene_box.min_z = std::min(scene_box.min_z, b.min_z);
                scene_box.max_x = std::max(scene_box.max_x, b.max_x);
                scene_box.max_y = std::max(scene_box.max_y, b.max_y);
                scene_box.max_z = std::max(scene_box.max_z, b.max_z);
            }
        }
        if (scene_bounded) {
            float m = 0.05f * std::max({scene_box.max_x - scene_box.min_x,
                                        scene_box.max_y - scene_box.min_y,
                                        scene_box.max_z - scene_box.min_z, 0.1f});
            scene_box.min_x -= m; scene_box.min_y -= m; scene_box.min_z -= m;
            scene_box.max_x += m; scene_box.max_y += m; scene_box.max_z += m;
        }
    }
    const bool use_bbox_clip = scene_bounded && cfg_.bbox_clip;

    // Optional: shadow and AO. Pass nullptr if they are not needed.
    llvm::Function* shadow_fn = cfg_.enable_shadows ? emit_shadow(sdf_fn) : nullptr;
    llvm::Function* ao_fn     = cfg_.enable_ao      ? emit_ao(sdf_fn)     : nullptr;

    // Shading is factored into shade_hit so both the primary ray and the
    // reflection bounce rays can reuse it. Reflectivity lookup + the
    // bounce loop are only emitted when reflections are enabled.
    llvm::Function* shade_fn = emit_shade_hit(scene, normal_fn, shader_fn,
                                              mat_fn, shadow_fn, ao_fn);
    llvm::Function* refl_fn  = (cfg_.max_bounces > 0)
                                 ? emit_scene_reflectivity(scene) : nullptr;
    // ── Type of render_tile ───────────────────────────────────────────────────
    std::vector<llvm::Type*> pty = {
        fptr(),                          // out_rgba
        i32(), i32(), i32(), i32(), i32(), i32(), // tx,ty,tw,th,iw,ih
        f32(),f32(),f32(),               // cam origin
        f32(),f32(),f32(),               // cam forward
        f32(),f32(),f32(),               // cam right
        f32(),f32(),f32(),               // cam up
        f32(),                           // fov_scale (sign = projection mode)
        fptr()                           // params (incremental mode buffer, may be null)
    };
    auto* fty = llvm::FunctionType::get(vd(), pty, false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "render_tile", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr(llvm::Attribute::WillReturn);

    // ── Argument names ────────────────────────────────────────────────────────
    auto it = fn->arg_begin();
    auto* out  = &*it++; out->setName("out");
    auto* TX   = &*it++; TX->setName("tx");
    auto* TY   = &*it++; TY->setName("ty");
    auto* TW   = &*it++; TW->setName("tw");
    auto* TH   = &*it++; TH->setName("th");
    auto* IW   = &*it++; IW->setName("iw");
    auto* IH   = &*it++; IH->setName("ih");
    auto* OX   = &*it++; OX->setName("ox");
    auto* OY   = &*it++; OY->setName("oy");
    auto* OZ   = &*it++; OZ->setName("oz");
    auto* FDX  = &*it++; FDX->setName("fdx");
    auto* FDY  = &*it++; FDY->setName("fdy");
    auto* FDZ  = &*it++; FDZ->setName("fdz");
    auto* RX   = &*it++; RX->setName("rx");
    auto* RY   = &*it++; RY->setName("ry");
    auto* RZ   = &*it++; RZ->setName("rz");
    auto* UX   = &*it++; UX->setName("ux");
    auto* UY   = &*it++; UY->setName("uy");
    auto* UZ   = &*it++; UZ->setName("uz");
    auto* FOVS = &*it++; FOVS->setName("fovs");
    auto* PARAMS = &*it++; PARAMS->setName("params");

    // ── Basic blocks ──────────────────────────────────────────────────────────
    // Structure:
    //  entry → py_cond ↔ py_body → px_cond ↔ px_body → st_cond ↔ st_body
    //  st_body → [hit_bb | miss_bb] → write_bb → px_next → px_cond
    //  px_cond (done) → py_next → py_cond
    //  py_cond (done) → exit

    auto mk = [&](const char* n) { return llvm::BasicBlock::Create(ctx_, n, fn); };
    auto* entry_bb  = mk("entry");
    auto* py_cond   = mk("py_cond");
    auto* py_body   = mk("py_body");
    auto* px_cond   = mk("px_cond");
    auto* px_body   = mk("px_body");
    auto* st_cond   = mk("st_cond");   // sphere tracing loop header
    auto* st_body   = mk("st_body");   // sphere tracing loop body
    auto* hit_bb    = mk("hit");
    auto* miss_bb   = mk("miss");
    auto* write_bb  = mk("write");
    auto* px_next   = mk("px_next");
    auto* py_next   = mk("py_next");
    auto* exit_bb   = mk("exit");

    // ── Allocas for the normal and shader (in the entry block) ────────────────
    // They must be in the entry block for correct SSA form.
    auto* a_nx = entry_alloca(fn, "a_nx");
    auto* a_ny = entry_alloca(fn, "a_ny");
    auto* a_nz = entry_alloca(fn, "a_nz");
    auto* a_cr = entry_alloca(fn, "a_cr");
    auto* a_cg = entry_alloca(fn, "a_cg");
    auto* a_cb = entry_alloca(fn, "a_cb");
    // Albedo of the nearest object (filled by scene_material in hit_bb)
    auto* a_ar = entry_alloca(fn, "a_ar");
    auto* a_ag = entry_alloca(fn, "a_ag");
    auto* a_ab = entry_alloca(fn, "a_ab");
    // Per-object roughness and metallic.
    auto* a_rough = entry_alloca(fn, "a_rough");
    auto* a_metal = entry_alloca(fn, "a_metal");
    // Per-light shader output (reused across the light loop).
    auto* a_lr = entry_alloca(fn, "a_lr");
    auto* a_lg = entry_alloca(fn, "a_lg");
    auto* a_lb = entry_alloca(fn, "a_lb");

    llvm::IRBuilder<> b(entry_bb);

    // Precomputed constants
    auto iw_f  = b.CreateSIToFP(IW, f32(), "iw_f");
    auto ih_f  = b.CreateSIToFP(IH, f32(), "ih_f");
    auto aspect = b.CreateFDiv(iw_f, ih_f, "aspect");

    // Lights are emitted per-hit inside hit_bb (see below). The scene's
    // PointLight list is compile-time known, so we unroll the light loop
    // statically — each iteration emits one shadow_ray + shade_pixel call
    // and accumulates into a_cr/a_cg/a_cb.

    // Per-object albedo is fetched from scene_material(hit_point).
    // No global ALB constants are needed here — they come from a call in hit_bb.

    b.CreateBr(py_cond);

    // ══ Per-tile Lipschitz cull ═══════════════════════════════════════════════
    // The IR analog of the GLSL tile cull. Once per tile (here, before the pixel
    // loops), split [near, max_dist] into cull_slabs depth slabs; for each slab
    // build an AABB enclosing the tile frustum's corners at the slab's near/far
    // depth, and mark the slab occupied if the surface can pass through it —
    // Lipschitz test |scene_sdf(center)| <= L * halfdiag. The occupied slabs'
    // depth span [t0,t1] then clamps every ray's march. Unlike the whole-scene
    // ray-box clip (which only trims empty space outside the scene box), this can
    // also skip empty slabs *between* surfaces within the box. Emitted only when
    // cull_slabs > 0 and the method resolves to Lipschitz (interval cull would
    // need an IR interval emitter — future work); computed into tile_t0/tile_t1
    // which stay null (no effect) when off.
    llvm::Value* tile_t0 = nullptr;
    llvm::Value* tile_t1 = nullptr;
    {
        using CM = TracerConfig::CullMethod;
        // Metric if every visible object's tree is unit-Lipschitz (so L=1 is a
        // sound occupancy bound). Mirrors what emit_render_tile's root would give.
        bool metric = true;
        for (auto& [id, obj] : scene.objects()) {
            if (!obj.visible || !obj.geometry) continue;
            if (!node_is_unit_lipschitz(*obj.geometry)) { metric = false; break; }
        }
        // Decide the occupancy test: Lipschitz (L*halfdiag) for metric trees, or
        // interval (sound for any tree, via the IR interval emitter). Auto picks
        // Lipschitz on metric trees (cheapest, exact at L=1) and interval
        // otherwise. Interval/Lipschitz explicit are honoured.
        bool use_cull = cfg_.cull_slabs > 0 && cfg_.cull_method != CM::Off;
        bool use_interval = false;
        if (use_cull) {
            if (cfg_.cull_method == CM::Interval)      use_interval = true;
            else if (cfg_.cull_method == CM::Lipschitz) use_interval = false;
            else /* Auto */                             use_interval = !metric;
        }
        bool want_lip = use_cull;
        if (want_lip) {
            const int   S    = cfg_.cull_slabs;
            const float near = 0.001f;
            const float st   = (cfg_.max_dist - near) / float(S);
            const float L    = (cfg_.cull_method == CM::Auto) ? 1.0f
                                                              : cfg_.cull_lipschitz;
            auto* term = entry_bb->getTerminator();
            b.SetInsertPoint(term);   // emit before the branch to py_cond
            // Local float-constant helper (the member fc() takes a builder arg).
            auto fc = [&](float v){ return llvm::ConstantFP::get(f32(), v); };

            // For interval culling, emit a scene_ival(lo3,hi3)->{lo,hi} function
            // (union of visible objects, min of their interval bounds) once, so
            // each slab is one call rather than an inlined interval tree.
            llvm::Function* ival_fn = nullptr;
            if (use_interval) {
                auto* pairty = llvm::StructType::get(ctx_, {f32(), f32()});
                auto* ivty = llvm::FunctionType::get(pairty,
                    {f32(), f32(), f32(), f32(), f32(), f32()}, false);
                ival_fn = llvm::Function::Create(ivty,
                    llvm::Function::InternalLinkage, "scene_ival", mod_.get());
                ival_fn->addFnAttr(llvm::Attribute::NoUnwind);
                ival_fn->addFnAttr(llvm::Attribute::WillReturn);
                auto* ivbb = llvm::BasicBlock::Create(ctx_, "entry", ival_fn);
                llvm::IRBuilder<> ib(ivbb);
                auto ia = ival_fn->arg_begin();
                auto* Xlo=&*ia++; auto* Ylo=&*ia++; auto* Zlo=&*ia++;
                auto* Xhi=&*ia++; auto* Yhi=&*ia++; auto* Zhi=&*ia++;
                frep::jit::NodeIntervalIR em(ctx_, ib);
                llvm::Value* blo = fc(1e30f); llvm::Value* bhi = fc(1e30f);
                for (auto& [id, obj] : scene.objects()) {
                    if (!obj.visible || !obj.geometry) continue;
                    frep::jit::IvV r = em.emit(*obj.geometry,
                        {Xlo, Xhi}, {Ylo, Yhi}, {Zlo, Zhi});
                    blo = frep::llvm_compat::binary_intrinsic(ib, llvm::Intrinsic::minnum, blo, r.lo);
                    bhi = frep::llvm_compat::binary_intrinsic(ib, llvm::Intrinsic::minnum, bhi, r.hi);
                }
                llvm::Value* agg = llvm::UndefValue::get(pairty);
                agg = ib.CreateInsertValue(agg, blo, {0});
                agg = ib.CreateInsertValue(agg, bhi, {1});
                ib.CreateRet(agg);
                verify_fn(ival_fn);
            }

            auto sfp = [&](llvm::Value* v){ return b.CreateSIToFP(v, f32()); };
            auto* iwf = sfp(IW); auto* ihf = sfp(IH);
            auto* aspc = b.CreateFDiv(iwf, ihf);
            auto* tx0f = sfp(TX); auto* ty0f = sfp(TY);
            auto* tx1f = sfp(b.CreateAdd(TX, TW));
            auto* ty1f = sfp(b.CreateAdd(TY, TH));
            auto* is_o = b.CreateFCmpOLT(FOVS, fc(0.0f));
            auto* afov = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::fabs, FOVS);

            auto corner = [&](llvm::Value* pxf, llvm::Value* pyf,
                              llvm::Value*& ox, llvm::Value*& oy, llvm::Value*& oz,
                              llvm::Value*& dx, llvm::Value*& dy, llvm::Value*& dz) {
                auto* u = b.CreateFMul(b.CreateFSub(
                    b.CreateFMul(fc(2.0f), b.CreateFDiv(pxf, iwf)), fc(1.0f)), aspc);
                auto* v = b.CreateFSub(fc(1.0f),
                    b.CreateFMul(fc(2.0f), b.CreateFDiv(pyf, ihf)));
                auto mk = [&](llvm::Value* f, llvm::Value* r, llvm::Value* up){
                    return b.CreateFAdd(f, b.CreateFAdd(
                        b.CreateFMul(b.CreateFMul(u, afov), r),
                        b.CreateFMul(b.CreateFMul(v, afov), up))); };
                auto* pdx = mk(FDX, RX, UX); auto* pdy = mk(FDY, RY, UY); auto* pdz = mk(FDZ, RZ, UZ);
                auto* pl = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt,
                    b.CreateFAdd(b.CreateFMul(pdx,pdx),
                    b.CreateFAdd(b.CreateFMul(pdy,pdy), b.CreateFMul(pdz,pdz))));
                auto* pin = b.CreateFDiv(fc(1.0f), pl);
                auto* oox = b.CreateFAdd(OX, b.CreateFAdd(
                    b.CreateFMul(b.CreateFMul(u, afov), RX),
                    b.CreateFMul(b.CreateFMul(v, afov), UX)));
                auto* ooy = b.CreateFAdd(OY, b.CreateFAdd(
                    b.CreateFMul(b.CreateFMul(u, afov), RY),
                    b.CreateFMul(b.CreateFMul(v, afov), UY)));
                auto* ooz = b.CreateFAdd(OZ, b.CreateFAdd(
                    b.CreateFMul(b.CreateFMul(u, afov), RZ),
                    b.CreateFMul(b.CreateFMul(v, afov), UZ)));
                ox = b.CreateSelect(is_o, oox, OX);
                oy = b.CreateSelect(is_o, ooy, OY);
                oz = b.CreateSelect(is_o, ooz, OZ);
                dx = b.CreateSelect(is_o, FDX, b.CreateFMul(pdx, pin));
                dy = b.CreateSelect(is_o, FDY, b.CreateFMul(pdy, pin));
                dz = b.CreateSelect(is_o, FDZ, b.CreateFMul(pdz, pin));
            };

            llvm::Value* cox[4]; llvm::Value* coy[4]; llvm::Value* coz[4];
            llvm::Value* cdx[4]; llvm::Value* cdy[4]; llvm::Value* cdz[4];
            llvm::Value* cxs[4] = {tx0f, tx1f, tx0f, tx1f};
            llvm::Value* cys[4] = {ty0f, ty0f, ty1f, ty1f};
            for (int c = 0; c < 4; ++c)
                corner(cxs[c], cys[c], cox[c], coy[c], coz[c], cdx[c], cdy[c], cdz[c]);

            auto mn = [&](llvm::Value* a, llvm::Value* b_){
                return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, a, b_); };
            auto mx = [&](llvm::Value* a, llvm::Value* b_){
                return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, a, b_); };

            llvm::Value* acc0 = fc(1e30f);
            llvm::Value* acc1 = fc(-1e30f);
            for (int s = 0; s < S; ++s) {
                float ta = near + st * float(s);
                float tb = ta + st;
                llvm::Value* lo_x=fc(1e30f), *lo_y=fc(1e30f), *lo_z=fc(1e30f);
                llvm::Value* hi_x=fc(-1e30f), *hi_y=fc(-1e30f), *hi_z=fc(-1e30f);
                for (int c = 0; c < 4; ++c) {
                    for (float t : {ta, tb}) {
                        auto* px = b.CreateFAdd(cox[c], b.CreateFMul(cdx[c], fc(t)));
                        auto* py = b.CreateFAdd(coy[c], b.CreateFMul(cdy[c], fc(t)));
                        auto* pz = b.CreateFAdd(coz[c], b.CreateFMul(cdz[c], fc(t)));
                        lo_x = mn(lo_x, px); hi_x = mx(hi_x, px);
                        lo_y = mn(lo_y, py); hi_y = mx(hi_y, py);
                        lo_z = mn(lo_z, pz); hi_z = mx(hi_z, pz);
                    }
                }
                auto* ccx = b.CreateFMul(fc(0.5f), b.CreateFAdd(lo_x, hi_x));
                auto* ccy = b.CreateFMul(fc(0.5f), b.CreateFAdd(lo_y, hi_y));
                auto* ccz = b.CreateFMul(fc(0.5f), b.CreateFAdd(lo_z, hi_z));
                llvm::Value* occ = nullptr;
                if (use_interval) {
                    // occupied if the field interval over the slab AABB spans 0.
                    auto* iv = b.CreateCall(ival_fn,
                        {lo_x, lo_y, lo_z, hi_x, hi_y, hi_z});
                    auto* ivlo = b.CreateExtractValue(iv, {0});
                    auto* ivhi = b.CreateExtractValue(iv, {1});
                    occ = b.CreateAnd(
                        b.CreateFCmpOLE(ivlo, fc(0.0f)),
                        b.CreateFCmpOGE(ivhi, fc(0.0f)));
                } else {
                    // Lipschitz: |scene_sdf(center)| <= L * halfdiag.
                    auto* ex = b.CreateFMul(fc(0.5f), b.CreateFSub(hi_x, lo_x));
                    auto* ey = b.CreateFMul(fc(0.5f), b.CreateFSub(hi_y, lo_y));
                    auto* ez = b.CreateFMul(fc(0.5f), b.CreateFSub(hi_z, lo_z));
                    auto* hd = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt,
                        b.CreateFAdd(b.CreateFMul(ex,ex),
                        b.CreateFAdd(b.CreateFMul(ey,ey), b.CreateFMul(ez,ez))));
                    auto* r = b.CreateFMul(fc(L), hd);
                    auto* fcv = b.CreateCall(sdf_fn, {ccx, ccy, ccz, PARAMS});
                    occ = b.CreateAnd(
                        b.CreateFCmpOLE(b.CreateFSub(fcv, r), fc(0.0f)),
                        b.CreateFCmpOGE(b.CreateFAdd(fcv, r), fc(0.0f)));
                }
                acc0 = b.CreateSelect(occ, mn(acc0, fc(ta)), acc0);
                acc1 = b.CreateSelect(occ, mx(acc1, fc(tb)), acc1);
            }
            float mrg = st;
            tile_t0 = b.CreateFSub(acc0, fc(mrg));
            tile_t1 = b.CreateFAdd(acc1, fc(mrg));
        }
    }

    // ══ py loop ═══════════════════════════════════════════════════════════════
    b.SetInsertPoint(py_cond);
    auto* py_phi = b.CreatePHI(i32(), 2, "py");
    py_phi->addIncoming(TY, entry_bb);
    auto* py_end = b.CreateAdd(TY, TH, "py_end");
    b.CreateCondBr(b.CreateICmpSLT(py_phi, py_end), py_body, exit_bb);

    b.SetInsertPoint(py_body);
    // NDC y ∈ [-1, 1]: uv_y = 1 - 2*(py/ih)
    auto* py_f  = b.CreateSIToFP(py_phi, f32(), "py_f");
    auto* uv_y  = b.CreateFSub(
        llvm::ConstantFP::get(f32(), 1.0f),
        b.CreateFMul(llvm::ConstantFP::get(f32(), 2.0f),
                     b.CreateFDiv(py_f, ih_f, "py_norm"), "py2"), "uv_y");
    b.CreateBr(px_cond);

    // ══ px loop ═══════════════════════════════════════════════════════════════
    b.SetInsertPoint(px_cond);
    auto* px_phi = b.CreatePHI(i32(), 2, "px");
    px_phi->addIncoming(TX, py_body);
    auto* px_end = b.CreateAdd(TX, TW, "px_end");
    b.CreateCondBr(b.CreateICmpSLT(px_phi, px_end), px_body, py_next);

    b.SetInsertPoint(px_body);
    // NDC x ∈ [-aspect, aspect]: uv_x = (2*(px/iw) - 1) * aspect
    auto* px_f  = b.CreateSIToFP(px_phi, f32(), "px_f");
    auto* uv_x  = b.CreateFMul(
        b.CreateFSub(
            b.CreateFMul(llvm::ConstantFP::get(f32(), 2.0f),
                         b.CreateFDiv(px_f, iw_f, "px_norm"), "px2"),
            llvm::ConstantFP::get(f32(), 1.0f), "pxm"),
        aspect, "uv_x");

    // Ray generation — projection-aware via the sign of FOVS:
    //   FOVS > 0 → perspective: rays diverge from the camera position.
    //   FOVS < 0 → orthographic: rays are parallel; the origin sweeps
    //              across the view plane. |FOVS| is the half-height of the
    //              view rectangle in world units.
    auto fcz = llvm::ConstantFP::get(f32(), 0.0f);
    auto is_ortho = b.CreateFCmpOLT(FOVS, fcz, "is_ortho");
    auto abs_fovs = frep::llvm_compat::unary_intrinsic(b,
                        llvm::Intrinsic::fabs, FOVS);

    // Perspective ray dir = forward + uv_x*right*FOVS + uv_y*up*FOVS
    auto rdx_persp = b.CreateFAdd(FDX, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, FOVS), RX),
        b.CreateFMul(b.CreateFMul(uv_y, FOVS), UX)), "rdx_persp");
    auto rdy_persp = b.CreateFAdd(FDY, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, FOVS), RY),
        b.CreateFMul(b.CreateFMul(uv_y, FOVS), UY)), "rdy_persp");
    auto rdz_persp = b.CreateFAdd(FDZ, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, FOVS), RZ),
        b.CreateFMul(b.CreateFMul(uv_y, FOVS), UZ)), "rdz_persp");

    // Orthographic ray dir = forward (already normalized at the caller).
    // Orthographic ray origin = camera_pos + uv_x*abs_fovs*right + uv_y*abs_fovs*up.
    auto ox_ortho = b.CreateFAdd(OX, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, abs_fovs), RX),
        b.CreateFMul(b.CreateFMul(uv_y, abs_fovs), UX)), "ox_ortho");
    auto oy_ortho = b.CreateFAdd(OY, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, abs_fovs), RY),
        b.CreateFMul(b.CreateFMul(uv_y, abs_fovs), UY)), "oy_ortho");
    auto oz_ortho = b.CreateFAdd(OZ, b.CreateFAdd(
        b.CreateFMul(b.CreateFMul(uv_x, abs_fovs), RZ),
        b.CreateFMul(b.CreateFMul(uv_y, abs_fovs), UZ)), "oz_ortho");

    // Select between the two modes.
    auto rdx = b.CreateSelect(is_ortho, FDX, rdx_persp, "rdx_raw");
    auto rdy = b.CreateSelect(is_ortho, FDY, rdy_persp, "rdy_raw");
    auto rdz = b.CreateSelect(is_ortho, FDZ, rdz_persp, "rdz_raw");

    // Effective ray origin per pixel — camera position for perspective,
    // shifted onto the view plane for orthographic.
    auto ox_eff = b.CreateSelect(is_ortho, ox_ortho, OX, "ox_eff");
    auto oy_eff = b.CreateSelect(is_ortho, oy_ortho, OY, "oy_eff");
    auto oz_eff = b.CreateSelect(is_ortho, oz_ortho, OZ, "oz_eff");

    // Normalize the ray direction
    auto rdl2 = b.CreateFAdd(b.CreateFAdd(
        b.CreateFMul(rdx, rdx), b.CreateFMul(rdy, rdy)), b.CreateFMul(rdz, rdz));
    auto rdl  = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, rdl2, "rdl");
    auto rdxn = b.CreateFDiv(rdx, rdl, "rdxn");
    auto rdyn = b.CreateFDiv(rdy, rdl, "rdyn");
    auto rdzn = b.CreateFDiv(rdz, rdl, "rdzn");

    // Optional ray-box near/far clip. Slab method: for each axis compute the two
    // t values where the ray crosses the box planes, take the near/far envelope.
    // t_enter = max over axes of the near crossings, t_exit = min of the far.
    // If the scene is unbounded or the flag is off, t_enter/t_exit default to the
    // usual [0.001, max_dist] so behaviour is unchanged.
    llvm::Value* t_enter0 = llvm::ConstantFP::get(f32(), 0.001f);
    llvm::Value* t_exit0  = llvm::ConstantFP::get(f32(), cfg_.max_dist);
    if (use_bbox_clip) {
        auto slab = [&](llvm::Value* o, llvm::Value* dn, float lo, float hi,
                        llvm::Value*& tmin, llvm::Value*& tmax) {
            // Guard against a near-zero direction component (ray parallel to slab):
            // use a large inv so the crossings land far outside [near,far] and don't
            // wrongly clip. 1/dn with dn floored away from 0 in magnitude.
            auto* inv = b.CreateFDiv(llvm::ConstantFP::get(f32(), 1.0f), dn);
            auto* t1  = b.CreateFMul(b.CreateFSub(llvm::ConstantFP::get(f32(), lo), o), inv);
            auto* t2  = b.CreateFMul(b.CreateFSub(llvm::ConstantFP::get(f32(), hi), o), inv);
            auto* mn  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, t1, t2);
            auto* mx  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, t1, t2);
            tmin = tmin ? frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, tmin, mn) : mn;
            tmax = tmax ? frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, tmax, mx) : mx;
        };
        llvm::Value* tmin = nullptr; llvm::Value* tmax = nullptr;
        slab(ox_eff, rdxn, scene_box.min_x, scene_box.max_x, tmin, tmax);
        slab(oy_eff, rdyn, scene_box.min_y, scene_box.max_y, tmin, tmax);
        slab(oz_eff, rdzn, scene_box.min_z, scene_box.max_z, tmin, tmax);
        // Start no earlier than the default near, and no earlier than box entry.
        t_enter0 = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum,
                       t_enter0, tmin);
        // Stop no later than max_dist, and no later than box exit. If the ray
        // misses the box (tmax < tmin), t_exit0 < t_enter0 and the loop's
        // dist_ok test fails immediately -> instant miss (correct, and fast).
        t_exit0 = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum,
                      t_exit0, tmax);
    }
    // Per-tile Lipschitz cull (computed before the loops) further narrows the
    // march: start no earlier than the first occupied slab, stop no later than
    // the last. If no slab was occupied, tile_t1 < tile_t0 and dist_ok fails
    // immediately -> the whole tile is a fast miss (correct — nothing in view).
    if (tile_t0 && tile_t1) {
        t_enter0 = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum,
                       t_enter0, tile_t0);
        t_exit0  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum,
                       t_exit0, tile_t1);
    }

    b.CreateBr(st_cond);

    // ══ Sphere tracing loop ═══════════════════════════════════════════════════
    b.SetInsertPoint(st_cond);
    auto* step_phi = b.CreatePHI(i32(),   2, "step");
    auto* t_phi    = b.CreatePHI(f32(), 2, "t");
    auto* lastd_phi = b.CreatePHI(f32(), 2, "last_d");
    auto* slen_phi  = b.CreatePHI(f32(), 2, "step_len");
    auto* omega_phi = b.CreatePHI(f32(), 2, "omega");
    step_phi ->addIncoming(b.getInt32(0),                       px_body);
    t_phi    ->addIncoming(t_enter0, px_body);
    lastd_phi->addIncoming(llvm::ConstantFP::get(f32(), 1e30f),  px_body);
    slen_phi ->addIncoming(llvm::ConstantFP::get(f32(), 0.0f),   px_body);
    omega_phi->addIncoming(llvm::ConstantFP::get(f32(), cfg_.over_relax), px_body);

    // step < MAX_STEPS && t < MAX_DIST
    auto* step_ok  = b.CreateICmpSLT(step_phi, b.getInt32(cfg_.max_steps), "step_ok");
    auto* dist_ok  = b.CreateFCmpOLT(t_phi, t_exit0, "dist_ok");
    b.CreateCondBr(b.CreateAnd(step_ok, dist_ok), st_body, miss_bb);

    b.SetInsertPoint(st_body);
    // p = origin + t * rd
    auto* px_pt = b.CreateFAdd(ox_eff, b.CreateFMul(t_phi, rdxn), "px_pt");
    auto* py_pt = b.CreateFAdd(oy_eff, b.CreateFMul(t_phi, rdyn), "py_pt");
    auto* pz_pt = b.CreateFAdd(oz_eff, b.CreateFMul(t_phi, rdzn), "pz_pt");

    auto* d = b.CreateCall(sdf_fn, {px_pt, py_pt, pz_pt, PARAMS}, "d");

    // Enhanced sphere tracing (over-relaxation with an overshoot guard).
    //   radius   = conservative safe distance (safety_factor*d)
    //   forward  = radius*omega (over-relaxed advance)
    //   sorFail  = the previous over-relaxed step jumped past a surface
    //              ((radius+prev_radius) < step_len); back up by step_len*(1-omega)
    //              and latch omega to 1 (classic, tunnel-proof) for the rest of
    //              the ray. lastd_phi holds the previous raw d (also the
    //              grazing-rescue input below).
    auto* fsafety   = llvm::ConstantFP::get(f32(), cfg_.safety_factor);
    auto* fone      = llvm::ConstantFP::get(f32(), 1.0f);
    auto* radius    = b.CreateFMul(d, fsafety, "radius");
    auto* omega_gt1 = b.CreateFCmpOGT(omega_phi, fone, "omega_gt1");
    auto* sum_r     = b.CreateFAdd(radius, lastd_phi, "sum_r");
    auto* overshoot = b.CreateFCmpOLT(sum_r, slen_phi, "overshoot");
    auto* sor_fail  = b.CreateAnd(omega_gt1, overshoot, "sor_fail");
    auto* fwd_step  = b.CreateFMul(radius, omega_phi, "fwd_step");
    auto* bt_step   = b.CreateFMul(slen_phi, b.CreateFSub(fone, omega_phi), "bt_step");
    auto* slen_next = b.CreateSelect(sor_fail, bt_step, fwd_step, "slen_next");
    auto* omega_next= b.CreateSelect(sor_fail, fone,    omega_phi, "omega_next");

    // hit: d < epsilon, but never accept a hit on a backtracking step.
    auto* raw_hit = b.CreateFCmpOLT(d, llvm::ConstantFP::get(f32(), cfg_.epsilon), "raw_hit");
    auto* is_hit  = b.CreateAnd(raw_hit, b.CreateNot(sor_fail), "is_hit");

    auto* t_next    = b.CreateFAdd(t_phi, slen_next, "t_next");
    auto* step_next = b.CreateAdd(step_phi, b.getInt32(1), "step_next");
    step_phi ->addIncoming(step_next,  st_body);
    t_phi    ->addIncoming(t_next,     st_body);
    lastd_phi->addIncoming(d,          st_body);
    slen_phi ->addIncoming(slen_next,  st_body);
    omega_phi->addIncoming(omega_next, st_body);

    b.CreateCondBr(is_hit, hit_bb, st_cond);

    // ══ Hit ═══════════════════════════════════════════════════════════════════
    b.SetInsertPoint(hit_bb);
    // The hit point (t_phi is valid here — it dominates hit_bb)
    auto* hx = b.CreateFAdd(ox_eff, b.CreateFMul(t_phi, rdxn), "hx");
    auto* hy = b.CreateFAdd(oy_eff, b.CreateFMul(t_phi, rdyn), "hy");
    auto* hz = b.CreateFAdd(oz_eff, b.CreateFMul(t_phi, rdzn), "hz");

    // View direction = -ray_dir (toward the eye).
    auto* vx = b.CreateFNeg(rdxn, "vx");
    auto* vy = b.CreateFNeg(rdyn, "vy");
    auto* vz = b.CreateFNeg(rdzn, "vz");

    // Shade the primary hit via the factored-out helper. Results land
    // in a_cr/a_cg/a_cb (reused here as the shade-output allocas).
    b.CreateCall(shade_fn, {hx, hy, hz, vx, vy, vz, a_cr, a_cg, a_cb, PARAMS});
    llvm::Value* hit_r = b.CreateLoad(f32(), a_cr, "hit_r");
    llvm::Value* hit_g = b.CreateLoad(f32(), a_cg, "hit_g");
    llvm::Value* hit_b = b.CreateLoad(f32(), a_cb, "hit_b");

    // ── Reflection bounces ────────────────────────────────────────────────────
    // Mirror the GLSL emitter: cast up to cfg_.max_bounces reflection
    // rays, blending the reflected colour in by a Schlick-Fresnel-
    // weighted throughput. Each bounce re-marches the scene SDF, and on
    // a hit re-shades via shade_fn; on a miss it samples the sky and
    // stops. Implemented as an unrolled chain of basic blocks (bounce
    // count is small and compile-time known).
    if (refl_fn && cfg_.max_bounces > 0) {
        // Working state held in allocas so the unrolled bounce blocks
        // can mutate it without a PHI swarm.
        auto* a_ro_x = b.CreateAlloca(f32(), nullptr, "ro_x");
        auto* a_ro_y = b.CreateAlloca(f32(), nullptr, "ro_y");
        auto* a_ro_z = b.CreateAlloca(f32(), nullptr, "ro_z");
        auto* a_rn_x = b.CreateAlloca(f32(), nullptr, "rn_x");
        auto* a_rn_y = b.CreateAlloca(f32(), nullptr, "rn_y");
        auto* a_rn_z = b.CreateAlloca(f32(), nullptr, "rn_z");
        auto* a_rd_x = b.CreateAlloca(f32(), nullptr, "rd_x");
        auto* a_rd_y = b.CreateAlloca(f32(), nullptr, "rd_y");
        auto* a_rd_z = b.CreateAlloca(f32(), nullptr, "rd_z");
        auto* a_thru = b.CreateAlloca(f32(), nullptr, "throughput");
        auto* a_colr = b.CreateAlloca(f32(), nullptr, "col_r");
        auto* a_colg = b.CreateAlloca(f32(), nullptr, "col_g");
        auto* a_colb = b.CreateAlloca(f32(), nullptr, "col_b");
        // Reflection-shade scratch outputs.
        auto* a_sr = b.CreateAlloca(f32(), nullptr, "s_r");
        auto* a_sg = b.CreateAlloca(f32(), nullptr, "s_g");
        auto* a_sb = b.CreateAlloca(f32(), nullptr, "s_b");
        // Normal scratch for reflection hits.
        auto* a_rnx = b.CreateAlloca(f32(), nullptr, "rnx");
        auto* a_rny = b.CreateAlloca(f32(), nullptr, "rny");
        auto* a_rnz = b.CreateAlloca(f32(), nullptr, "rnz");

        auto fcf = [&](float v){ return llvm::ConstantFP::get(f32(), v); };

        // Need the primary hit's normal for the first reflection. Refetch
        // (cheap; folds with shade_fn's own fetch under inlining).
        b.CreateCall(normal_fn, {hx, hy, hz, a_rnx, a_rny, a_rnz, PARAMS});
        b.CreateStore(hx, a_ro_x);
        b.CreateStore(hy, a_ro_y);
        b.CreateStore(hz, a_ro_z);
        b.CreateStore(b.CreateLoad(f32(), a_rnx), a_rn_x);
        b.CreateStore(b.CreateLoad(f32(), a_rny), a_rn_y);
        b.CreateStore(b.CreateLoad(f32(), a_rnz), a_rn_z);
        b.CreateStore(rdxn, a_rd_x);
        b.CreateStore(rdyn, a_rd_y);
        b.CreateStore(rdzn, a_rd_z);
        b.CreateStore(fcf(1.0f), a_thru);
        b.CreateStore(hit_r, a_colr);
        b.CreateStore(hit_g, a_colg);
        b.CreateStore(hit_b, a_colb);

        // Chain of bounce blocks. Each: lookup reflectivity at origin →
        // if ~0 stop; else Fresnel-weight throughput, reflect, march,
        // shade-or-sky, blend.
        llvm::BasicBlock* refl_done = llvm::BasicBlock::Create(ctx_, "refl_done", fn);

        for (int bnc = 0; bnc < cfg_.max_bounces; ++bnc) {
            auto* bb_body = llvm::BasicBlock::Create(
                ctx_, "refl_b" + std::to_string(bnc), fn);
            auto* bb_next = llvm::BasicBlock::Create(
                ctx_, "refl_n" + std::to_string(bnc), fn);
            b.CreateBr(bb_body);
            b.SetInsertPoint(bb_body);

            auto* ro_x = b.CreateLoad(f32(), a_ro_x);
            auto* ro_y = b.CreateLoad(f32(), a_ro_y);
            auto* ro_z = b.CreateLoad(f32(), a_ro_z);
            auto* rn_x = b.CreateLoad(f32(), a_rn_x);
            auto* rn_y = b.CreateLoad(f32(), a_rn_y);
            auto* rn_z = b.CreateLoad(f32(), a_rn_z);
            auto* rd_x = b.CreateLoad(f32(), a_rd_x);
            auto* rd_y = b.CreateLoad(f32(), a_rd_y);
            auto* rd_z = b.CreateLoad(f32(), a_rd_z);

            auto* refl = b.CreateCall(refl_fn, {ro_x, ro_y, ro_z, PARAMS}, "refl");
            auto* refl_pos = b.CreateFCmpOGT(refl, fcf(0.001f));
            // If reflectivity ~0, jump to done.
            b.CreateCondBr(refl_pos, bb_next, refl_done);

            b.SetInsertPoint(bb_next);
            // Fresnel-Schlick (scalar, F0 = refl): cosi = dot(-rd, n).
            auto* cosi = frep::llvm_compat::max_num(b,
                b.CreateFNeg(b.CreateFAdd(
                    b.CreateFAdd(b.CreateFMul(rd_x, rn_x), b.CreateFMul(rd_y, rn_y)),
                    b.CreateFMul(rd_z, rn_z))),
                fcf(0.0f));
            auto* one_minus = b.CreateFSub(fcf(1.0f), cosi);
            auto* p2 = b.CreateFMul(one_minus, one_minus);
            auto* p4 = b.CreateFMul(p2, p2);
            auto* p5 = b.CreateFMul(p4, one_minus);
            auto* fres = b.CreateFAdd(refl,
                b.CreateFMul(b.CreateFSub(fcf(1.0f), refl), p5));
            auto* thru = b.CreateFMul(b.CreateLoad(f32(), a_thru), fres);
            b.CreateStore(thru, a_thru);

            // Reflect: rd' = rd - 2*(rd·n)*n
            auto* dot_rn = b.CreateFAdd(
                b.CreateFAdd(b.CreateFMul(rd_x, rn_x), b.CreateFMul(rd_y, rn_y)),
                b.CreateFMul(rd_z, rn_z));
            auto* two_dot = b.CreateFMul(fcf(2.0f), dot_rn);
            auto* refl_dx = b.CreateFSub(rd_x, b.CreateFMul(two_dot, rn_x));
            auto* refl_dy = b.CreateFSub(rd_y, b.CreateFMul(two_dot, rn_y));
            auto* refl_dz = b.CreateFSub(rd_z, b.CreateFMul(two_dot, rn_z));
            b.CreateStore(refl_dx, a_rd_x);
            b.CreateStore(refl_dy, a_rd_y);
            b.CreateStore(refl_dz, a_rd_z);

            // March the reflection ray. Origin offset off surface.
            auto* mo_x = b.CreateFAdd(ro_x, b.CreateFMul(rn_x, fcf(0.01f)));
            auto* mo_y = b.CreateFAdd(ro_y, b.CreateFMul(rn_y, fcf(0.01f)));
            auto* mo_z = b.CreateFAdd(ro_z, b.CreateFMul(rn_z, fcf(0.01f)));

            auto* march_cond = llvm::BasicBlock::Create(ctx_, "rm_cond" + std::to_string(bnc), fn);
            auto* march_body = llvm::BasicBlock::Create(ctx_, "rm_body" + std::to_string(bnc), fn);
            auto* march_hit  = llvm::BasicBlock::Create(ctx_, "rm_hit"  + std::to_string(bnc), fn);
            auto* march_miss = llvm::BasicBlock::Create(ctx_, "rm_miss" + std::to_string(bnc), fn);
            auto* march_post = llvm::BasicBlock::Create(ctx_, "rm_post" + std::to_string(bnc), fn);

            b.CreateBr(march_cond);
            b.SetInsertPoint(march_cond);
            auto* rt_phi   = b.CreatePHI(f32(), 2, "rt");
            auto* rstep_phi= b.CreatePHI(i32(), 2, "rstep");
            rt_phi   ->addIncoming(fcf(0.0f), bb_next);
            rstep_phi->addIncoming(b.getInt32(0), bb_next);
            auto* rstep_ok = b.CreateICmpSLT(rstep_phi, b.getInt32(cfg_.max_steps));
            auto* rdist_ok = b.CreateFCmpOLT(rt_phi, fcf(cfg_.max_dist));
            b.CreateCondBr(b.CreateAnd(rstep_ok, rdist_ok), march_body, march_miss);

            b.SetInsertPoint(march_body);
            auto* mpx = b.CreateFAdd(mo_x, b.CreateFMul(rt_phi, refl_dx));
            auto* mpy = b.CreateFAdd(mo_y, b.CreateFMul(rt_phi, refl_dy));
            auto* mpz = b.CreateFAdd(mo_z, b.CreateFMul(rt_phi, refl_dz));
            auto* md  = b.CreateCall(sdf_fn, {mpx, mpy, mpz, PARAMS}, "md");
            auto* mhit = b.CreateFCmpOLT(md, fcf(cfg_.epsilon));
            auto* rt_next = b.CreateFAdd(rt_phi, b.CreateFMul(md, fcf(cfg_.safety_factor)));
            auto* rstep_next = b.CreateAdd(rstep_phi, b.getInt32(1));
            rt_phi   ->addIncoming(rt_next, march_body);
            rstep_phi->addIncoming(rstep_next, march_body);
            b.CreateCondBr(mhit, march_hit, march_cond);

            // Reflection ray hit something — shade it.
            b.SetInsertPoint(march_hit);
            auto* fhx = b.CreateFAdd(mo_x, b.CreateFMul(rt_phi, refl_dx));
            auto* fhy = b.CreateFAdd(mo_y, b.CreateFMul(rt_phi, refl_dy));
            auto* fhz = b.CreateFAdd(mo_z, b.CreateFMul(rt_phi, refl_dz));
            // View dir for the reflected surface = -refl_dir.
            b.CreateCall(shade_fn, {fhx, fhy, fhz,
                b.CreateFNeg(refl_dx), b.CreateFNeg(refl_dy), b.CreateFNeg(refl_dz),
                a_sr, a_sg, a_sb, PARAMS});
            // Update origin/normal for the next bounce.
            b.CreateCall(normal_fn, {fhx, fhy, fhz, a_rnx, a_rny, a_rnz, PARAMS});
            b.CreateStore(fhx, a_ro_x);
            b.CreateStore(fhy, a_ro_y);
            b.CreateStore(fhz, a_ro_z);
            b.CreateStore(b.CreateLoad(f32(), a_rnx), a_rn_x);
            b.CreateStore(b.CreateLoad(f32(), a_rny), a_rn_y);
            b.CreateStore(b.CreateLoad(f32(), a_rnz), a_rn_z);
            b.CreateBr(march_post);

            // Reflection ray escaped — sample sky gradient.
            b.SetInsertPoint(march_miss);
            // s = 0.5 + 0.5 * refl_dir.y
            auto* sky_s = b.CreateFAdd(fcf(0.5f), b.CreateFMul(refl_dy, fcf(0.5f)));
            auto sky_mix = [&](float h, float tp){
                return b.CreateFAdd(fcf(h),
                    b.CreateFMul(fcf(tp - h), sky_s));
            };
            b.CreateStore(sky_mix(cfg_.sky_horizon[0], cfg_.sky_top[0]), a_sr);
            b.CreateStore(sky_mix(cfg_.sky_horizon[1], cfg_.sky_top[1]), a_sg);
            b.CreateStore(sky_mix(cfg_.sky_horizon[2], cfg_.sky_top[2]), a_sb);
            // Zero throughput so no further bounces (loop will see refl=0
            // next iteration anyway, but make the miss terminal).
            b.CreateBr(march_post);

            b.SetInsertPoint(march_post);
            // Blend: col = mix(col, shaded, throughput).
            auto* t = b.CreateLoad(f32(), a_thru);
            auto blend = [&](llvm::Value* acc_ptr, llvm::Value* s_ptr){
                auto* acc = b.CreateLoad(f32(), acc_ptr);
                auto* s   = b.CreateLoad(f32(), s_ptr);
                auto* mixed = b.CreateFAdd(acc,
                    b.CreateFMul(t, b.CreateFSub(s, acc)));
                b.CreateStore(mixed, acc_ptr);
            };
            blend(a_colr, a_sr);
            blend(a_colg, a_sg);
            blend(a_colb, a_sb);

            // Continue to next bounce (b's insert point is march_post;
            // the next loop iteration creates its own bb_body and the
            // CreateBr(bb_body) at the top connects them).
        }

        // After the last bounce, fall through to done.
        b.CreateBr(refl_done);
        b.SetInsertPoint(refl_done);
        hit_r = b.CreateLoad(f32(), a_colr, "final_r");
        hit_g = b.CreateLoad(f32(), a_colg, "final_g");
        hit_b = b.CreateLoad(f32(), a_colb, "final_b");
    }

    // The block that actually branches into write_bb on the hit path —
    // hit_bb when reflections are off, refl_done (current block) when on.
    // The write PHI must reference this, not hit_bb, or the PHI's
    // incoming value won't dominate its edge.
    llvm::BasicBlock* hit_pred_bb = b.GetInsertBlock();
    b.CreateBr(write_bb);

    // ══ Miss (sky) ════════════════════════════════════════════════════════════
    b.SetInsertPoint(miss_bb);
    // Grazing-ray rescue (mirrors the GLSL emitter). A silhouette ray
    // can exhaust max_steps while still just above epsilon, creeping
    // through the thin valley outside the surface. Falling through to
    // the sky here would paint a bright fringe along the silhouette
    // wherever the background is dark. If the last sampled distance was
    // still close to the surface and we hadn't run past max_dist, divert
    // to the hit path instead — the shaded surface is far closer to the
    // truth than the sky. t_phi / lastd_phi dominate this block (both
    // are PHIs in st_cond, which dominates miss_bb).
    {
        auto* near_surf = b.CreateFCmpOLT(
            lastd_phi, llvm::ConstantFP::get(f32(), cfg_.epsilon * 80.0f), "near_surf");
        auto* in_range  = b.CreateFCmpOLT(
            t_phi, llvm::ConstantFP::get(f32(), cfg_.max_dist), "in_range");
        auto* rescue    = b.CreateAnd(near_surf, in_range, "rescue");
        auto* real_miss_bb = llvm::BasicBlock::Create(ctx_, "real_miss", fn);
        b.CreateCondBr(rescue, hit_bb, real_miss_bb);
        b.SetInsertPoint(real_miss_bb);
    }
    // Sky: a soft blue gradient driven by the ray's vertical component.
    // The constants come from cfg_.sky_horizon and cfg_.sky_top, so a
    // Render-tab Sky-colour edit forces a JIT recompile (the tracer
    // config is part of the compiler's structural hash). Mirrors the
    // GLSL emitter's
    //   color = mix(vec3(horizon...), vec3(top...), s);
    // where s = 0.5 + 0.5 * ray_dir.y.
    auto* sky_t = b.CreateFAdd(
        llvm::ConstantFP::get(f32(), 0.5f),
        b.CreateFMul(uv_y, llvm::ConstantFP::get(f32(), 0.5f)),
        "sky_t");
    // mix(horizon, top, t) = horizon + (top - horizon) * t
    auto mix_sky = [&](float h, float tp) -> llvm::Value* {
        return b.CreateFAdd(
            llvm::ConstantFP::get(f32(), h),
            b.CreateFMul(
                llvm::ConstantFP::get(f32(), tp - h),
                sky_t));
    };
    auto* miss_r = mix_sky(cfg_.sky_horizon[0], cfg_.sky_top[0]);
    auto* miss_g = mix_sky(cfg_.sky_horizon[1], cfg_.sky_top[1]);
    auto* miss_b = mix_sky(cfg_.sky_horizon[2], cfg_.sky_top[2]);
    // The sky colour is now computed in the `real_miss` block (miss_bb
    // first does the grazing-rescue branch), so the miss edge into
    // write_bb originates there. Capture the actual predecessor.
    llvm::BasicBlock* miss_pred_bb = b.GetInsertBlock();
    b.CreateBr(write_bb);

    // ══ Write pixel ═══════════════════════════════════════════════════════════
    b.SetInsertPoint(write_bb);
    // PHI for the final color (hit or miss)
    auto* phi_r = b.CreatePHI(f32(), 2, "fr");
    auto* phi_g = b.CreatePHI(f32(), 2, "fg");
    auto* phi_b = b.CreatePHI(f32(), 2, "fb");
    phi_r->addIncoming(hit_r,  hit_pred_bb);  phi_r->addIncoming(miss_r, miss_pred_bb);
    phi_g->addIncoming(hit_g,  hit_pred_bb);  phi_g->addIncoming(miss_g, miss_pred_bb);
    phi_b->addIncoming(hit_b,  hit_pred_bb);  phi_b->addIncoming(miss_b, miss_pred_bb);

    // Clamp [0,1]
    auto clamp1 = [&](llvm::Value* v) -> llvm::Value* {
        return frep::llvm_compat::min_num(b,
            frep::llvm_compat::max_num(b, v, llvm::ConstantFP::get(f32(), 0.0f)),
            llvm::ConstantFP::get(f32(), 1.0f));
    };

    // Color pipeline: clamp to [0, 1], no gamma encoding. The GLSL
    // emitter (core/gpu/glsl_emitter.cpp) writes its `imageStore`
    // output with only `clamp(color, 0, 1)` — no sqrt, no tone map.
    // We deliberately mirror that behaviour here so the CPU JIT and
    // both GPU paths produce visually identical results on
    // identical scenes.
    //
    // The historical sqrt(color) gamma approximation produced
    // perceptually-darker output (because the buffer is then read by
    // QImage as if it were already sRGB-encoded and re-decoded for
    // display). The GPU-style look — slightly washed but with more
    // saturated mid-tones — is now the reference for all three
    // paths. Verified on NVIDIA GTX 1050 Ti in v4.0.5.
    auto finalize = [&](llvm::Value* v) -> llvm::Value* {
        return clamp1(v);
    };

    // Pixel index: (py * iw + px) * 4
    // Everything in i32 for simplicity (safe up to ~23K*23K images)
    auto* pix_i = b.CreateAdd(b.CreateMul(py_phi, IW), px_phi, "pix_i");
    auto* base  = b.CreateMul(pix_i, b.getInt32(4), "base");

    // Write RGBA — float* + offset
    auto store_ch = [&](llvm::Value* val, int off) {
        auto* idx = b.CreateAdd(base, b.getInt32(off));
        auto* ptr = b.CreateGEP(f32(), out, idx, "gep");
        b.CreateStore(val, ptr);
    };
    store_ch(finalize(phi_r), 0);
    store_ch(finalize(phi_g), 1);
    store_ch(finalize(phi_b), 2);
    store_ch(llvm::ConstantFP::get(f32(), 1.0f), 3); // alpha

    b.CreateBr(px_next);

    // ══ px_next ═══════════════════════════════════════════════════════════════
    b.SetInsertPoint(px_next);
    auto* px_inc = b.CreateAdd(px_phi, b.getInt32(1), "px_inc");
    px_phi->addIncoming(px_inc, px_next);
    b.CreateBr(px_cond);

    // ══ py_next ═══════════════════════════════════════════════════════════════
    b.SetInsertPoint(py_next);
    auto* py_inc = b.CreateAdd(py_phi, b.getInt32(1), "py_inc");
    py_phi->addIncoming(py_inc, py_next);
    b.CreateBr(py_cond);

    // ══ Exit ══════════════════════════════════════════════════════════════════
    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_scene_material
// void scene_material(float x,y,z,
//                     float* ar, float* ag, float* ab,
//                     float* rough, float* metal)
//
// BVH-accelerated material lookup. Returns the per-object material parameters
// of the nearest object at the query point.
//
// The linear version was O(n) — every object is SDF-evaluated for each hit pixel.
// The BVH version: for every internal node we emit an AABB containment test;
// if the point is outside the subtree's AABB → skip the whole subtree.
// On average O(log n) SDF evaluations.
//
// Since n is fixed at compile time, we unroll the BVH tree statically in IR
// as branches (each internal node → conditional, each leaf → SDF eval).
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_material(const SceneGraph& scene,
                                                  llvm::Function* sdf_fn) {
    // signature: (x, y, z, out_ar, out_ag, out_ab, out_rough, out_metal, params)
    auto* fty = llvm::FunctionType::get(vd(),
        {f32(), f32(), f32(),
         fptr(), fptr(), fptr(),    // out albedo r, g, b
         fptr(), fptr(),            // out roughness, metallic
         fptr()},                   // params buffer (incremental mode)
        false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "scene_material", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(entry);

    auto it = fn->arg_begin();
    auto* x      = &*it++; x->setName("x");
    auto* y      = &*it++; y->setName("y");
    auto* z      = &*it++; z->setName("z");
    auto* oar    = &*it++; oar->setName("out_ar");
    auto* oag    = &*it++; oag->setName("out_ag");
    auto* oab    = &*it++; oab->setName("out_ab");
    auto* orough = &*it++; orough->setName("out_rough");
    auto* ometal = &*it++; ometal->setName("out_metal");
    auto* pb     = &*it++; pb->setName("params");

    // Build the BVH
    BVH bvh = BVH::build(scene);

    // Gather every textured material's pixels into one flat RGBA8 buffer and
    // embed it as a private module-level constant. The Texture pattern case
    // samples it directly via a GEP into this global — no runtime binding and
    // no extra function argument, since texture data is fixed at codegen time
    // (unlike per-frame params). Mirrors the GPU emitter's tex_pixels
    // aggregation so both paths sample identical bytes. Per-entry byte offset
    // is recorded so each Texture case bakes (offset,w,h) as IR constants.
    texture_pixels_.clear();
    std::unordered_map<const void*, int> tex_offset_by_entry;
    for (const auto& e : bvh.entries()) {
        if (e.pattern == Material::Pattern::Texture && !e.texture_rgba.empty()) {
            tex_offset_by_entry[&e] = static_cast<int>(texture_pixels_.size());
            texture_pixels_.insert(texture_pixels_.end(),
                e.texture_rgba.begin(), e.texture_rgba.end());
        }
    }
    llvm::GlobalVariable* tex_global = nullptr;
    llvm::ArrayType*      tex_global_arr_ty_ = nullptr;
    if (!texture_pixels_.empty()) {
        auto* arr_ty = llvm::ArrayType::get(b_i8t(), texture_pixels_.size());
        tex_global_arr_ty_ = arr_ty;
        std::vector<llvm::Constant*> bytes;
        bytes.reserve(texture_pixels_.size());
        for (std::uint8_t v : texture_pixels_)
            bytes.push_back(llvm::ConstantInt::get(b_i8t(), v));
        auto* init = llvm::ConstantArray::get(arr_ty, bytes);
        tex_global = new llvm::GlobalVariable(
            *mod_, arr_ty, /*isConstant=*/true,
            llvm::GlobalValue::PrivateLinkage, init, "scene_textures");
    }

    // The result accumulates in allocas (best_d, best_r/g/b, best_rough,
    // best_metal) — this allows simple stores from different branches
    // without a PHI swarm.
    auto* a_d = b.CreateAlloca(f32(), nullptr, "best_d");
    auto* a_r = b.CreateAlloca(f32(), nullptr, "best_r");
    auto* a_g = b.CreateAlloca(f32(), nullptr, "best_g");
    auto* a_b = b.CreateAlloca(f32(), nullptr, "best_b");
    auto* a_rough = b.CreateAlloca(f32(), nullptr, "best_rough");
    auto* a_metal = b.CreateAlloca(f32(), nullptr, "best_metal");
    b.CreateStore(llvm::ConstantFP::get(f32(), 1e9f), a_d);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.5f), a_r);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.5f), a_g);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.5f), a_b);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.5f), a_rough);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.0f), a_metal);

    auto* exit = llvm::BasicBlock::Create(ctx_, "exit", fn);

    if (bvh.empty()) {
        b.CreateBr(exit);
    } else {
        // Per-object codegen needs the params buffer so primitives load
        // their per-node parameters in Incremental mode.
        CgCtx cctx = make_cgctx(b, pb);

        // ── Procedural pattern emitter ──────────────────────────────────────
        // Emits an RGB triple at the hit point (x, y, z) by blending between
        // albedo (primary) and albedo2 (secondary) according to a position-
        // dependent function. All math is in IR, so the pattern is evaluated
        // per pixel by the JIT'd render_tile.
        auto emit_pattern = [&](const BVH::Entry& e)
            -> std::array<llvm::Value*, 3>
        {
            auto k1 = [&](int i) {
                return llvm::ConstantFP::get(f32(), e.albedo[i]);
            };
            auto k2 = [&](int i) {
                return llvm::ConstantFP::get(f32(), e.albedo2[i]);
            };
            auto fc = [&](float v) {
                return llvm::ConstantFP::get(f32(), v);
            };

            if (e.pattern == Material::Pattern::Solid) {
                return {k1(0), k1(1), k1(2)};
            }

            auto blend = [&](llvm::Value* t) -> std::array<llvm::Value*, 3> {
                // out = albedo * (1-t) + albedo2 * t
                auto one_minus_t = b.CreateFSub(fc(1.0f), t);
                std::array<llvm::Value*, 3> out;
                for (int i = 0; i < 3; ++i) {
                    auto a = b.CreateFMul(k1(i), one_minus_t);
                    auto b2 = b.CreateFMul(k2(i), t);
                    out[i] = b.CreateFAdd(a, b2);
                }
                return out;
            };

            llvm::Value* t = nullptr;
            auto scale = fc(e.pattern_scale);

            switch (e.pattern) {
                case Material::Pattern::Solid:
                    return {k1(0), k1(1), k1(2)};  // unreachable; handled above

                case Material::Pattern::Checker: {
                    // Classic 3D checker: ((floor(s*x) + floor(s*y) +
                    //                       floor(s*z)) & 1) ? 1 : 0
                    auto sx = b.CreateFMul(x, scale);
                    auto sy = b.CreateFMul(y, scale);
                    auto sz = b.CreateFMul(z, scale);
                    auto fl = [&](llvm::Value* v) {
                        return frep::llvm_compat::unary_intrinsic(
                            b, llvm::Intrinsic::floor, v);
                    };
                    auto sum = b.CreateFAdd(b.CreateFAdd(fl(sx), fl(sy)), fl(sz));
                    auto isum = b.CreateFPToSI(sum, i32());
                    auto parity = b.CreateAnd(isum,
                                  llvm::ConstantInt::get(i32(), 1));
                    t = b.CreateSIToFP(parity, f32());
                    break;
                }

                case Material::Pattern::Stripes: {
                    // Bands along Y. period = 1/scale; binary toggle.
                    auto sy = b.CreateFMul(y, scale);
                    auto fl = frep::llvm_compat::unary_intrinsic(
                        b, llvm::Intrinsic::floor, sy);
                    auto ifl = b.CreateFPToSI(fl, i32());
                    auto parity = b.CreateAnd(ifl,
                                  llvm::ConstantInt::get(i32(), 1));
                    t = b.CreateSIToFP(parity, f32());
                    break;
                }

                case Material::Pattern::GradientY: {
                    // Smooth gradient: t = clamp((y / scale + 1) / 2, 0, 1)
                    auto y_over = b.CreateFDiv(y, scale);
                    auto plus1  = b.CreateFAdd(y_over, fc(1.0f));
                    auto half   = b.CreateFMul(plus1, fc(0.5f));
                    auto lo = frep::llvm_compat::max_num(b, half, fc(0.0f));
                    t  = frep::llvm_compat::min_num(b, lo,   fc(1.0f));
                    break;
                }

                case Material::Pattern::Noise: {
                    // Simple hash-based noise. Quantize position to cells of
                    // size 1/scale; hash the integer coords; map to [0, 1].
                    auto sx = b.CreateFMul(x, scale);
                    auto sy = b.CreateFMul(y, scale);
                    auto sz = b.CreateFMul(z, scale);
                    auto fl = [&](llvm::Value* v) {
                        return frep::llvm_compat::unary_intrinsic(
                            b, llvm::Intrinsic::floor, v);
                    };
                    auto ix = b.CreateFPToSI(fl(sx), i32());
                    auto iy = b.CreateFPToSI(fl(sy), i32());
                    auto iz = b.CreateFPToSI(fl(sz), i32());

                    // Murmur-like integer hash, kept simple for IR.
                    auto K1 = llvm::ConstantInt::get(i32(), 0x9E3779B9);
                    auto K2 = llvm::ConstantInt::get(i32(), 0x85EBCA6B);
                    auto K3 = llvm::ConstantInt::get(i32(), 0xC2B2AE35);
                    auto h = b.CreateMul(ix, K1);
                    h = b.CreateXor(h, b.CreateMul(iy, K2));
                    h = b.CreateXor(h, b.CreateMul(iz, K3));
                    // Mix and mask to [0, 2^23) so SI->FP is exact.
                    h = b.CreateXor(h, b.CreateLShr(h,
                            llvm::ConstantInt::get(i32(), 16)));
                    auto mask = llvm::ConstantInt::get(i32(), 0x7FFFFF);
                    h = b.CreateAnd(h, mask);
                    auto hf = b.CreateSIToFP(h, f32());
                    // / (2^23) → [0, 1)
                    t = b.CreateFMul(hf, fc(1.0f / 8388608.0f));
                    break;
                }

                case Material::Pattern::Texture: {
                    // Image texture via triplanar nearest sampling, matching
                    // the GLSL emitter byte-for-byte:
                    //   n = abs(tex_normal(p))^4, normalized
                    //   q = p * scale
                    //   sample zy, xz, xy planes (nearest), blend by n
                    // tex_normal is the scene SDF's central-difference normal
                    // (h=1e-3), same as the shader's _tex_normal.
                    auto itoff = tex_offset_by_entry.find(&e);
                    if (!tex_global || !sdf_fn ||
                        itoff == tex_offset_by_entry.end()) {
                        return {k1(0), k1(1), k1(2)};  // no texture → albedo
                    }
                    int off_px = itoff->second / 4;  // byte offset → pixel index
                    int tw = e.texture_width, th = e.texture_height;

                    // central-difference normal from the scene SDF
                    auto sdf_at = [&](llvm::Value* px, llvm::Value* py,
                                      llvm::Value* pz) -> llvm::Value* {
                        return b.CreateCall(sdf_fn, {px, py, pz, pb});
                    };
                    auto* hc = fc(1e-3f);
                    auto axis_d = [&](int axis) -> llvm::Value* {
                        llvm::Value* p[3] = {x, y, z};
                        llvm::Value* pp[3] = {p[0], p[1], p[2]};
                        llvm::Value* pm[3] = {p[0], p[1], p[2]};
                        pp[axis] = b.CreateFAdd(p[axis], hc);
                        pm[axis] = b.CreateFSub(p[axis], hc);
                        return b.CreateFSub(sdf_at(pp[0], pp[1], pp[2]),
                                            sdf_at(pm[0], pm[1], pm[2]));
                    };
                    auto* dnx = axis_d(0);
                    auto* dny = axis_d(1);
                    auto* dnz = axis_d(2);
                    // abs
                    auto fabs_ = [&](llvm::Value* v) {
                        return b.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
                            mod_.get(), llvm::Intrinsic::fabs, {f32()}), {v});
                    };
                    // Normalize the central-difference normal before taking
                    // abs()^4, exactly like the GLSL _tex_normal/triplanar:
                    // the raw differences are O(h)=O(1e-3), so n^4 underflows
                    // to ~1e-12 and the 1e-6 epsilon then dominates wsum,
                    // collapsing every weight to ~0 (black). Normalizing first
                    // puts the components in [0,1] so the weights are
                    // meaningful (and matches the shader exactly).
                    auto* nlen2 = b.CreateFAdd(
                        b.CreateFAdd(b.CreateFMul(dnx, dnx),
                                     b.CreateFMul(dny, dny)),
                        b.CreateFAdd(b.CreateFMul(dnz, dnz), fc(1e-20f)));
                    auto* nrt = b.CreateCall(
                        llvm::Intrinsic::getOrInsertDeclaration(
                            mod_.get(), llvm::Intrinsic::sqrt, {f32()}), {nlen2});
                    auto* ninv = b.CreateFDiv(fc(1.0f), nrt);
                    auto* anx = fabs_(b.CreateFMul(dnx, ninv));
                    auto* any = fabs_(b.CreateFMul(dny, ninv));
                    auto* anz = fabs_(b.CreateFMul(dnz, ninv));
                    // ^4
                    auto p4 = [&](llvm::Value* v) {
                        auto* v2 = b.CreateFMul(v, v);
                        return b.CreateFMul(v2, v2);
                    };
                    auto* wx = p4(anx);
                    auto* wy = p4(any);
                    auto* wz = p4(anz);
                    auto* wsum = b.CreateFAdd(b.CreateFAdd(wx, wy),
                                              b.CreateFAdd(wz, fc(1e-6f)));
                    wx = b.CreateFDiv(wx, wsum);
                    wy = b.CreateFDiv(wy, wsum);
                    wz = b.CreateFDiv(wz, wsum);

                    // q = p * scale
                    auto* sc = fc(e.pattern_scale);
                    auto* qx = b.CreateFMul(x, sc);
                    auto* qy = b.CreateFMul(y, sc);
                    auto* qz = b.CreateFMul(z, sc);

                    // sample one channel set from a uv pair, nearest, with
                    // fract + clamp, reading RGBA8 bytes from tex_global.
                    auto frac = [&](llvm::Value* v) {
                        auto* fl = b.CreateCall(
                            llvm::Intrinsic::getOrInsertDeclaration(
                                mod_.get(), llvm::Intrinsic::floor, {f32()}), {v});
                        return b.CreateFSub(v, fl);
                    };
                    auto sample_uv = [&](llvm::Value* u, llvm::Value* v)
                        -> std::array<llvm::Value*, 3> {
                        auto* fu = frac(u);
                        auto* fv = frac(v);
                        // iu = clamp(int(fu*tw), 0, tw-1)
                        auto* iu = b.CreateFPToSI(b.CreateFMul(fu, fc((float)tw)), i32());
                        auto* iv = b.CreateFPToSI(b.CreateFMul(fv, fc((float)th)), i32());
                        auto clampi = [&](llvm::Value* iv2, int hi) {
                            auto* lo0 = b.CreateSelect(
                                b.CreateICmpSLT(iv2, b.getInt32(0)), b.getInt32(0), iv2);
                            return b.CreateSelect(
                                b.CreateICmpSGT(lo0, b.getInt32(hi)), b.getInt32(hi), lo0);
                        };
                        iu = clampi(iu, tw - 1);
                        iv = clampi(iv, th - 1);
                        // idx = off_px + iv*tw + iu ; byte base = idx*4
                        auto* idx = b.CreateAdd(b.getInt32(off_px),
                            b.CreateAdd(b.CreateMul(iv, b.getInt32(tw)), iu));
                        auto* base = b.CreateMul(idx, b.getInt32(4));
                        auto load_ch = [&](int c) -> llvm::Value* {
                            // tex_global is [N x i8]*. Index into the array
                            // with a leading 0 (step into the aggregate) then
                            // the byte offset — a flat i8* GEP on the array
                            // pointer would index whole arrays, not bytes.
                            auto* gep = b.CreateInBoundsGEP(
                                tex_global_arr_ty_, tex_global,
                                {b.getInt32(0), b.CreateAdd(base, b.getInt32(c))});
                            auto* byte = b.CreateLoad(b_i8t(), gep);
                            auto* zext = b.CreateZExt(byte, i32());
                            return b.CreateFMul(b.CreateSIToFP(zext, f32()),
                                                fc(1.0f / 255.0f));
                        };
                        return {load_ch(0), load_ch(1), load_ch(2)};
                    };

                    // tex_global is [N x i8]; GEP needs a pointer to its first
                    // element. Decay the array global to an i8* once.
                    // (CreateGEP on the global with i8 element type indexes
                    // bytes directly since the global is i8[].)
                    auto cx = sample_uv(qz, qy);  // zy plane
                    auto cy = sample_uv(qx, qz);  // xz plane
                    auto cz = sample_uv(qx, qy);  // xy plane

                    std::array<llvm::Value*, 3> out;
                    for (int c = 0; c < 3; ++c) {
                        auto* a = b.CreateFMul(cx[c], wx);
                        auto* bb = b.CreateFMul(cy[c], wy);
                        auto* cc = b.CreateFMul(cz[c], wz);
                        out[c] = b.CreateFAdd(b.CreateFAdd(a, bb), cc);
                    }
                    return out;
                }
            }
            return blend(t);
        };

        // Recursive emission of the BVH traversal.
        // visit() emits code for one BVH node; continues into the `cont` BB.
        int bb_counter = 0;
        std::function<void(const BVHNode*, llvm::BasicBlock*)> visit =
            [&](const BVHNode* node, llvm::BasicBlock* cont) {
            if (node->is_leaf()) {
                // Leaf: SDF eval of the object, select it if it is closer.
                const auto& e = bvh.entries()[node->object_index];
                auto* d = e.geom->codegen(cctx, x, y, z);
                auto* cur_d = b.CreateLoad(f32(), a_d, "cur_d");
                auto* closer = b.CreateFCmpOLT(d, cur_d, "closer");
                auto* nd = b.CreateSelect(closer, d, cur_d);
                b.CreateStore(nd, a_d);

                // Procedural albedo: emit pattern eval at the hit point.
                // For Solid this folds to three constant loads; for the
                // patterned variants it produces a couple of FP ops.
                auto rgb = emit_pattern(e);

                b.CreateStore(b.CreateSelect(closer, rgb[0],
                    b.CreateLoad(f32(), a_r)), a_r);
                b.CreateStore(b.CreateSelect(closer, rgb[1],
                    b.CreateLoad(f32(), a_g)), a_g);
                b.CreateStore(b.CreateSelect(closer, rgb[2],
                    b.CreateLoad(f32(), a_b)), a_b);
                b.CreateStore(b.CreateSelect(closer,
                    llvm::ConstantFP::get(f32(), e.roughness),
                    b.CreateLoad(f32(), a_rough)), a_rough);
                b.CreateStore(b.CreateSelect(closer,
                    llvm::ConstantFP::get(f32(), e.metallic),
                    b.CreateLoad(f32(), a_metal)), a_metal);
                b.CreateBr(cont);
                return;
            }

            // Internal node: AABB containment test.
            // If the point is NOT in the AABB → skip the whole subtree.
            // We add a small margin (eps) — the hit point from sphere tracing
            // may be slightly outside the exact AABB due to epsilon tolerance.
            const auto& box = node->box;
            const float m = 0.05f;  // AABB margin
            auto in_range = [&](llvm::Value* v, float lo, float hi) {
                auto* ge = b.CreateFCmpOGE(v, llvm::ConstantFP::get(f32(), lo - m));
                auto* le = b.CreateFCmpOLE(v, llvm::ConstantFP::get(f32(), hi + m));
                return b.CreateAnd(ge, le);
            };
            auto* inside =
                b.CreateAnd(b.CreateAnd(in_range(x, box.min_x, box.max_x),
                                        in_range(y, box.min_y, box.max_y)),
                            in_range(z, box.min_z, box.max_z), "in_aabb");

            auto* visit_bb = llvm::BasicBlock::Create(
                ctx_, "bvh_visit_" + std::to_string(bb_counter++), fn);
            b.CreateCondBr(inside, visit_bb, cont);

            // visit_bb: check both children, then → cont
            b.SetInsertPoint(visit_bb);
            auto* after_left = llvm::BasicBlock::Create(
                ctx_, "bvh_mid_" + std::to_string(bb_counter++), fn);
            visit(node->left.get(), after_left);

            b.SetInsertPoint(after_left);
            visit(node->right.get(), cont);
        };

        // Start the traversal; finish in exit.
        b.SetInsertPoint(entry);
        // entry already has stores — continue the traversal from here.
        // But visit() expects us to be at the right insert point. We make
        // a separate start BB for cleanliness.
        auto* traverse_start = llvm::BasicBlock::Create(ctx_, "traverse", fn);
        b.CreateBr(traverse_start);
        b.SetInsertPoint(traverse_start);
        visit(bvh.root(), exit);
    }

    // exit: load the result and store it into the out parameters.
    b.SetInsertPoint(exit);
    b.CreateStore(b.CreateLoad(f32(), a_r),     oar);
    b.CreateStore(b.CreateLoad(f32(), a_g),     oag);
    b.CreateStore(b.CreateLoad(f32(), a_b),     oab);
    b.CreateStore(b.CreateLoad(f32(), a_rough), orough);
    b.CreateStore(b.CreateLoad(f32(), a_metal), ometal);
    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_scene_reflectivity
// float scene_reflectivity(float x, float y, float z, params)
//
// Compact analogue of emit_scene_material: walks the same BVH but tracks
// only the mirror reflectivity of the nearest object. Used by the
// reflection bounce loop in emit_tracer to decide how much of the
// reflected colour to blend in. CPU parity with the GLSL emitter's
// scene_reflectivity(). Only called when cfg_.max_bounces > 0.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_reflectivity(const SceneGraph& scene) {
    auto* fty = llvm::FunctionType::get(f32(),
        {f32(), f32(), f32(), fptr()}, false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "scene_reflectivity", mod_.get());
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(entry);

    auto it = fn->arg_begin();
    auto* x  = &*it++; x->setName("x");
    auto* y  = &*it++; y->setName("y");
    auto* z  = &*it++; z->setName("z");
    auto* pb = &*it++; pb->setName("params");

    BVH bvh = BVH::build(scene);

    auto* a_d    = b.CreateAlloca(f32(), nullptr, "best_d");
    auto* a_refl = b.CreateAlloca(f32(), nullptr, "best_refl");
    b.CreateStore(llvm::ConstantFP::get(f32(), 1e9f), a_d);
    b.CreateStore(llvm::ConstantFP::get(f32(), 0.0f), a_refl);

    auto* exit = llvm::BasicBlock::Create(ctx_, "exit", fn);

    if (bvh.empty()) {
        b.CreateBr(exit);
    } else {
        CgCtx cctx = make_cgctx(b, pb);
        int bb_counter = 0;
        std::function<void(const BVHNode*, llvm::BasicBlock*)> visit =
            [&](const BVHNode* node, llvm::BasicBlock* cont) {
            if (node->is_leaf()) {
                const auto& e = bvh.entries()[node->object_index];
                auto* d = e.geom->codegen(cctx, x, y, z);
                auto* cur_d = b.CreateLoad(f32(), a_d, "cur_d");
                auto* closer = b.CreateFCmpOLT(d, cur_d, "closer");
                b.CreateStore(b.CreateSelect(closer, d, cur_d), a_d);
                b.CreateStore(b.CreateSelect(closer,
                    llvm::ConstantFP::get(f32(), e.reflectivity),
                    b.CreateLoad(f32(), a_refl)), a_refl);
                b.CreateBr(cont);
                return;
            }
            const auto& box = node->box;
            const float m = 0.05f;
            auto in_range = [&](llvm::Value* v, float lo, float hi) {
                auto* ge = b.CreateFCmpOGE(v, llvm::ConstantFP::get(f32(), lo - m));
                auto* le = b.CreateFCmpOLE(v, llvm::ConstantFP::get(f32(), hi + m));
                return b.CreateAnd(ge, le);
            };
            auto* inside =
                b.CreateAnd(b.CreateAnd(in_range(x, box.min_x, box.max_x),
                                        in_range(y, box.min_y, box.max_y)),
                            in_range(z, box.min_z, box.max_z), "in_aabb");
            auto* visit_bb = llvm::BasicBlock::Create(
                ctx_, "refl_visit_" + std::to_string(bb_counter++), fn);
            b.CreateCondBr(inside, visit_bb, cont);
            b.SetInsertPoint(visit_bb);
            auto* after_left = llvm::BasicBlock::Create(
                ctx_, "refl_mid_" + std::to_string(bb_counter++), fn);
            visit(node->left.get(), after_left);
            b.SetInsertPoint(after_left);
            visit(node->right.get(), cont);
        };
        b.SetInsertPoint(entry);
        auto* traverse_start = llvm::BasicBlock::Create(ctx_, "traverse", fn);
        b.CreateBr(traverse_start);
        b.SetInsertPoint(traverse_start);
        visit(bvh.root(), exit);
    }

    b.SetInsertPoint(exit);
    b.CreateRet(b.CreateLoad(f32(), a_refl));

    verify_fn(fn);
    return fn;
}
// int32 scene_pick(float ox,oy,oz, float dx,dy,dz)
//
// Casts a ray from (ox,oy,oz) along (dx,dy,dz), sphere-traces and returns
// the index of the hit object (in the order of traversal of visible objects).
// Returns -1 if the ray hits nothing within max_dist.
//
// Algorithm: on every step we compute the per-object SDF of all objects,
// take the minimum (= scene SDF) AND the index of the object that yields it.
// When scene SDF < epsilon → hit; return the saved index.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_scene_pick(const SceneGraph& scene) {
    auto* fty = llvm::FunctionType::get(i32(),
        {f32(), f32(), f32(), f32(), f32(), f32()}, false);
    auto* fn  = llvm::Function::Create(fty,
                    llvm::Function::ExternalLinkage, "scene_pick", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);
    fn->addFnAttr(llvm::Attribute::WillReturn);

    auto it = fn->arg_begin();
    auto* OX = &*it++; OX->setName("ox");
    auto* OY = &*it++; OY->setName("oy");
    auto* OZ = &*it++; OZ->setName("oz");
    auto* DX = &*it++; DX->setName("dx");
    auto* DY = &*it++; DY->setName("dy");
    auto* DZ = &*it++; DZ->setName("dz");

    // Gather the visible objects in a stable order — the index we return
    // corresponds to this order. The GUI uses the same order to find the id.
    std::vector<const FRepNode*> geoms;
    for (auto& [id, obj] : scene.objects())
        if (obj.visible) geoms.push_back(obj.geometry.get());

    auto* entry   = llvm::BasicBlock::Create(ctx_, "entry",   fn);
    auto* st_cond = llvm::BasicBlock::Create(ctx_, "st_cond", fn);
    auto* st_body = llvm::BasicBlock::Create(ctx_, "st_body", fn);
    auto* hit_bb  = llvm::BasicBlock::Create(ctx_, "hit",     fn);
    auto* miss_bb = llvm::BasicBlock::Create(ctx_, "miss",    fn);

    llvm::IRBuilder<> b(entry);

    // Empty scene → always a miss.
    if (geoms.empty()) {
        b.CreateRet(ic(b, -1));
        return fn;
    }

    llvm::BasicBlock* pre_loop = entry;  // BB that branches to st_cond

    // ── Early-exit ray-AABB test against the scene's combined AABB ───────────
    // Compute the union AABB of all visible objects and emit a slab-method
    // ray-AABB intersection: if the ray misses the bounding box entirely,
    // there is no need to sphere-trace at all. This is a pure speedup for
    // rays that fall outside the scene; rays that hit the box still go
    // through the existing linear loop below (which is already O(n) per
    // step but n is typically small for picker queries).
    //
    // Skip the early-exit when any object has a non-finite AABB (e.g. an
    // infinite Plane) — in that case the union is unbounded.
    bool all_finite = true;
    float bmin[3] = { 1e30f,  1e30f,  1e30f};
    float bmax[3] = {-1e30f, -1e30f, -1e30f};
    for (const auto* g : geoms) {
        auto a = g->aabb();
        if (!std::isfinite(a.min_x) || !std::isfinite(a.max_x)) {
            all_finite = false; break;
        }
        bmin[0] = std::min(bmin[0], a.min_x);
        bmin[1] = std::min(bmin[1], a.min_y);
        bmin[2] = std::min(bmin[2], a.min_z);
        bmax[0] = std::max(bmax[0], a.max_x);
        bmax[1] = std::max(bmax[1], a.max_y);
        bmax[2] = std::max(bmax[2], a.max_z);
    }
    if (all_finite) {
        // Small margin so we don't miss surface intersections right on the
        // box face (sphere-tracing stops a tiny epsilon away from the
        // surface).
        const float m = 0.1f;
        for (int k = 0; k < 3; ++k) { bmin[k] -= m; bmax[k] += m; }

        // Slab method: for each axis, compute the t-interval over which
        // the ray is inside the slab. Intersection of the three intervals
        // gives the t-range inside the box. If the range is empty, miss.
        //
        // To avoid division-by-zero when a direction component is 0, we
        // use a large reciprocal — the corresponding t1/t2 then sit at
        // +-1e30, which the min/max ops will correctly fold away if the
        // origin is inside the slab.
        auto safe_recip = [&](llvm::Value* d) {
            // Use the bit-trick reciprocal but guarded: where |d| is tiny
            // (near zero), return a huge magnitude with the right sign.
            // Implemented as: r = 1/d if |d| >= 1e-20 else copysign(1e30, d).
            auto* abs_d = frep::llvm_compat::unary_intrinsic(b,
                            llvm::Intrinsic::fabs, d);
            auto* tiny = b.CreateFCmpOLT(abs_d,
                            llvm::ConstantFP::get(f32(), 1e-20f));
            auto* huge = llvm::ConstantFP::get(f32(), 1e30f);
            auto* one_over_d = b.CreateFDiv(
                                llvm::ConstantFP::get(f32(), 1.0f), d);
            return b.CreateSelect(tiny, huge, one_over_d);
        };
        auto* inv_dx = safe_recip(DX);
        auto* inv_dy = safe_recip(DY);
        auto* inv_dz = safe_recip(DZ);

        auto axis_range = [&](llvm::Value* o, llvm::Value* inv_d,
                              float lo, float hi,
                              llvm::Value*& t_near, llvm::Value*& t_far) {
            auto* t1 = b.CreateFMul(b.CreateFSub(
                            llvm::ConstantFP::get(f32(), lo), o), inv_d);
            auto* t2 = b.CreateFMul(b.CreateFSub(
                            llvm::ConstantFP::get(f32(), hi), o), inv_d);
            t_near = frep::llvm_compat::min_num(b, t1, t2);
            t_far  = frep::llvm_compat::max_num(b, t1, t2);
        };

        llvm::Value *xn, *xf, *yn, *yf, *zn, *zf;
        axis_range(OX, inv_dx, bmin[0], bmax[0], xn, xf);
        axis_range(OY, inv_dy, bmin[1], bmax[1], yn, yf);
        axis_range(OZ, inv_dz, bmin[2], bmax[2], zn, zf);

        // t_enter = max of the per-axis near; t_exit = min of the per-axis far.
        auto* t_enter = frep::llvm_compat::max_num(b,
                            frep::llvm_compat::max_num(b, xn, yn), zn);
        auto* t_exit  = frep::llvm_compat::min_num(b,
                            frep::llvm_compat::min_num(b, xf, yf), zf);

        // Miss if t_exit < max(t_enter, 0) or t_enter > max_dist.
        auto* t_enter_clamped = frep::llvm_compat::max_num(b, t_enter,
                                    llvm::ConstantFP::get(f32(), 0.0f));
        auto* ok1 = b.CreateFCmpOGE(t_exit, t_enter_clamped, "ray_aabb_ok1");
        auto* ok2 = b.CreateFCmpOLE(t_enter,
                            llvm::ConstantFP::get(f32(), cfg_.max_dist),
                            "ray_aabb_ok2");
        auto* hits_box = b.CreateAnd(ok1, ok2, "hits_box");

        auto* enter_loop_bb = llvm::BasicBlock::Create(ctx_,
                                "enter_loop", fn);
        b.CreateCondBr(hits_box, enter_loop_bb, miss_bb);
        b.SetInsertPoint(enter_loop_bb);
        pre_loop = enter_loop_bb;
    }

    b.CreateBr(st_cond);

    // ── Sphere tracing loop ───────────────────────────────────────────────────
    // PHI: step_count, t (distance along the ray)
    b.SetInsertPoint(st_cond);
    auto* step_phi = b.CreatePHI(i32(), 2, "step");
    auto* t_phi    = b.CreatePHI(f32(), 2, "t");
    step_phi->addIncoming(ic(b, 0), pre_loop);
    t_phi   ->addIncoming(llvm::ConstantFP::get(f32(), 0.0f), pre_loop);

    auto* step_ok = b.CreateICmpSLT(step_phi, ic(b, cfg_.max_steps), "step_ok");
    auto* dist_ok = b.CreateFCmpOLT(t_phi,
                       llvm::ConstantFP::get(f32(), cfg_.max_dist), "dist_ok");
    b.CreateCondBr(b.CreateAnd(step_ok, dist_ok), st_body, miss_bb);

    // ── st_body: compute position, per-object SDF, min + argmin ───────────────
    b.SetInsertPoint(st_body);
    auto* px = b.CreateFAdd(OX, b.CreateFMul(t_phi, DX), "px");
    auto* py = b.CreateFAdd(OY, b.CreateFMul(t_phi, DY), "py");
    auto* pz = b.CreateFAdd(OZ, b.CreateFMul(t_phi, DZ), "pz");

    // scene_pick is a standalone helper (called from the GUI on click).
    // It does not receive a params buffer; in Incremental mode the picker
    // uses the parameter defaults baked into the IR. This is acceptable
    // because picking is rare and a small staleness on object boundaries
    // is irrelevant (object IDs do not change with parameter values).
    CgCtx cctx = make_cgctx(b, nullptr);

    // best_d = +inf, best_idx = -1; then fold over every object.
    llvm::Value* best_d   = llvm::ConstantFP::get(f32(), 1e9f);
    llvm::Value* best_idx = ic(b, -1);
    for (std::size_t i = 0; i < geoms.size(); ++i) {
        auto* d = geoms[i]->codegen(cctx, px, py, pz);
        auto* closer = b.CreateFCmpOLT(d, best_d, "closer");
        best_d   = b.CreateSelect(closer, d, best_d, "best_d");
        best_idx = b.CreateSelect(closer, ic(b, static_cast<int>(i)),
                                  best_idx, "best_idx");
    }

    // Hit if best_d < epsilon.
    auto* is_hit = b.CreateFCmpOLT(best_d,
                       llvm::ConstantFP::get(f32(), cfg_.epsilon), "is_hit");

    // Next step: t += best_d * safety_factor
    auto* t_step = b.CreateFMul(best_d,
                       llvm::ConstantFP::get(f32(), cfg_.safety_factor));
    auto* t_next = b.CreateFAdd(t_phi, t_step, "t_next");
    auto* s_next = b.CreateAdd(step_phi, ic(b, 1), "s_next");
    step_phi->addIncoming(s_next, st_body);
    t_phi   ->addIncoming(t_next, st_body);

    b.CreateCondBr(is_hit, hit_bb, st_cond);

    // ── hit: return the index of the nearest object ───────────────────────────
    // best_idx is defined in st_body — but st_body is the only predecessor
    // of hit_bb, so we can use it directly via PHI.
    b.SetInsertPoint(hit_bb);
    auto* hit_idx = b.CreatePHI(i32(), 1, "hit_idx");
    hit_idx->addIncoming(best_idx, st_body);
    b.CreateRet(hit_idx);

    // ── miss: -1 ──────────────────────────────────────────────────────────────
    b.SetInsertPoint(miss_bb);
    b.CreateRet(ic(b, -1));

    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_render_tile — main method
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_render_tile(const SceneGraph& scene,
                                               SceneSdfMode mode) {
    // Build a Union of all visible objects for scene_sdf.
    std::vector<FRepNode::Ptr> geoms;
    for (auto& [id, obj] : scene.objects())
        if (obj.visible) geoms.push_back(obj.geometry);

    if (geoms.empty())
        throw std::runtime_error("emit_render_tile: empty scene");

    auto root = union_all(geoms);   // keep geoms for optional split path


    // Adaptive raymarch step. safety_factor < 1 is only needed when the
    // scene's distance field isn't a true Euclidean SDF — i.e. it
    // contains CSG / plugin / mesh / custom-expr nodes, OR we wrapped
    // multiple objects in a Union above (which is itself a CSG min). A
    // scene that is a single primitive (optionally affine/twist
    // transformed) is a true SDF, so a full step is safe and ~20%
    // faster. We raise safety_factor to 1.0 for that case and restore
    // the configured value afterwards so the change is local to this
    // compile. The grazing-ray rescue still backs up either choice.
    const bool multiple_objects = (root->kind == NodeKind::Union)
                                  || (root->kind == NodeKind::Scene);
    const bool needs_safety = multiple_objects
                              || node_requires_safety_step(*root);
    // RAII guard restores the configured safety_factor when this compile
    // returns, so the override stays local. Captures the original value
    // before we touch cfg_.
    struct SafetyRestore {
        float& slot; float val;
        ~SafetyRestore() { slot = val; }
    } restore{cfg_.safety_factor, cfg_.safety_factor};
    if (!needs_safety)
        cfg_.safety_factor = 1.0f;

    // The march-loop scene_sdf can be built three ways (see SceneSdfMode).
    // Only this scene_sdf varies; the AD-gradient SDF stays inlined (it
    // has a different signature), so Split/Guarded measure the partial
    // effect on the march loop, which dominates step count.
    llvm::Function* sdf_fn = nullptr;
    switch (mode) {
        case SceneSdfMode::Split:   sdf_fn = emit_scene_sdf_split(geoms);   break;
        case SceneSdfMode::Guarded: sdf_fn = emit_scene_sdf_guarded(geoms); break;
        case SceneSdfMode::Inlined:
        default:                    sdf_fn = emit_scene_sdf(*root);         break;
    }
    auto* sdf_grad_fn = emit_scene_sdf_grad(*root);
    auto* normal_fn   = emit_scene_normal(sdf_grad_fn);
    auto* shader_fn   = emit_shader();
    auto* mat_fn      = emit_scene_material(scene, sdf_fn);
    return emit_tracer(scene, sdf_fn, normal_fn, shader_fn, mat_fn);
}

// ── Approach B, Stage 1: CPU vector (W-wide) render_tile emitted in IR ────────
// Mirrors the verified host packet tracer (core/render/cpu_trace.hpp): W rays
// per packet, masked march, central-difference normals, single-light Lambert.
// Behind FREP4_VEC_RENDER; the scalar path is untouched.
llvm::Function* SceneCodegen::emit_render_tile_vec(const SceneGraph& scene,
                                                   SceneSdfMode mode,
                                                   unsigned W) {
    std::vector<FRepNode::Ptr> geoms;
    for (auto& [id, obj] : scene.objects())
        if (obj.visible) geoms.push_back(obj.geometry);
    if (geoms.empty()) throw std::runtime_error("emit_render_tile_vec: empty scene");
    auto root = union_all(geoms);
    (void)mode;

    // Scene AABB (compile-time), mirroring emit_tracer, for the ray-box clip: the
    // march must start at the box entry, not t=1e-3, or the extra empty-space
    // steps accumulate float rounding and shift the hit point on sensitive fields
    // (the gyroid), which then shifts the shadow — the concave-scene penumbra gap.
    bool scene_bounded = false; FRepNode::AABB scene_box{};
    for (auto& g : geoms) {
        FRepNode::AABB bx = g->aabb();
        bool ok = std::isfinite(bx.min_x)&&std::isfinite(bx.min_y)&&std::isfinite(bx.min_z)&&
                  std::isfinite(bx.max_x)&&std::isfinite(bx.max_y)&&std::isfinite(bx.max_z);
        if (!ok) { scene_bounded = false; break; }
        if (!scene_bounded) { scene_box = bx; scene_bounded = true; }
        else {
            scene_box.min_x = std::min(scene_box.min_x, bx.min_x);
            scene_box.min_y = std::min(scene_box.min_y, bx.min_y);
            scene_box.min_z = std::min(scene_box.min_z, bx.min_z);
            scene_box.max_x = std::max(scene_box.max_x, bx.max_x);
            scene_box.max_y = std::max(scene_box.max_y, bx.max_y);
            scene_box.max_z = std::max(scene_box.max_z, bx.max_z);
        }
    }
    if (scene_bounded) {
        float m = 0.05f * std::max({scene_box.max_x - scene_box.min_x,
                                    scene_box.max_y - scene_box.min_y,
                                    scene_box.max_z - scene_box.min_z, 0.1f});
        scene_box.min_x -= m; scene_box.min_y -= m; scene_box.min_z -= m;
        scene_box.max_x += m; scene_box.max_y += m; scene_box.max_z += m;
    }
    const bool use_clip = scene_bounded && cfg_.bbox_clip;

    auto* F32 = llvm::Type::getFloatTy(ctx_);
    auto* I32 = llvm::Type::getInt32Ty(ctx_);
    auto* VT  = llvm::VectorType::get(F32, W, false);
    auto* VI  = llvm::VectorType::get(I32, W, false);
    auto* PF  = llvm::PointerType::getUnqual(ctx_);

    std::vector<llvm::Type*> at = { PF, I32,I32,I32,I32, I32,I32,
                                    F32,F32,F32, F32,F32,F32, F32,F32,F32, F32,F32,F32, F32 };
    auto* fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), at, false);
    auto* fn  = llvm::Function::Create(fty, llvm::Function::ExternalLinkage,
                                       "render_tile", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    auto A = fn->arg_begin();
    llvm::Value* OUT = &*A++;
    llvm::Value* TX = &*A++, *TY = &*A++, *TW = &*A++, *TH = &*A++;
    llvm::Value* IW = &*A++, *IH = &*A++;
    llvm::Value* Ox = &*A++, *Oy = &*A++, *Oz = &*A++;
    llvm::Value* Dx = &*A++, *Dy = &*A++, *Dz = &*A++;
    llvm::Value* Rx = &*A++, *Ry = &*A++, *Rz = &*A++;
    llvm::Value* Ux = &*A++, *Uy = &*A++, *Uz = &*A++;
    llvm::Value* FOV = &*A++;

    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
    llvm::IRBuilder<> b(entry);
    auto splat = [&](llvm::Value* s){ return b.CreateVectorSplat(W, s); };
    auto fcv   = [&](float v){ return b.CreateVectorSplat(W, llvm::ConstantFP::get(F32, v)); };
    auto sdf = [&](llvm::Value* X, llvm::Value* Y, llvm::Value* Z) -> llvm::Value* {
        CgCtx cg{ctx_, *mod_, b}; cg.width = W;
        return root->codegen(cg, X, Y, Z);
    };
    auto vlen = [&](llvm::Value* x, llvm::Value* y, llvm::Value* z){
        auto* s = b.CreateFAdd(b.CreateFMul(x,x),
                    b.CreateFAdd(b.CreateFMul(y,y), b.CreateFMul(z,z)));
        return frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, s);
    };

    auto* iwf = b.CreateSIToFP(IW, F32), *ihf = b.CreateSIToFP(IH, F32);
    auto* aspV = splat(b.CreateFDiv(iwf, ihf)), *fovV = splat(FOV);
    auto* iwV = splat(iwf), *ihV = splat(ihf);
    auto* OxV = splat(Ox), *OyV = splat(Oy), *OzV = splat(Oz);
    auto* DxV = splat(Dx), *DyV = splat(Dy), *DzV = splat(Dz);
    auto* RxV = splat(Rx), *RyV = splat(Ry), *RzV = splat(Rz);
    auto* UxV = splat(Ux), *UyV = splat(Uy), *UzV = splat(Uz);
    auto* total = b.CreateMul(TW, TH);

    auto* pre = b.GetInsertBlock();
    auto* ploop = llvm::BasicBlock::Create(ctx_, "packet", fn);
    auto* pdone = llvm::BasicBlock::Create(ctx_, "done", fn);
    b.CreateBr(ploop);
    b.SetInsertPoint(ploop);
    auto* base = b.CreatePHI(I32, 2, "base");
    base->addIncoming(b.getInt32(0), pre);

    llvm::Value* PXv = llvm::UndefValue::get(VI);
    llvm::Value* PYv = llvm::UndefValue::get(VI);
    for (unsigned l = 0; l < W; ++l) {
        auto* idxRaw = b.CreateAdd(base, b.getInt32((int)l));
        auto* over = b.CreateICmpSGE(idxRaw, total);
        auto* idx  = b.CreateSelect(over, b.CreateSub(total, b.getInt32(1)), idxRaw);
        PXv = b.CreateInsertElement(PXv, b.CreateAdd(TX, b.CreateSRem(idx, TW)), l);
        PYv = b.CreateInsertElement(PYv, b.CreateAdd(TY, b.CreateSDiv(idx, TW)), l);
    }
    // Match emit_tracer's ray setup EXACTLY: uv_x = (2*px/iw - 1)*aspect,
    // uv_y = 1 - 2*py/ih (pixel corner, no +0.5), dir = D + uv_x*FOVS*R + uv_y*FOVS*U.
    auto* pxf = b.CreateSIToFP(PXv, VT), *pyf = b.CreateSIToFP(PYv, VT);
    auto* sx = b.CreateFMul(b.CreateFMul(
                 b.CreateFSub(b.CreateFDiv(b.CreateFMul(fcv(2.0f), pxf), iwV), fcv(1.0f)),
                 aspV), fovV);
    auto* sy = b.CreateFMul(
                 b.CreateFSub(fcv(1.0f), b.CreateFDiv(b.CreateFMul(fcv(2.0f), pyf), ihV)),
                 fovV);
    llvm::Value* DXp = b.CreateFAdd(DxV, b.CreateFAdd(b.CreateFMul(sx,RxV), b.CreateFMul(sy,UxV)));
    llvm::Value* DYp = b.CreateFAdd(DyV, b.CreateFAdd(b.CreateFMul(sx,RyV), b.CreateFMul(sy,UyV)));
    llvm::Value* DZp = b.CreateFAdd(DzV, b.CreateFAdd(b.CreateFMul(sx,RzV), b.CreateFMul(sy,UzV)));
    auto* inv = b.CreateFDiv(fcv(1.0f), vlen(DXp,DYp,DZp));
    DXp = b.CreateFMul(DXp,inv); DYp = b.CreateFMul(DYp,inv); DZp = b.CreateFMul(DZp,inv);

    // Match emit_render_tile's step: a true single-primitive SDF marches with a
    // full step (safety 1.0), CSG/plugin/mesh/custom trees use the reduced
    // safety_factor. Using 0.85 everywhere shifted the hit point on the sphere
    // just enough for the pow(ndoth,6) specular to differ visibly.
    const bool multiple = (root->kind == NodeKind::Union) || (root->kind == NodeKind::Scene);
    const bool needs_safety = multiple || node_requires_safety_step(*root);
    const float EPS = cfg_.epsilon;
    const float SAFE = needs_safety ? cfg_.safety_factor : 1.0f;
    const int MAXS = cfg_.max_steps;

    // Ray-box clip (slab method): t starts at the box entry, stops at the box
    // exit — same as emit_tracer, so the step sequence (and thus the hit point)
    // coincides. A missed ray gets tExit < tEnter and the march terminates fast.
    llvm::Value* tEnter = fcv(1e-3f);
    llvm::Value* tExit  = fcv(cfg_.max_dist);
    if (use_clip) {
        auto vmn = [&](llvm::Value* a, llvm::Value* c){ return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, a, c); };
        auto vmx = [&](llvm::Value* a, llvm::Value* c){ return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, a, c); };
        auto slab = [&](llvm::Value* o, llvm::Value* dn, float lo, float hi,
                        llvm::Value*& tmn, llvm::Value*& tmx){
            auto* invd = b.CreateFDiv(fcv(1.0f), dn);
            auto* t1 = b.CreateFMul(b.CreateFSub(fcv(lo), o), invd);
            auto* t2 = b.CreateFMul(b.CreateFSub(fcv(hi), o), invd);
            auto* mn = vmn(t1,t2), *mx = vmx(t1,t2);
            tmn = tmn ? vmx(tmn,mn) : mn;
            tmx = tmx ? vmn(tmx,mx) : mx;
        };
        llvm::Value *tmn=nullptr, *tmx=nullptr;
        slab(OxV, DXp, scene_box.min_x, scene_box.max_x, tmn, tmx);
        slab(OyV, DYp, scene_box.min_y, scene_box.max_y, tmn, tmx);
        slab(OzV, DZp, scene_box.min_z, scene_box.max_z, tmn, tmx);
        tEnter = vmx(tEnter, tmn);
        tExit  = vmn(tExit, tmx);
    }
    auto* preM = b.GetInsertBlock();
    auto* mloop = llvm::BasicBlock::Create(ctx_, "march", fn);
    auto* mbody = llvm::BasicBlock::Create(ctx_, "mbody", fn);
    auto* mend  = llvm::BasicBlock::Create(ctx_, "mend", fn);
    b.CreateBr(mloop);
    b.SetInsertPoint(mloop);
    auto* Tphi = b.CreatePHI(VT, 2, "t");
    auto* live = b.CreatePHI(VT, 2, "live");
    auto* step = b.CreatePHI(I32, 2, "s");
    Tphi->addIncoming(tEnter, preM);
    live->addIncoming(fcv(1.0f), preM);
    step->addIncoming(b.getInt32(0), preM);
    // Early-exit: stop the packet as soon as every lane has hit or escaped
    // (reduce-max of the live mask). Without this the packet runs the full
    // step budget even after all rays converge — the dominant Stage-1 cost.
    auto* anyLive = b.CreateFCmpOGT(
        b.CreateIntrinsic(llvm::Intrinsic::vector_reduce_fmax, {VT}, {live}),
        llvm::ConstantFP::get(F32, 0.5f));
    auto* go = b.CreateAnd(b.CreateICmpSLT(step, b.getInt32(MAXS)), anyLive);
    b.CreateCondBr(go, mbody, mend);

    b.SetInsertPoint(mbody);
    auto* fval = sdf(b.CreateFAdd(OxV, b.CreateFMul(Tphi,DXp)),
                     b.CreateFAdd(OyV, b.CreateFMul(Tphi,DYp)),
                     b.CreateFAdd(OzV, b.CreateFMul(Tphi,DZp)));
    auto* newHit = b.CreateAnd(b.CreateFCmpOLT(fval, fcv(EPS)),
                               b.CreateFCmpOGT(live, fcv(0.5f)));
    auto* liveNext = b.CreateSelect(newHit, fcv(0.0f), live);
    auto* stepAmt = b.CreateSelect(b.CreateFCmpOGT(liveNext, fcv(0.5f)),
                        b.CreateFMul(fcv(SAFE), fval), fcv(0.0f));
    auto* Tnext = b.CreateFAdd(Tphi, stepAmt);
    liveNext = b.CreateSelect(b.CreateFCmpOGE(Tnext, tExit), fcv(0.0f), liveNext);
    Tphi->addIncoming(Tnext, mbody);
    live->addIncoming(liveNext, mbody);
    step->addIncoming(b.CreateAdd(step, b.getInt32(1)), mbody);
    b.CreateBr(mloop);

    b.SetInsertPoint(mend);
    auto* finalHit = b.CreateAnd(b.CreateFCmpOLE(live, fcv(0.5f)),
                                 b.CreateFCmpOLT(Tphi, tExit));
    auto* HX = b.CreateFAdd(OxV, b.CreateFMul(Tphi,DXp));
    auto* HY = b.CreateFAdd(OyV, b.CreateFMul(Tphi,DYp));
    auto* HZ = b.CreateFAdd(OzV, b.CreateFMul(Tphi,DZp));
    // Stage 2: analytic normals via forward-mode AD (same as the scalar path),
    // three seeded gradient evaluations instead of six finite differences. The
    // node codegen_grad vectorises through CgCtx.width, so the whole gradient is
    // W-wide; .dot is the directional derivative along the seed = one component.
    auto gradN = [&](llvm::Value* X, llvm::Value* Y, llvm::Value* Z, int axis) -> llvm::Value* {
        CgCtx cg{ctx_, *mod_, b}; cg.width = W;
        FRepNode::DualVal dx{X, axis==0 ? fcv(1.0f) : fcv(0.0f)},
                          dy{Y, axis==1 ? fcv(1.0f) : fcv(0.0f)},
                          dz{Z, axis==2 ? fcv(1.0f) : fcv(0.0f)};
        return root->codegen_grad(cg, dx, dy, dz).dot;
    };
    auto* nx = gradN(HX,HY,HZ,0);
    auto* ny = gradN(HX,HY,HZ,1);
    auto* nz = gradN(HX,HY,HZ,2);
    auto* ninv = b.CreateFDiv(fcv(1.0f), vlen(nx,ny,nz));
    nx = b.CreateFMul(nx,ninv); ny = b.CreateFMul(ny,ninv); nz = b.CreateFMul(nz,ninv);
    // Ambient occlusion (matches emit_ambient_occlusion): 5 SDF samples along
    // the normal, ao = 1 - 0.6*clamp(sum((i*0.15) - sdf(p+n*i*0.15)) * 0.5^(i-1)).
    llvm::Value* aoTotal = fcv(0.0f); float aow = 1.0f;
    for (int i = 1; i <= 5; ++i) {
        float dist = i * 0.15f;
        auto* hh = sdf(b.CreateFAdd(HX, b.CreateFMul(nx, fcv(dist))),
                       b.CreateFAdd(HY, b.CreateFMul(ny, fcv(dist))),
                       b.CreateFAdd(HZ, b.CreateFMul(nz, fcv(dist))));
        aoTotal = b.CreateFAdd(aoTotal, b.CreateFMul(b.CreateFSub(fcv(dist), hh), fcv(aow)));
        aow *= 0.5f;
    }
    auto* aoCl = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum,
                   frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, aoTotal, fcv(0.0f)), fcv(1.0f));
    auto* aoV = b.CreateFSub(fcv(1.0f), b.CreateFMul(aoCl, fcv(0.6f)));
    // Stage 3: the scalar path's Blinn-Phong shading with the default material
    // (albedo 0.8, roughness 0.5 -> shininess 6, metallic 0) and the scene's
    // first point light (fallback {5,10,5}). ambient = albedo*0.08; per light
    // Lo += (albedo*ndotl + 0.04*pow(ndoth,shin)). Shadows (Stage 4) and multi-
    // light are not applied yet, so lit surfaces match but shadowed ones differ.
    auto vnorm3 = [&](llvm::Value*& x, llvm::Value*& y, llvm::Value*& z){
        auto* i = b.CreateFDiv(fcv(1.0f), vlen(x,y,z));
        x = b.CreateFMul(x,i); y = b.CreateFMul(y,i); z = b.CreateFMul(z,i);
    };
    llvm::Value* Lx = b.CreateFSub(fcv(5.0f), HX), *Ly = b.CreateFSub(fcv(10.0f), HY), *Lz = b.CreateFSub(fcv(5.0f), HZ);
    auto* shMaxT = vlen(Lx,Ly,Lz);                 // distance to the light (before normalize)
    vnorm3(Lx,Ly,Lz);

    // Stage 4: vector soft shadow (Inigo Quilez, single ray — shadow_samples=1).
    // March from hit+n*0.01 toward the light: k=min(k, 16*h/t); h<0.002 => full
    // shadow; stop at the light. A second masked packet march, only hit lanes live.
    auto* shOx = b.CreateFAdd(HX, b.CreateFMul(nx, fcv(0.01f)));
    auto* shOy = b.CreateFAdd(HY, b.CreateFMul(ny, fcv(0.01f)));
    auto* shOz = b.CreateFAdd(HZ, b.CreateFMul(nz, fcv(0.01f)));
    auto* shPre = b.GetInsertBlock();
    auto* shLive0 = b.CreateSelect(finalHit, fcv(1.0f), fcv(0.0f));  // emit in shPre (mend)
    auto* shloop = llvm::BasicBlock::Create(ctx_, "shmarch", fn);
    auto* shbody = llvm::BasicBlock::Create(ctx_, "shbody", fn);
    auto* shendB = llvm::BasicBlock::Create(ctx_, "shend", fn);
    b.CreateBr(shloop);
    b.SetInsertPoint(shloop);
    auto* shT = b.CreatePHI(VT, 2, "sht");
    auto* shK = b.CreatePHI(VT, 2, "shk");
    auto* shLive = b.CreatePHI(VT, 2, "shlive");
    auto* shStep = b.CreatePHI(I32, 2, "shs");
    shT->addIncoming(fcv(0.05f), shPre);
    shK->addIncoming(fcv(1.0f), shPre);
    shLive->addIncoming(shLive0, shPre);
    shStep->addIncoming(b.getInt32(0), shPre);
    auto* shAny = b.CreateFCmpOGT(
        b.CreateIntrinsic(llvm::Intrinsic::vector_reduce_fmax, {VT}, {shLive}),
        llvm::ConstantFP::get(F32, 0.5f));
    b.CreateCondBr(b.CreateAnd(b.CreateICmpSLT(shStep, b.getInt32(64)), shAny), shbody, shendB);
    b.SetInsertPoint(shbody);
    auto* shH = sdf(b.CreateFAdd(shOx, b.CreateFMul(shT, Lx)),
                    b.CreateFAdd(shOy, b.CreateFMul(shT, Ly)),
                    b.CreateFAdd(shOz, b.CreateFMul(shT, Lz)));
    auto* shLiveB = b.CreateFCmpOGT(shLive, fcv(0.5f));
    auto* fullSh  = b.CreateAnd(b.CreateFCmpOLT(shH, fcv(0.002f)), shLiveB);
    auto* shKfull = b.CreateSelect(fullSh, fcv(0.0f), shK);
    auto* ratio   = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum,
                        b.CreateFDiv(b.CreateFMul(fcv(16.0f), shH), shT), fcv(0.0f));
    auto* shKmin  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, shKfull, ratio);
    auto* shKnext = b.CreateSelect(shLiveB, shKmin, shK);
    auto* shTnext = b.CreateFAdd(shT, b.CreateFMul(fcv(SAFE), shH));  // t += h*safety, like the scalar
    auto* shLiveNext = b.CreateSelect(b.CreateOr(fullSh, b.CreateFCmpOGE(shTnext, shMaxT)),
                                      fcv(0.0f), shLive);
    shT->addIncoming(shTnext, shbody);
    shK->addIncoming(shKnext, shbody);
    shLive->addIncoming(shLiveNext, shbody);
    shStep->addIncoming(b.CreateAdd(shStep, b.getInt32(1)), shbody);
    b.CreateBr(shloop);
    b.SetInsertPoint(shendB);
    auto* shadow = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum,
                     frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, shK, fcv(0.0f)), fcv(1.0f));
    llvm::Value* Vx = b.CreateFSub(OxV, HX), *Vy = b.CreateFSub(OyV, HY), *Vz = b.CreateFSub(OzV, HZ);
    vnorm3(Vx,Vy,Vz);
    llvm::Value* Hx = b.CreateFAdd(Lx,Vx), *Hy = b.CreateFAdd(Ly,Vy), *Hz = b.CreateFAdd(Lz,Vz);
    vnorm3(Hx,Hy,Hz);
    auto mx0 = [&](llvm::Value* v){ return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, v, fcv(0.0f)); };
    auto* ndotl = mx0(b.CreateFAdd(b.CreateFMul(nx,Lx), b.CreateFAdd(b.CreateFMul(ny,Ly), b.CreateFMul(nz,Lz))));
    auto* ndoth = mx0(b.CreateFAdd(b.CreateFMul(nx,Hx), b.CreateFAdd(b.CreateFMul(ny,Hy), b.CreateFMul(nz,Hz))));
    // Cook-Torrance GGX — the default shading model (not Blinn-Phong). Matches
    // emit_shader: D = a2/(pi*(ndoth^2*(a2-1)+1)^2); G via Schlick k=(r+1)^2/8;
    // F = 0.04 + 0.96*(1-VdotH)^5; spec = min(F*D/(4*gv*gl)*PI, 8*PI). Default
    // material: albedo 0.8, roughness 0.5, metallic 0.
    auto* rgh   = fcv(0.5f);
    auto* alpha = b.CreateFMul(rgh, rgh);
    auto* alpha2= b.CreateFMul(alpha, alpha);
    auto* ndotv = mx0(b.CreateFAdd(b.CreateFMul(nx,Vx), b.CreateFAdd(b.CreateFMul(ny,Vy), b.CreateFMul(nz,Vz))));
    auto* vdoth = mx0(b.CreateFAdd(b.CreateFMul(Vx,Hx), b.CreateFAdd(b.CreateFMul(Vy,Hy), b.CreateFMul(Vz,Hz))));
    auto* ndoth2= b.CreateFMul(ndoth, ndoth);
    auto* inr   = b.CreateFAdd(b.CreateFMul(ndoth2, b.CreateFSub(alpha2, fcv(1.0f))), fcv(1.0f));
    auto* inr2  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, b.CreateFMul(inr,inr), fcv(1e-7f));
    auto* Dg    = b.CreateFDiv(alpha2, b.CreateFMul(fcv(3.14159265f), inr2));
    auto* kgr   = b.CreateFAdd(rgh, fcv(1.0f));
    auto* kg    = b.CreateFDiv(b.CreateFMul(kgr,kgr), fcv(8.0f));
    auto* omk   = b.CreateFSub(fcv(1.0f), kg);
    auto* gvs   = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, b.CreateFAdd(b.CreateFMul(ndotv,omk), kg), fcv(1e-5f));
    auto* gls   = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, b.CreateFAdd(b.CreateFMul(ndotl,omk), kg), fcv(1e-5f));
    auto* visd  = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, b.CreateFMul(b.CreateFMul(fcv(4.0f), gvs), gls), fcv(1e-5f));
    auto* dgd   = b.CreateFDiv(Dg, visd);
    auto* omvh  = b.CreateFSub(fcv(1.0f), vdoth);
    auto* vh2   = b.CreateFMul(omvh,omvh);
    auto* vh5   = b.CreateFMul(b.CreateFMul(vh2,vh2), omvh);
    auto* Fr    = b.CreateFAdd(fcv(0.04f), b.CreateFMul(fcv(0.96f), vh5));
    auto* front = b.CreateFCmpOGT(ndotl, fcv(0.0f));
    auto* specRaw = b.CreateFMul(b.CreateFMul(Fr, dgd), fcv(3.14159265f));
    auto* specC = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, specRaw, fcv(8.0f * 3.14159265f));
    auto* spec  = b.CreateSelect(front, specC, fcv(0.0f));
    auto* diffB = b.CreateFMul(b.CreateFSub(fcv(1.0f), Fr), fcv(0.8f));  // (1-F)*albedo
    // lit = ambient*ao + (diffuse + specular) * ndotl * shadow
    auto* lit = b.CreateFAdd(b.CreateFMul(fcv(0.064f), aoV),
                  b.CreateFMul(b.CreateFMul(b.CreateFAdd(diffB, spec), ndotl), shadow));
    // Sky gradient on miss, matching the scalar path: mix(horizon, top, s),
    // s = 0.5 + 0.5*ray_dir.y; horizon {0.6,0.7,0.85}, top {0.4,0.5,0.7}.
    // Final colour clamped to [0,1] (the scalar path does no gamma, just clamp).
    // The scalar sky uses uv_y (NDC pixel y = 1 - 2*py/ih), not the normalized
    // ray-dir y — matching it removes a systematic gradient offset over the whole
    // background.
    auto* uvY  = b.CreateFSub(fcv(1.0f), b.CreateFDiv(b.CreateFMul(fcv(2.0f), pyf), ihV));
    auto* skyS = b.CreateFAdd(fcv(0.5f), b.CreateFMul(fcv(0.5f), uvY));
    auto skyC = [&](float hz, float tp){ return b.CreateFAdd(fcv(hz), b.CreateFMul(fcv(tp-hz), skyS)); };
    auto clamp01 = [&](llvm::Value* v){
        return frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum,
                 frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, v, fcv(0.0f)), fcv(1.0f)); };
    auto* colR = clamp01(b.CreateSelect(finalHit, lit, skyC(0.60f, 0.40f)));
    auto* colG = clamp01(b.CreateSelect(finalHit, lit, skyC(0.70f, 0.50f)));
    auto* colB = clamp01(b.CreateSelect(finalHit, lit, skyC(0.85f, 0.70f)));

    for (unsigned l = 0; l < W; ++l) {
        auto* idxRaw = b.CreateAdd(base, b.getInt32((int)l));
        auto* inR = b.CreateICmpSLT(idxRaw, total);
        auto* pix = b.CreateAdd(b.CreateMul(b.CreateExtractElement(PYv,l), IW),
                                b.CreateExtractElement(PXv,l));
        auto* off = b.CreateMul(pix, b.getInt32(4));
        auto st = [&](llvm::Value* vv, int c){
            auto* gep = b.CreateGEP(F32, OUT, b.CreateAdd(off, b.getInt32(c)));
            auto* cur = b.CreateLoad(F32, gep);
            b.CreateStore(b.CreateSelect(inR, b.CreateExtractElement(vv,l), cur), gep);
        };
        st(colR,0); st(colG,1); st(colB,2);
        auto* gep = b.CreateGEP(F32, OUT, b.CreateAdd(off, b.getInt32(3)));
        auto* cur = b.CreateLoad(F32, gep);
        b.CreateStore(b.CreateSelect(inR, llvm::ConstantFP::get(F32,1.0f), cur), gep);
    }
    auto* nbase = b.CreateAdd(base, b.getInt32((int)W));
    base->addIncoming(nbase, b.GetInsertBlock());
    b.CreateCondBr(b.CreateICmpSLT(nbase, total), ploop, pdone);

    b.SetInsertPoint(pdone);
    b.CreateRetVoid();
    verify_fn(fn);
    return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_gpu_kernel — per-pixel CUDA/NVPTX kernel
//
// Strategy: emit the normal per-tile renderer (the full raymarch), rename it
// to render_tile_impl, then emit a thin kernel named render_tile that reads
// the pixel from the NVPTX thread/block id and calls render_tile_impl for a
// 1×1 tile at that pixel. A W×H launch grid then renders the whole frame in
// parallel — each GPU thread does one pixel — reusing all the tracing logic
// with no duplication.
// ─────────────────────────────────────────────────────────────────────────────
llvm::Function* SceneCodegen::emit_gpu_kernel(const SceneGraph& scene,
                                              SceneSdfMode mode) {
    // 1. Emit the per-tile renderer and repurpose it as the device callee.
    llvm::Function* impl = emit_render_tile(scene, mode);
    impl->setName("render_tile_impl");
    impl->setLinkage(llvm::Function::InternalLinkage);
    // Make sure it is inlinable into the kernel (keeps the device code in one
    // function, which the PTX backend prefers).
    impl->addFnAttr(llvm::Attribute::AlwaysInline);

    auto& C = ctx_;
    llvm::IRBuilder<> b(C);

    // 2. The kernel has the SAME signature/ABI as render_tile (so CudaCtx
    //    launches it unchanged): the host passes the whole-frame tile, and
    //    each thread narrows it to its own pixel.
    auto* fty = impl->getFunctionType();
    auto* fn  = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "render_tile", mod_.get());
    fn->addFnAttr(llvm::Attribute::NoUnwind);

    // Name the args we need; forward the rest verbatim.
    auto args = fn->arg_begin();
    std::vector<llvm::Value*> a;
    for (auto& arg : fn->args()) a.push_back(&arg);
    // a[0]=out, a[1]=tx, a[2]=ty, a[3]=tw, a[4]=th, a[5]=iw, a[6]=ih, …

    auto* entry = llvm::BasicBlock::Create(C, "entry", fn);
    auto* body  = llvm::BasicBlock::Create(C, "body", fn);
    auto* done  = llvm::BasicBlock::Create(C, "done", fn);
    b.SetInsertPoint(entry);

    // 3. Global pixel index from NVPTX special registers:
    //    gx = ctaid.x * ntid.x + tid.x ; gy = ctaid.y * ntid.y + tid.y
    auto sreg = [&](const char* name) -> llvm::Value* {
        auto* ity = llvm::Type::getInt32Ty(C);
        auto* ft  = llvm::FunctionType::get(ity, false);
        auto callee = mod_->getOrInsertFunction(name, ft);
        return b.CreateCall(callee, {}, name + 5);  // drop "llvm." for a name
    };
    auto* tidx  = sreg("llvm.nvvm.read.ptx.sreg.tid.x");
    auto* ntidx = sreg("llvm.nvvm.read.ptx.sreg.ntid.x");
    auto* ctaix = sreg("llvm.nvvm.read.ptx.sreg.ctaid.x");
    auto* tidy  = sreg("llvm.nvvm.read.ptx.sreg.tid.y");
    auto* ntidy = sreg("llvm.nvvm.read.ptx.sreg.ntid.y");
    auto* ctaiy = sreg("llvm.nvvm.read.ptx.sreg.ctaid.y");
    auto* gx = b.CreateAdd(b.CreateMul(ctaix, ntidx), tidx, "gx");
    auto* gy = b.CreateAdd(b.CreateMul(ctaiy, ntidy), tidy, "gy");

    // Absolute pixel = tile origin + thread index.
    auto* px = b.CreateAdd(a[1], gx, "px");   // tx + gx
    auto* py = b.CreateAdd(a[2], gy, "py");   // ty + gy

    // Bounds: skip threads outside the tile (tx+tw, ty+th) and the image.
    auto* tx_end = b.CreateAdd(a[1], a[3], "tx_end");
    auto* ty_end = b.CreateAdd(a[2], a[4], "ty_end");
    llvm::Value* in = b.CreateICmpSLT(px, tx_end);
    in = b.CreateAnd(in, b.CreateICmpSLT(py, ty_end));
    in = b.CreateAnd(in, b.CreateICmpSLT(px, a[5]));   // px < iw
    in = b.CreateAnd(in, b.CreateICmpSLT(py, a[6]));   // py < ih
    in = b.CreateAnd(in, b.CreateICmpSGE(px, b.getInt32(0)));
    in = b.CreateAnd(in, b.CreateICmpSGE(py, b.getInt32(0)));
    b.CreateCondBr(in, body, done);

    // 4. Call render_tile_impl for the 1×1 tile at (px,py). Same args, but
    //    tx=px, ty=py, tw=1, th=1.
    b.SetInsertPoint(body);
    std::vector<llvm::Value*> call_args = a;  // copy
    call_args[1] = px;
    call_args[2] = py;
    call_args[3] = b.getInt32(1);   // tw
    call_args[4] = b.getInt32(1);   // th
    b.CreateCall(impl, call_args);
    b.CreateBr(done);

    b.SetInsertPoint(done);
    b.CreateRetVoid();

    verify_fn(fn);
    return fn;
}

} // namespace frep
