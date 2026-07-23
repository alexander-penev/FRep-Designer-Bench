#pragma once
// core/frep/custom_expr.hpp
//
// CustomExprNode — F-Rep node whose SDF is described by a text expression.
//
// The expression is parsed once (per node instance, on demand) into a
// shared AST (see expr_ast.hpp), and that AST is then consumed by
// three different back-ends:
//
//   1. LLVM IR codegen      — for the CPU JIT pipeline
//                              (`CustomExprNode::codegen`)
//   2. Direct interpretation — for `FRepNode::eval()` calls coming from
//                              the picker, marching cubes, and BVH
//   3. GLSL emission         — for the GPU compute path
//                              (`CustomExprNode::emit_glsl`)
//
// All three back-ends walk the same AST and so are guaranteed to agree
// on the syntax accepted, the arity of functions, and the meaning of
// each operator.

#include "core/frep/expr_ast.hpp"
#include "core/frep/node.hpp"
#include "core/frep/template_fn.hpp"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <unordered_map>
#include <ostream>
#include <string>

namespace frep {

struct TemplateFn;
class  TemplateRegistry;

// CustomExprCompiler — emits an LLVM function from a CustomExpr-AST.
// This is the LLVM back-end; the parse is done once by
// frep::expr::parse() (called inside CustomExprNode::codegen) and the
// resulting AST is passed in.
class CustomExprCompiler {
public:
    // Build an LLVM function `<fn_name>(float, float, float) -> float`
    // implementing the AST. Returns nullptr on failure (see last_error).
    llvm::Function* compile(llvm::Module&         mod,
                            llvm::LLVMContext&    ctx,
                            const std::string&    fn_name,
                            const expr::NodePtr&  ast);

    // Convenience overload that parses the source string first.
    llvm::Function* compile(llvm::Module&         mod,
                            llvm::LLVMContext&    ctx,
                            const std::string&    fn_name,
                            const std::string&    expr_src);

    // Vector variant: builds `void <fn>(const float* X,const float* Y,
    // const float* Z, float* O)` evaluating W lanes per call (W = 8 → AVX2).
    // Same AST, DAG-shared, transcendentals without a vector intrinsic fall
    // back to per-lane scalar libm.
    llvm::Function* compile_vec(llvm::Module&        mod,
                                llvm::LLVMContext&   ctx,
                                const std::string&   fn_name,
                                const expr::NodePtr& ast,
                                unsigned             width = 8);

    // Interval variant: `void <fn>(const float* B, float* O)` where
    // B=[xlo,xhi,ylo,yhi,zlo,zhi], O=[flo,fhi]. Used for octree pruning.
    // Arithmetic + abs/sqrt/min/max only (no trig yet -> returns error).
    llvm::Function* compile_interval(llvm::Module&        mod,
                                     llvm::LLVMContext&   ctx,
                                     const std::string&   fn_name,
                                     const expr::NodePtr& ast);

    // Inline the vector AST into an existing builder (SIMD broadcast codegen).
    llvm::Value* gen_vec_inline(llvm::Module& mod, llvm::LLVMContext& ctx,
                                llvm::IRBuilder<>& b, const expr::NodePtr& ast,
                                llvm::Value* x, llvm::Value* y, llvm::Value* z,
                                unsigned width);

    // ── User-defined template functions ──────────────────────────────────────
    // Provide the registry so a Call to a user template resolves. Its functions
    // must be emitted into the module first (emit_templates), then a call site
    // references them by name.
    void set_templates(const TemplateRegistry* r) { reg_ = r; }

    // Emit every template in `reg_` into `mod` as a callable
    //   float frep_tmpl_<name>(float p0, ..., float x, float y, float z)
    // in definition order (a template may call ones defined before it).
    // Idempotent per module. Returns false on error (see last_error()).
    bool emit_templates(llvm::Module& mod, llvm::LLVMContext& ctx);

    const std::string& last_error() const { return error_; }

private:
    std::string        error_;
    llvm::LLVMContext* ctx_ = nullptr;
    llvm::Module*      mod_ = nullptr;
    llvm::IRBuilder<>* b_   = nullptr;
    llvm::Value*       vx_  = nullptr;
    llvm::Value*       vy_  = nullptr;
    llvm::Value*       vz_  = nullptr;
    std::unordered_map<const expr::Node*, llvm::Value*> memo_;  // DAG: emit shared subtrees once

    // User templates: registry + scalar parameter bindings for the function
    // currently being emitted (param name -> its llvm argument).
    const TemplateRegistry* reg_ = nullptr;
    std::unordered_map<std::string, llvm::Value*> params_;
    llvm::Function* emit_template_fn(const TemplateFn& t);

    llvm::Value* gen(const expr::Node& n);
    llvm::Value* gen_call(const expr::Node& n);

    // Interval twin: same AST -> {lo,hi} arithmetic. lo_/hi_ hold the x/y/z
    // interval endpoints while an interval function is built.
    llvm::Value *xlo_=nullptr,*xhi_=nullptr,*ylo_=nullptr,*yhi_=nullptr,*zlo_=nullptr,*zhi_=nullptr;
    std::unordered_map<const expr::Node*, std::pair<llvm::Value*,llvm::Value*>> imemo_;
    std::unordered_map<std::string, std::pair<llvm::Value*,llvm::Value*>> iparams_;  // template params (interval)
    std::pair<llvm::Value*,llvm::Value*> gen_ival(const expr::Node& n);
    std::pair<llvm::Value*,llvm::Value*> gen_call_ival(const expr::Node& n);

    // SIMD twin: same AST, <W x float> lanes. vx_/vy_/vz_ hold the vector args
    // while a vector function is built.
    unsigned     vw_ = 0;   // active vector width (0 = scalar)
    std::unordered_map<const expr::Node*, llvm::Value*> vmemo_;
    std::unordered_map<std::string, llvm::Value*> vparams_;  // template params (vector)
    llvm::Value* gen_vec(const expr::Node& n);
    llvm::Value* gen_call_vec(const expr::Node& n);

    llvm::Type*  f32() { return llvm::Type::getFloatTy(*ctx_); }
    llvm::Value* fc(float v);

    void fail(const std::string& msg) {
        if (error_.empty()) error_ = msg;
    }
};


// CustomExprNode — FRepNode wrapping a user-supplied analytic expression.
class CustomExprNode final : public FRepNode {
public:
    CustomExprNode(std::string expr_text, std::string nid = "expr")
        : expr_(std::move(expr_text))
    {
        kind = NodeKind::Plugin;
        id   = std::move(nid);
    }

    float eval(float x, float y, float z) const override {
        ensure_parsed();
        return eval_ast(*ast_, x, y, z, nullptr, reg_);
    }

    // Make user-defined template functions visible to this node's expression
    // (so its body may call them and reference their parameters). Non-owning:
    // the registry is owned by the scene and must outlive the node. Resets the
    // cached parse so the body re-parses with the templates in scope.
    void set_templates(const TemplateRegistry* r) { reg_ = r; ast_ = nullptr; }
    const TemplateRegistry* templates() const { return reg_; }

    llvm::Value* codegen(CgCtx& c, llvm::Value* x,
                         llvm::Value* y, llvm::Value* z) const override;

    std::size_t structural_hash() const noexcept override {
        return std::hash<std::string>{}(expr_) ^ 0xCAFE'BABEull;
    }
    std::size_t structure_hash() const noexcept override {
        return std::hash<std::string>{}(expr_) ^ 0xCAFE'BABEull;
    }

    const char* type_name() const noexcept override { return "CustomExpr"; }

    // A CustomExpr has no analyzable bounds, so it defaults to an infinite AABB.
    // That defeats any spatial acceleration (the RTX multi-BLAS TLAS builds one
    // box per group, and an infinite box makes the broad phase cull nothing).
    // When the author knows the geometry's extent — a bounded prototype used as
    // an instance — set it here so the BVH can prune. Purely an acceleration
    // hint; it must conservatively contain the surface or the render clips.
    void set_bounds(const AABB& b) { bounds_ = b; }
    AABB aabb() const override { return bounds_; }

    bool emit_glsl(std::ostream& out,
                   const std::vector<std::string>& /*child_exprs*/,
                   const std::string& /*var_prefix*/) const override {
        ensure_parsed();
        out << "(";
        emit_glsl_ast(out, *ast_, reg_);
        out << ")";
        return true;
    }

    const std::string& expression() const { return expr_; }

private:
    std::string expr_;
    const TemplateRegistry* reg_ = nullptr;   // user templates in scope (non-owning)
    AABB bounds_ = AABB::infinite();          // acceleration hint (see set_bounds)

    // Cached AST — parsed lazily on first back-end call.
public:
    // Parsed+folded AST (parses on first use). Used by the SIMD compile path.
    const expr::NodePtr& ast() const { ensure_parsed(); return ast_; }
    const void* custom_expr_ast() const override { return &ast(); }
private:
    mutable expr::NodePtr ast_;

    // Cached LLVM function name — deduplicates IR when scene_normal
    // calls the SDF 6 times in a row.
    mutable std::string cached_fn_name_;

    void ensure_parsed() const {
        if (ast_) return;
        if (reg_ && !reg_->empty()) {
            expr::ParseScope scope;
            for (const auto& t : reg_->all())
                scope.templates.push_back({t.name,
                                           static_cast<int>(t.params.size())});
            ast_ = expr::fold(expr::parse(expr_, scope));
        } else {
            ast_ = expr::fold(expr::parse(expr_));
        }
    }

public:
    static float eval_ast(const expr::Node& n, float x, float y, float z);
    // Extended interpreter: `params` binds template scalar parameters by name,
    // `reg` resolves calls to user templates (recursively). Both may be null,
    // in which case only x,y,z + builtins are accepted (the base behaviour).
    static float eval_ast(const expr::Node& n, float x, float y, float z,
                          const std::unordered_map<std::string, float>* params,
                          const TemplateRegistry* reg);
    // Registry-aware GLSL: a Call to a user template emits
    //   frep_tmpl_<name>(args..., x, y, z)
    // and a Var that is a template parameter emits its name (an argument of the
    // enclosing template function). `reg` may be null (base behaviour).
    static void emit_glsl_ast(std::ostream& out, const expr::Node& n,
                              const TemplateRegistry* reg);
    // Emit every template as a GLSL function
    //   float frep_tmpl_<name>(float p0, ..., float x, float y, float z) {...}
    // in definition order, ready to be prepended before scene_sdf.
    static void emit_templates_glsl(std::ostream& out, const TemplateRegistry& reg);

private:
    static void  emit_glsl_ast(std::ostream& out, const expr::Node& n);
};

} // namespace frep
