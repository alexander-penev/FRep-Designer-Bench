// tests/test_expr_backends.cpp
//
// Integration test for the shared-AST refactor of CustomExprNode. The
// refactor introduced one parser feeding three back-ends (LLVM IR codegen,
// CPU eval, GLSL emission). This file verifies that all three back-ends
// produce numerically equivalent results for the same expression — the
// whole point of the refactor.
//
// We can't run GLSL in a unit test trivially, but we CAN verify that:
//   1. AST parsing produces the same tree for the same input
//   2. CPU eval and LLVM JIT compute the same float value at the same
//      coordinates
//   3. Bad expressions throw consistent ParseError
//   4. Function arity errors are caught at parse time, before any
//      back-end is invoked

#include "core/frep/expr_ast.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>

#include <cmath>
#include <memory>

using namespace frep;

namespace {

class ExprBackendsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};

using SdfFn = float(*)(float, float, float);

// JIT-compile a single CustomExpr scene and return a callable.
// Each call leaks a JitEngine (held by ownership in the unique_ptr
// returned to the test), but that's fine: the test process terminates
// immediately after, and a `JitEngine` holds Onto its compiled module
// so the function pointer stays alive while we use it.
struct JittedSdf {
    std::unique_ptr<JitEngine> jit;
    SdfFn                       fn = nullptr;
};

JittedSdf jit_expr(const std::string& expr_text) {
    JittedSdf out;
    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx);
    auto node = std::make_shared<CustomExprNode>(expr_text, "e");
    cg.emit_scene_sdf(*node);

    auto& mod = *cg.module();
    auto* sdf_fn = mod.getFunction("scene_sdf");
    if (!sdf_fn) return out;

    // Need a public wrapper for JitEngine::load (looks up "render_tile").
    auto* fty = llvm::FunctionType::get(
        llvm::Type::getFloatTy(*ctx),
        {llvm::Type::getFloatTy(*ctx),
         llvm::Type::getFloatTy(*ctx),
         llvm::Type::getFloatTy(*ctx)}, false);
    auto* wrapper = llvm::Function::Create(
        fty, llvm::Function::ExternalLinkage, "render_tile", &mod);
    auto* bb = llvm::BasicBlock::Create(*ctx, "entry", wrapper);
    llvm::IRBuilder<> b(bb);
    auto it = wrapper->arg_begin();
    auto* x = &*it++; auto* y = &*it++; auto* z = &*it++;
    b.CreateRet(b.CreateCall(sdf_fn, {x, y, z}));

    auto mod_ptr = cg.take_module();
    out.jit = std::make_unique<JitEngine>();
    auto fn_or = out.jit->load(std::move(mod_ptr), std::move(ctx));
    if (!fn_or) return out;
    out.fn = reinterpret_cast<SdfFn>(*fn_or);
    return out;
}

float cpu_eval(const std::string& expr, float x, float y, float z) {
    CustomExprNode n(expr);
    return n.eval(x, y, z);
}

} // anon

TEST_F(ExprBackendsTest, ParserProducesAst) {
    auto ast = expr::parse("x*x + y - 1.0");
    ASSERT_TRUE(ast);
    // Top level is `(x*x + y) - 1.0`, a binop Sub. Left is a binop Add.
    EXPECT_EQ(ast->kind, expr::Node::Kind::BinOp);
    EXPECT_EQ(ast->bop,  expr::Op::Sub);
    EXPECT_EQ(ast->children[0]->bop, expr::Op::Add);
    EXPECT_EQ(ast->children[1]->kind, expr::Node::Kind::Number);
    EXPECT_FLOAT_EQ(ast->children[1]->num, 1.0f);
}

TEST_F(ExprBackendsTest, CpuAndJitAgreeOnSphere) {
    const char* expr = "sqrt(x*x + y*y + z*z) - 1.0";
    auto j = jit_expr(expr);
    ASSERT_TRUE(j.fn);
    struct P { float x, y, z; };
    const std::array<P, 4> pts = {
        P{0, 0, 0}, P{1, 0, 0}, P{0.5f, 0.5f, 0.5f}, P{-0.7f, 0.3f, -0.2f},
    };
    for (auto& p : pts) {
        float a = cpu_eval(expr, p.x, p.y, p.z);
        float b = j.fn(p.x, p.y, p.z);
        EXPECT_NEAR(a, b, 1e-5)
            << "Mismatch at (" << p.x << "," << p.y << "," << p.z << ")";
    }
}

TEST_F(ExprBackendsTest, CpuAndJitAgreeOnGyroid) {
    const char* expr = "sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)";
    auto j = jit_expr(expr);
    ASSERT_TRUE(j.fn);
    for (float u = -1.5f; u < 1.5f; u += 0.4f) {
        float a = cpu_eval(expr, u, u + 0.1f, u - 0.2f);
        float b = j.fn(u, u + 0.1f, u - 0.2f);
        EXPECT_NEAR(a, b, 1e-4) << "Mismatch at u=" << u;
    }
}

TEST_F(ExprBackendsTest, CpuAndJitAgreeOnTranscendentals) {
    struct Case { const char* expr; float x, y, z; };
    const std::array<Case, 8> cases = {
        Case{"sin(x)",   1.2f, 0, 0},
        Case{"cos(x)",   2.7f, 0, 0},
        Case{"tan(x)",   0.3f, 0, 0},
        Case{"sqrt(x*x + 1)", 0.5f, 0, 0},
        Case{"abs(-x)",  3.0f, 0, 0},
        Case{"exp(x/2)", 0.4f, 0, 0},
        Case{"log(x+2)", 0.3f, 0, 0},
        Case{"pow(x, 3) + min(y, z)", 1.5f, 2.0f, 1.0f},
    };
    for (auto& c : cases) {
        auto j = jit_expr(c.expr);
        ASSERT_TRUE(j.fn) << "JIT failed on '" << c.expr << "'";
        float a = cpu_eval(c.expr, c.x, c.y, c.z);
        float b = j.fn(c.x, c.y, c.z);
        EXPECT_NEAR(a, b, 1e-3) << "Mismatch on '" << c.expr << "'";
    }
}

TEST_F(ExprBackendsTest, ConstantsResolveSame) {
    auto j_pi = jit_expr("pi");
    auto j_e  = jit_expr("e");
    ASSERT_TRUE(j_pi.fn);
    ASSERT_TRUE(j_e.fn);
    EXPECT_NEAR(cpu_eval("pi", 0, 0, 0), j_pi.fn(0, 0, 0), 1e-6);
    EXPECT_NEAR(cpu_eval("e",  0, 0, 0), j_e.fn(0, 0, 0),  1e-6);
}

TEST_F(ExprBackendsTest, ParseErrorOnUnknownIdent) {
    EXPECT_THROW(expr::parse("foo + 1"), expr::ParseError);
}

TEST_F(ExprBackendsTest, ParseErrorOnUnknownFunction) {
    EXPECT_THROW(expr::parse("frobnicate(x)"), expr::ParseError);
}

TEST_F(ExprBackendsTest, ParseErrorOnArityMismatch) {
    EXPECT_THROW(expr::parse("pow(x)"), expr::ParseError);
    EXPECT_THROW(expr::parse("pow(x, y, z)"), expr::ParseError);
    EXPECT_THROW(expr::parse("sin(x, y)"), expr::ParseError);
}

TEST_F(ExprBackendsTest, ParseErrorOnTrailingJunk) {
    EXPECT_THROW(expr::parse("1 + 2 garbage"), expr::ParseError);
}

TEST_F(ExprBackendsTest, GlslEmissionContainsRightFunctions) {
    CustomExprNode n("sin(x) * cos(y) + sqrt(z)");
    std::ostringstream os;
    bool ok = n.emit_glsl(os, {}, "v_");
    ASSERT_TRUE(ok);
    auto s = os.str();
    EXPECT_NE(s.find("sin"),  std::string::npos);
    EXPECT_NE(s.find("cos"),  std::string::npos);
    EXPECT_NE(s.find("sqrt"), std::string::npos);
    EXPECT_NE(s.find("+"),    std::string::npos);
}

TEST_F(ExprBackendsTest, AstSharedAcrossBackendCalls) {
    // Sequential calls on the same node use the same cached AST.
    CustomExprNode n("x*x - 0.5");
    EXPECT_FLOAT_EQ(n.eval(1.0f, 0, 0),  0.5f);
    std::ostringstream os;
    n.emit_glsl(os, {}, "v_");
    EXPECT_FLOAT_EQ(n.eval(0.0f, 0, 0), -0.5f);
}
