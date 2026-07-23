#pragma once
// core/frep/expr_ast.hpp
//
// Shared abstract syntax tree + parser for scalar math expressions used
// by CustomExprNode. The same AST is consumed by three back-ends:
//
//   - LLVM IR codegen       (for the CPU JIT pipeline)
//   - Direct interpretation (for FRepNode::eval — picker, marching cubes,
//                            BVH voxelization)
//   - GLSL source emission  (for the GPU compute path)
//
// Centralising parsing in one place means new functions or operators
// are added once and immediately available in every back-end. The
// previous design duplicated parsing in two places (the LLVM compiler
// in custom_expr.cpp and a private interpreter inside CustomExprNode);
// keeping them in sync was a maintenance hazard.
//
// Grammar (recursive descent, all left-associative):
//   expr     := term  (('+'|'-') term)*
//   term     := unary (('*'|'/') unary)*
//   unary    := '-' unary | '+' unary | primary
//   primary  := NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
//   args     := expr (',' expr)*
//
// Supported identifiers:
//   - Variables:  x, y, z
//   - Constants:  pi, e
//   - Unary functions:  sin, cos, tan, sqrt, abs, exp, log, floor, ceil
//   - Binary functions: pow, min, max

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace frep::expr {

// ── AST nodes ───────────────────────────────────────────────────────────────
//
// We keep the AST small — eight node types — and intentionally do not
// pool/intern them. Expressions are short (<100 chars) and parsed once
// per CustomExprNode, so allocation cost is irrelevant.

enum class Op { Add, Sub, Mul, Div };
enum class UnaryOp { Neg };

struct Node;
using NodePtr = std::shared_ptr<const Node>;

struct Node {
    enum class Kind {
        Number,     // a numeric literal — value in `num`
        Var,        // a variable reference — name in `ident` (x / y / z)
        Const,      // a named constant — name in `ident` (pi / e)
        BinOp,      // children[0..1], op stored in `bop`
        UnaryNeg,   // children[0]
        Call,       // function — `ident` is the name, children are args
    } kind;

    // Payload (union-ish — only the relevant field per kind is set).
    Op          bop  {};
    float       num  = 0.0f;
    std::string ident;

    std::vector<NodePtr> children;

    // Factories — keep ctors private-ish via these.
    static NodePtr number(float v) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Number; n->num = v;
        return n;
    }
    static NodePtr var(std::string name) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Var; n->ident = std::move(name);
        return n;
    }
    static NodePtr named_const(std::string name) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Const; n->ident = std::move(name);
        return n;
    }
    static NodePtr binop(Op op, NodePtr lhs, NodePtr rhs) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::BinOp; n->bop = op;
        n->children = {std::move(lhs), std::move(rhs)};
        return n;
    }
    static NodePtr neg(NodePtr child) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::UnaryNeg;
        n->children = {std::move(child)};
        return n;
    }
    static NodePtr call(std::string name, std::vector<NodePtr> args) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Call; n->ident = std::move(name);
        n->children = std::move(args);
        return n;
    }
};

// ── Parser ──────────────────────────────────────────────────────────────────
//
// Errors are reported by throwing ParseError; the caller (CustomExprNode
// or CustomExprCompiler) translates as appropriate. Throwing rather than
// returning a result keeps the recursive descent compact.

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;

    // Optional 0-based column position into the original source where
    // the error was detected. -1 means "no position available". The
    // GUI expression editor renders an arrow / underline at this
    // column; CLI tools can ignore it.
    int column = -1;

    ParseError(const std::string& msg, int col)
        : std::runtime_error(msg), column(col) {}
};

// Extra names in scope while parsing, beyond the built-in x,y,z / pi,e /
// builtins. `params` are additional identifiers that resolve to Var nodes
// (a template's scalar parameters); `templates` are callable names with their
// arity, so `name(args)` becomes a Call node even though it is not a builtin.
// Kept as plain data so this header has no dependency on the template registry.
struct ParseScope {
    std::vector<std::string>                  params;     // extra Var names
    std::vector<std::pair<std::string, int>>  templates;  // (name, arity)

    bool is_param(const std::string& n) const {
        for (const auto& p : params) if (p == n) return true;
        return false;
    }
    int template_arity(const std::string& n) const {
        for (const auto& t : templates) if (t.first == n) return t.second;
        return -1;
    }
};

// Parse a full expression string into an AST. The whole input must be
// consumed — trailing junk is an error. Returns a non-null NodePtr; on
// failure throws ParseError.
NodePtr parse(const std::string& src);

// Parse with extra parameters / callable templates in scope (see ParseScope).
NodePtr parse(const std::string& src, const ParseScope& scope);

// Constant folding pass — produces a semantically equivalent AST in
// which any subtree that has no variable references (pure constants
// + function calls) has been pre-evaluated to a single Number node.
//
// Example: fold(parse("(1 + 2) * x + sin(0)")) returns the same AST as
// parse("3 * x + 0") (which CSE might then simplify further, but the
// AST visitor leaves this to the backends).
//
// Always safe to apply — no side effects, no precision loss beyond
// what the back-ends would do at runtime. Returns a new tree; the
// original is unchanged (Nodes are immutable / shared_ptr).
NodePtr fold(const NodePtr& n);

// ── Function arity table ────────────────────────────────────────────────────
// Shared by all back-ends so arity-mismatch errors are caught once at
// parse time rather than per back-end.
//
// Returns -1 for unknown names; back-ends can either treat them as an
// error or extend the table. The parse() function itself only checks
// existence (so the AST can be walked by back-ends that intentionally
// support different sets); arity check happens in the back-ends.
inline int builtin_arity(const std::string& name) {
    if (name == "sin"   ) return 1;
    if (name == "cos"   ) return 1;
    if (name == "tan"   ) return 1;
    if (name == "sqrt"  ) return 1;
    if (name == "abs"   ) return 1;
    if (name == "exp"   ) return 1;
    if (name == "log"   ) return 1;
    if (name == "floor" ) return 1;
    if (name == "ceil"  ) return 1;
    if (name == "atan"  ) return 1;
    if (name == "asin"  ) return 1;
    if (name == "acos"  ) return 1;
    if (name == "pow"   ) return 2;
    // nth_root(a, b) == a^(1/b), matching libfive's array evaluator. Note b is
    // a plain float, not an integer: converted scenes carry b = 0.5 (i.e. a^2).
    if (name == "nth_root") return 2;
    if (name == "atan2" ) return 2;
    if (name == "mod"   ) return 2;
    if (name == "min"   ) return 2;
    if (name == "max"   ) return 2;
    return -1;
}

} // namespace frep::expr
