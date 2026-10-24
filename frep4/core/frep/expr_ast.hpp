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
#include <optional>
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

// ── Structural equality ─────────────────────────────────────────────────────
// The parser does not intern subtrees: "1.2*z" written twice is two nodes, and
// fold() rebuilds what it touches. A rule that needs "the same argument" in two
// places therefore compares structure, never pointers.
inline bool same_structure(const Node& a, const Node& b) {
    if (&a == &b) return true;
    if (a.kind != b.kind || a.children.size() != b.children.size()) return false;
    switch (a.kind) {
        case Node::Kind::Number:   if (a.num != b.num)   return false; break;
        case Node::Kind::Var:
        case Node::Kind::Const:
        case Node::Kind::Call:     if (a.ident != b.ident) return false; break;
        case Node::Kind::BinOp:    if (a.bop != b.bop)   return false; break;
        case Node::Kind::UnaryNeg: break;
    }
    for (std::size_t i = 0; i < a.children.size(); ++i)
        if (!same_structure(*a.children[i], *b.children[i])) return false;
    return true;
}

// ── Rotation patterns ───────────────────────────────────────────────────────
// Shared by the interval evaluators (gen_ival, expr_interval, the GLSL interval
// emitter and the bounds search's ival), so that all of them see the same trees.
//
// A rotation pair is ±a·cos θ ± b·sin θ: two products, one carrying cos and the
// other sin of structurally the same argument, in either factor order and with
// either sign on either product. A unary minus or the "0.0 - p" the scene files
// use for a leading minus both count as a sign. Term by term, interval
// arithmetic bounds the sum as if a·cos θ and b·sin θ could both be extreme at
// once, |a| + |b| in the worst case; the exact fact is |a·cos θ + b·sin θ| ≤
// sqrt(a² + b²) for every θ, up to √2 tighter.
struct RotationPair {
    const Node* a;        // the factor of cos θ
    const Node* b;        // the factor of sin θ
    const Node* theta;    // the shared argument
    int         sign;     // +1 when the two products carry the same sign, -1 otherwise
};

namespace pattern_detail {
// Peels a leading minus (unary, or "0.0 - p") off p, flipping `sign` each time.
inline const Node* strip_sign(const Node* p, int& sign) {
    for (;;) {
        if (p->kind == Node::Kind::UnaryNeg) { sign = -sign; p = p->children[0].get(); continue; }
        if (p->kind == Node::Kind::BinOp && p->bop == Op::Sub &&
            p->children[0]->kind == Node::Kind::Number && p->children[0]->num == 0.0f) {
            sign = -sign; p = p->children[1].get(); continue;
        }
        return p;
    }
}
struct TrigProduct { const Node* factor; const Node* theta; bool is_sin; };
// f·sin(θ) or f·cos(θ) in either order.
inline std::optional<TrigProduct> match_trig_product(const Node& n) {
    if (n.kind != Node::Kind::BinOp || n.bop != Op::Mul) return std::nullopt;
    for (int i = 0; i < 2; ++i) {
        const Node& t = *n.children[i];
        if (t.kind == Node::Kind::Call && t.children.size() == 1 &&
            (t.ident == "sin" || t.ident == "cos"))
            return TrigProduct{n.children[1 - i].get(), t.children[0].get(), t.ident == "sin"};
    }
    return std::nullopt;
}
// abs(p) with an optional constant offset: abs(p) + k, k + abs(p), abs(p) - k.
inline const Node* match_abs_offset(const Node& n, float& k) {
    const Node* p = &n; k = 0.0f;
    if (n.kind == Node::Kind::BinOp && (n.bop == Op::Add || n.bop == Op::Sub)) {
        const Node& l = *n.children[0]; const Node& r = *n.children[1];
        if (r.kind == Node::Kind::Number)                      { k = n.bop == Op::Sub ? -r.num : r.num; p = &l; }
        else if (l.kind == Node::Kind::Number && n.bop == Op::Add) { k = l.num; p = &r; }
        else return nullptr;
    }
    if (p->kind != Node::Kind::Call || p->ident != "abs" || p->children.size() != 1) return nullptr;
    return p->children[0].get();
}
} // namespace pattern_detail

inline std::optional<RotationPair> match_rotation_pair(const Node& n) {
    using namespace pattern_detail;
    if (n.kind != Node::Kind::BinOp || (n.bop != Op::Add && n.bop != Op::Sub)) return std::nullopt;
    int sl = 1, sr = n.bop == Op::Sub ? -1 : 1;
    const Node* l = strip_sign(n.children[0].get(), sl);
    const Node* r = strip_sign(n.children[1].get(), sr);
    auto pl = match_trig_product(*l), pr = match_trig_product(*r);
    if (!pl || !pr || pl->is_sin == pr->is_sin) return std::nullopt;
    if (!same_structure(*pl->theta, *pr->theta)) return std::nullopt;
    const TrigProduct& c = pl->is_sin ? *pr : *pl;
    const TrigProduct& s = pl->is_sin ? *pl : *pr;
    return RotationPair{c.factor, s.factor, c.theta, sl * sr};
}

// A rotated box is max(|P| + c1, |Q| + c2) with P and Q rotation pairs of the
// same θ in which a and b exchange roles and the sign products are opposite:
// P = ±a·cos θ ± b·sin θ, Q = ∓b·cos θ ± a·sin θ, so that P² + Q² = a² + b² —
// (P, Q) is (a, b) turned by θ, the cross-section of a bar twisted along θ. Term
// by term, max(|P| − c, |Q| − c) is bounded below by 0 − c as soon as P and Q
// both straddle 0, which far from the axis they do whenever θ spans more than a
// quarter turn over the box; the bounds search for the twisted bar refines
// every such cell to the finest level for nothing. The exact fact is
// max(|P|, |Q|) ≥ sqrt((P² + Q²)/2) = sqrt((a² + b²)/2), which is positive on
// any box that keeps away from the axis.
struct RotatedBox {
    const Node* a;
    const Node* b;
    float       c1, c2;   // the constant offsets (0 when absent)
};

inline std::optional<RotatedBox> match_rotated_box(const Node& n) {
    using namespace pattern_detail;
    if (n.kind != Node::Kind::Call || n.ident != "max" || n.children.size() != 2) return std::nullopt;
    float c1, c2;
    const Node* p = match_abs_offset(*n.children[0], c1);
    const Node* q = match_abs_offset(*n.children[1], c2);
    if (!p || !q) return std::nullopt;
    auto P = match_rotation_pair(*p), Q = match_rotation_pair(*q);
    if (!P || !Q || P->sign == Q->sign) return std::nullopt;
    if (!same_structure(*P->theta, *Q->theta)) return std::nullopt;
    if (!same_structure(*P->a, *Q->b) || !same_structure(*P->b, *Q->a)) return std::nullopt;
    return RotatedBox{P->a, P->b, c1, c2};
}

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
