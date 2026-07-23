// core/frep/expr_ast.cpp
//
// Implementation of frep::expr::parse — recursive descent expression
// parser that produces the shared AST defined in expr_ast.hpp.
//
// The lexer is a single forward sweep over the input; the parser is
// pure recursive descent matching the grammar in the header. Both phases
// throw ParseError on any malformed input — there are no error-recovery
// or "best effort" semantics, because the caller (CustomExprNode) wants
// to know precisely when an expression won't be evaluable on any
// back-end.

#include "expr_ast.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

#include <map>
namespace frep::expr {

namespace {

// ── Tokens ──────────────────────────────────────────────────────────────────
enum class Tok {
    Num, Ident,
    Plus, Minus, Star, Slash,
    LParen, RParen, Comma,
    Semi, Assign,
    End,
};
struct Token {
    Tok          k;
    std::string  s;   // for Ident
    float        v;   // for Num
    int          col = 0;  // 0-based column in source
};

// ── Lexer ───────────────────────────────────────────────────────────────────
std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> out;
    std::size_t i = 0;
    while (i < src.size()) {
        char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }

        // Number: matches optional digits + optional '.' + digits + optional
        // exponent. We don't bother with strict validation here — strtof
        // will accept the longest valid prefix and we trust it.
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            int col = static_cast<int>(i);
            std::size_t j = i;
            while (j < src.size() &&
                   (std::isdigit(static_cast<unsigned char>(src[j])) ||
                    src[j] == '.' || src[j] == 'e' || src[j] == 'E' ||
                    ((src[j] == '+' || src[j] == '-') && j > i &&
                     (src[j-1] == 'e' || src[j-1] == 'E'))))
                ++j;
            out.push_back({Tok::Num, {}, std::strtof(src.c_str() + i, nullptr), col});
            i = j;
            continue;
        }

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            int col = static_cast<int>(i);
            std::size_t j = i;
            while (j < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_'))
                ++j;
            out.push_back({Tok::Ident, src.substr(i, j - i), 0.0f, col});
            i = j;
            continue;
        }

        int col = static_cast<int>(i);
        switch (c) {
            case '+': out.push_back({Tok::Plus,   {}, 0, col}); ++i; break;
            case '-': out.push_back({Tok::Minus,  {}, 0, col}); ++i; break;
            case '*': out.push_back({Tok::Star,   {}, 0, col}); ++i; break;
            case '/': out.push_back({Tok::Slash,  {}, 0, col}); ++i; break;
            case '(': out.push_back({Tok::LParen, {}, 0, col}); ++i; break;
            case ')': out.push_back({Tok::RParen, {}, 0, col}); ++i; break;
            case ',': out.push_back({Tok::Comma,  {}, 0, col}); ++i; break;
            case ';': out.push_back({Tok::Semi,   {}, 0, col}); ++i; break;
            case '=': out.push_back({Tok::Assign, {}, 0, col}); ++i; break;
            default:
                throw ParseError(
                    std::string("unexpected character '") + c +
                    "' in expression", col);
        }
    }
    out.push_back({Tok::End, {}, 0, static_cast<int>(src.size())});
    return out;
}

// ── Parser (mutable cursor) ─────────────────────────────────────────────────
struct Parser {
    std::vector<Token> toks;
    std::map<std::string, NodePtr> env;   // let-bound names -> shared AST
    const ParseScope*  scope = nullptr;   // template params / callable templates
    std::size_t        pos = 0;

    Token& peek() { return toks[pos]; }
    Token  consume() { return toks[pos++]; }
    bool   match(Tok k) {
        if (peek().k == k) { ++pos; return true; }
        return false;
    }

    NodePtr parse_expr() {
        auto lhs = parse_term();
        while (peek().k == Tok::Plus || peek().k == Tok::Minus) {
            Op op = (consume().k == Tok::Plus) ? Op::Add : Op::Sub;
            auto rhs = parse_term();
            lhs = Node::binop(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    NodePtr parse_term() {
        auto lhs = parse_unary();
        while (peek().k == Tok::Star || peek().k == Tok::Slash) {
            Op op = (consume().k == Tok::Star) ? Op::Mul : Op::Div;
            auto rhs = parse_unary();
            lhs = Node::binop(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    NodePtr parse_unary() {
        if (match(Tok::Minus)) return Node::neg(parse_unary());
        if (match(Tok::Plus))  return parse_unary();  // unary + is a no-op
        return parse_primary();
    }

    NodePtr parse_primary() {
        auto& t = peek();
        if (t.k == Tok::Num) {
            float v = consume().v;
            return Node::number(v);
        }
        if (t.k == Tok::LParen) {
            int lp_col = consume().col;
            auto e = parse_expr();
            if (!match(Tok::RParen))
                throw ParseError("expected ')'", lp_col);
            return e;
        }
        if (t.k == Tok::Ident) {
            int ident_col = peek().col;
            std::string name = consume().s;
            // Function call?
            if (match(Tok::LParen)) {
                std::vector<NodePtr> args;
                if (peek().k != Tok::RParen) {
                    while (true) {
                        args.push_back(parse_expr());
                        if (match(Tok::Comma)) continue;
                        break;
                    }
                }
                if (!match(Tok::RParen))
                    throw ParseError("expected ')' in call to '" + name + "'",
                                     ident_col);
                int want = builtin_arity(name);
                if (want < 0 && scope)          // maybe a user template
                    want = scope->template_arity(name);
                if (want < 0)
                    throw ParseError("unknown function '" + name + "'", ident_col);
                if (static_cast<int>(args.size()) != want)
                    throw ParseError("function '" + name + "' expects " +
                        std::to_string(want) + " arg(s), got " +
                        std::to_string(args.size()), ident_col);
                return Node::call(std::move(name), std::move(args));
            }
            // Let-bound name -> shared subtree (preserves DAG).
            if (auto it = env.find(name); it != env.end())
                return it->second;
            // Variable.
            if (name == "x" || name == "y" || name == "z")
                return Node::var(name);
            if (name == "pi" || name == "e")
                return Node::named_const(name);
            // A declared template parameter also resolves to a Var; the
            // back-ends bind it to the function argument of that name.
            if (scope && scope->is_param(name))
                return Node::var(name);
            throw ParseError("unknown identifier '" + name + "'", ident_col);
        }
        throw ParseError("unexpected token in expression", t.col);
    }
};

} // anon

// Let-prelude: zero or more "IDENT = expr ;" bindings, then the result expr.
// Each binding shares its AST via env, so repeated names reuse one subtree.
static NodePtr parse_with_lets(Parser& p) {
    while (p.peek().k == Tok::Ident && p.toks[p.pos + 1].k == Tok::Assign) {
        int col = p.peek().col;
        std::string name = p.consume().s;
        if (name == "x" || name == "y" || name == "z" || name == "pi" || name == "e")
            throw ParseError("cannot bind reserved name '" + name + "'", col);
        p.consume(); // '='
        auto val = p.parse_expr();
        if (!p.match(Tok::Semi))
            throw ParseError("expected ';' after binding '" + name + "'", col);
        p.env[name] = val;
    }
    return p.parse_expr();
}
NodePtr parse(const std::string& src) {
    Parser p;
    p.toks = tokenize(src);
    auto root = parse_with_lets(p);
    if (p.peek().k != Tok::End)
        throw ParseError("trailing tokens at end of expression", p.peek().col);
    return root;
}

NodePtr parse(const std::string& src, const ParseScope& scope) {
    Parser p;
    p.toks  = tokenize(src);
    p.scope = &scope;
    auto root = parse_with_lets(p);
    if (p.peek().k != Tok::End)
        throw ParseError("trailing tokens at end of expression", p.peek().col);
    return root;
}

// ── Constant folding ───────────────────────────────────────────────────────
namespace {

// Numeric value of a named constant, used by the folder.
float const_value(const std::string& name) {
    if (name == "pi") return 3.14159265358979323846f;
    if (name == "e")  return 2.71828182845904523536f;
    return 0.0f;
}

// Apply a builtin function. Caller guarantees the right arity (parser
// checked at parse time). Returns NaN on unknown.
float apply_func(const std::string& name, const std::vector<float>& args) {
    if (name == "sin")   return std::sin(args[0]);
    if (name == "cos")   return std::cos(args[0]);
    if (name == "tan")   return std::tan(args[0]);
    if (name == "sqrt")  return std::sqrt(args[0]);
    if (name == "abs")   return std::abs(args[0]);
    if (name == "exp")   return std::exp(args[0]);
    if (name == "log")   return std::log(args[0]);
    if (name == "floor") return std::floor(args[0]);
    if (name == "ceil")  return std::ceil(args[0]);
    if (name == "pow")   return std::pow(args[0], args[1]);
    if (name == "nth_root") return std::pow(args[0], 1.0f / args[1]);
    if (name == "min")   return std::fmin(args[0], args[1]);
    if (name == "max")   return std::fmax(args[0], args[1]);
    return std::nan("");
}

bool is_number(const NodePtr& n) {
    return n && n->kind == Node::Kind::Number;
}

} // anon

static NodePtr fold_impl(const NodePtr& n,
                        std::map<const Node*, NodePtr>& memo);
NodePtr fold(const NodePtr& n) {
    std::map<const Node*, NodePtr> memo;
    return fold_impl(n, memo);
}
static NodePtr fold_impl(const NodePtr& n,
                         std::map<const Node*, NodePtr>& memo) {
    if (!n) return n;
    if (auto it = memo.find(n.get()); it != memo.end()) return it->second;
    using Kind = Node::Kind;
    switch (n->kind) {
        case Kind::Number:
            return n;  // already a constant — share
        case Kind::Var:
            return n;  // depends on x/y/z — not foldable
        case Kind::Const:
            // A named constant has a known numeric value at compile time.
            // Fold into a Number so downstream emit/eval doesn't have to
            // know about it.
            return Node::number(const_value(n->ident));
        case Kind::UnaryNeg: {
            auto c = fold_impl(n->children[0], memo);
            if (is_number(c)) return Node::number(-c->num);
            return memo[n.get()] = Node::neg(c);
        }
        case Kind::BinOp: {
            auto l = fold_impl(n->children[0], memo);
            auto r = fold_impl(n->children[1], memo);
            if (is_number(l) && is_number(r)) {
                float v = 0;
                switch (n->bop) {
                    case Op::Add: v = l->num + r->num; break;
                    case Op::Sub: v = l->num - r->num; break;
                    case Op::Mul: v = l->num * r->num; break;
                    case Op::Div: v = l->num / r->num; break;
                }
                return Node::number(v);
            }
            // Algebraic simplifications — small set, low risk of changing
            // numerical behaviour:
            //   x*0 → 0   0*x → 0   x*1 → x   1*x → x
            //   x+0 → x   0+x → x
            //   x-0 → x
            //   x/1 → x
            if (n->bop == Op::Mul) {
                if (is_number(l) && l->num == 0.0f) return Node::number(0.0f);
                if (is_number(r) && r->num == 0.0f) return Node::number(0.0f);
                if (is_number(l) && l->num == 1.0f) return r;
                if (is_number(r) && r->num == 1.0f) return l;
            } else if (n->bop == Op::Add) {
                if (is_number(l) && l->num == 0.0f) return r;
                if (is_number(r) && r->num == 0.0f) return l;
            } else if (n->bop == Op::Sub) {
                if (is_number(r) && r->num == 0.0f) return l;
            } else if (n->bop == Op::Div) {
                if (is_number(r) && r->num == 1.0f) return l;
            }
            return memo[n.get()] = Node::binop(n->bop, l, r);
        }
        case Kind::Call: {
            std::vector<NodePtr> folded;
            folded.reserve(n->children.size());
            std::vector<float>   nums;
            nums.reserve(n->children.size());
            bool all_const = true;
            for (const auto& c : n->children) {
                auto fc = fold_impl(c, memo);
                folded.push_back(fc);
                if (is_number(fc)) nums.push_back(fc->num);
                else               all_const = false;
            }
            // Only constant-fold builtins; a user template call (unknown to
            // apply_func, which returns NaN for unknown names) must be kept as
            // a Call node with its children folded.
            if (all_const && builtin_arity(n->ident) >= 0) {
                float v = apply_func(n->ident, nums);
                return Node::number(v);
            }
            return memo[n.get()] = Node::call(n->ident, std::move(folded));
        }
    }
    return n;
}

} // namespace frep::expr
