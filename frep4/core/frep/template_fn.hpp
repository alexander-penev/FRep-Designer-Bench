#pragma once
// core/frep/template_fn.hpp
//
// User-defined template functions: named, scalar-parameterised sub-expressions
// that other CustomExpr bodies (and each other) can call by name — the "named
// function" abstraction the visual language otherwise lacks.
//
// A template is `name(p0, p1, ...) = body`, where `body` is an ordinary scalar
// expression over the built-in coordinates x,y,z plus the declared parameters.
// It compiles to a real *callable* function on every back-end:
//
//   LLVM/CPU:  float frep_tmpl_<name>(float p0, ..., float x, float y, float z)
//   GLSL/GPU:  float frep_tmpl_<name>(float p0, ..., vec3 p)  // x,y,z from p
//
// A call `name(a0, a1)` inside another expression evaluates the argument
// expressions, then calls the template with (a0, a1, <ambient x>, <y>, <z>).
// Because the body is *called* rather than inlined at every use, a scene that
// repeats a template (e.g. 25 columns) keeps its IR/GLSL small — smaller code
// and faster JIT than one flattened monolithic expression.
//
// Normals: CustomExpr uses finite-difference gradients (it does not emit a dual
// twin), so a template needs only its value emission — the gradient falls out of
// evaluating the value function at offset points.

#include "core/frep/expr_ast.hpp"

#include <optional>
#include <string>
#include <vector>

namespace frep {

// A single template definition. `body` is parsed once with `params` (and any
// earlier templates) in scope; a null `body` means "not yet parsed".
struct TemplateFn {
    std::string              name;      // call name, e.g. "gear"
    std::vector<std::string> params;    // scalar parameter names, in order
    std::string              body_src;  // the expression text (source of truth)
    expr::NodePtr            body;      // parsed AST (filled by TemplateRegistry)
};

// An ordered set of templates. Order is definition order: a template may call
// any template defined before it (no forward references / recursion in v1,
// which keeps compilation a simple topological walk and guarantees the AABB /
// Lipschitz reasoning stays well-founded).
class TemplateRegistry {
public:
    // Look up by name; nullptr if absent.
    const TemplateFn* find(const std::string& name) const {
        for (const auto& t : fns_)
            if (t.name == name) return &t;
        return nullptr;
    }
    int arity(const std::string& name) const {
        const TemplateFn* t = find(name);
        return t ? static_cast<int>(t->params.size()) : -1;
    }
    bool empty() const { return fns_.empty(); }
    const std::vector<TemplateFn>& all() const { return fns_; }

    // Add a template, parsing its body with its own params and every template
    // already in the registry in scope. Returns an error string on failure
    // (duplicate name, reserved name, parse error), empty on success.
    std::string add(const std::string& name,
                    std::vector<std::string> params,
                    const std::string& body_src);

private:
    std::vector<TemplateFn> fns_;
};

}  // namespace frep
