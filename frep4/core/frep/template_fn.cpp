// core/frep/template_fn.cpp — see template_fn.hpp.

#include "core/frep/template_fn.hpp"

namespace frep {

std::string TemplateRegistry::add(const std::string& name,
                                  std::vector<std::string> params,
                                  const std::string& body_src) {
    if (find(name))
        return "duplicate template '" + name + "'";
    for (const auto& p : params)
        if (p == "x" || p == "y" || p == "z" || p == "pi" || p == "e")
            return "template '" + name + "': parameter '" + p +
                   "' is a reserved name";

    // The body may use x,y,z, this template's own parameters, and any template
    // defined earlier (definition-order scoping — no forward references).
    expr::ParseScope scope;
    scope.params = params;
    for (const auto& t : fns_)
        scope.templates.push_back({t.name, static_cast<int>(t.params.size())});

    expr::NodePtr body;
    try {
        body = expr::fold(expr::parse(body_src, scope));
    } catch (const expr::ParseError& e) {
        return "template '" + name + "': " + e.what();
    }

    fns_.push_back(TemplateFn{name, std::move(params), body_src, std::move(body)});
    return {};
}

}  // namespace frep
