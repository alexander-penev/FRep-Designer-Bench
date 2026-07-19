// tmpl_check.cpp — Phase 1a verification for user-defined template functions.
//
// Defines templates, compiles caller expressions that call them, JITs the
// result on the CPU path, and checks the numeric output against the hand
// evaluation. Proves: parser accepts scalar params + template calls; the
// compiler binds params to function args and lowers a template call to a real
// CreateCall of the emitted frep_tmpl_<name> function; and a template may call
// an earlier template.

#include "core/frep/custom_expr.hpp"
#include "core/frep/template_fn.hpp"
#include "core/frep/expr_ast.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/io/scene_io.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/compile_sdf.hpp"
#include "core/compiler/sdf_validate.hpp"
#include "core/frep/smart_import.hpp"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using namespace frep;
using CallerFn = float (*)(float, float, float);

static int g_failures = 0;

static void check(const char* what, float got, float want) {
    float d = std::fabs(got - want);
    bool ok = d < 1e-4f;
    std::printf("  %-32s got=%9.5f want=%9.5f  %s\n",
                what, got, want, ok ? "OK" : "FAIL");
    if (!ok) ++g_failures;
}

// Emit the registry's templates + a caller expression into one module, JIT it,
// and hand back the callable. `eng` must outlive the returned pointer.
static CallerFn build(std::unique_ptr<JitEngine>& eng,
                      const TemplateRegistry& reg,
                      const std::string& caller_src) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("tmpl_test", *ctx);

    CustomExprCompiler comp;
    comp.set_templates(&reg);
    if (!comp.emit_templates(*mod, *ctx)) {
        std::printf("  emit_templates: %s\n", comp.last_error().c_str());
        return nullptr;
    }

    expr::ParseScope scope;
    for (const auto& t : reg.all())
        scope.templates.push_back({t.name, static_cast<int>(t.params.size())});
    expr::NodePtr ast;
    try {
        ast = expr::parse(caller_src, scope);
    } catch (const expr::ParseError& e) {
        std::printf("  parse '%s': %s\n", caller_src.c_str(), e.what());
        return nullptr;
    }

    auto* fn = comp.compile(*mod, *ctx, "caller", ast);
    if (!fn) {
        std::printf("  compile '%s': %s\n", caller_src.c_str(),
                    comp.last_error().c_str());
        return nullptr;
    }
    fn->setLinkage(llvm::Function::ExternalLinkage);  // JIT-visible entry

    eng = std::make_unique<JitEngine>();
    auto r = eng->load_as<CallerFn>(std::move(mod), std::move(ctx), "caller");
    if (!r) { std::printf("  jit: %s\n", r.error().c_str()); return nullptr; }
    return *r;
}

// Evaluate a caller expression through the AST interpreter (no JIT), with the
// registry resolving template calls — this is the FRepNode::eval path.
static float interp(const TemplateRegistry& reg, const std::string& src,
                    float x, float y, float z) {
    expr::ParseScope scope;
    for (const auto& t : reg.all())
        scope.templates.push_back({t.name, static_cast<int>(t.params.size())});
    auto ast = expr::parse(src, scope);
    return CustomExprNode::eval_ast(*ast, x, y, z, nullptr, &reg);
}

using VecFn = void (*)(const float*, const float*, const float*, float*);

// JIT the W-wide SIMD form of a caller expression (compile_vec), template calls
// inlined via the registry.
static VecFn build_vec(std::unique_ptr<JitEngine>& eng, const TemplateRegistry& reg,
                       const std::string& src, unsigned W) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("tmpl_vec", *ctx);
    CustomExprCompiler comp;
    comp.set_templates(&reg);
    expr::ParseScope scope;
    for (const auto& t : reg.all())
        scope.templates.push_back({t.name, static_cast<int>(t.params.size())});
    expr::NodePtr ast;
    try { ast = expr::parse(src, scope); }
    catch (const expr::ParseError& e) { std::printf("  vparse: %s\n", e.what()); return nullptr; }
    auto* fn = comp.compile_vec(*mod, *ctx, "caller_vec", ast, W);
    if (!fn) { std::printf("  compile_vec: %s\n", comp.last_error().c_str()); return nullptr; }
    eng = std::make_unique<JitEngine>();
    auto r = eng->load_as<VecFn>(std::move(mod), std::move(ctx), "caller_vec");
    if (!r) { std::printf("  jit: %s\n", r.error().c_str()); return nullptr; }
    return *r;
}

using IvalFn = void (*)(const float*, float*);

// JIT the interval form (compile_interval): fn(B[6], O[2]), B=[xlo,xhi,...].
static IvalFn build_ival(std::unique_ptr<JitEngine>& eng, const TemplateRegistry* reg,
                         const std::string& src) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("tmpl_ival", *ctx);
    CustomExprCompiler comp;
    expr::ParseScope scope;
    if (reg) {
        comp.set_templates(reg);
        for (const auto& t : reg->all())
            scope.templates.push_back({t.name, static_cast<int>(t.params.size())});
    }
    expr::NodePtr ast;
    try { ast = reg ? expr::parse(src, scope) : expr::parse(src); }
    catch (const expr::ParseError& e) { std::printf("  iparse: %s\n", e.what()); return nullptr; }
    auto* fn = comp.compile_interval(*mod, *ctx, "iv", ast);
    if (!fn) { std::printf("  compile_interval: %s\n", comp.last_error().c_str()); return nullptr; }
    eng = std::make_unique<JitEngine>();
    auto r = eng->load_as<IvalFn>(std::move(mod), std::move(ctx), "iv");
    if (!r) { std::printf("  jit: %s\n", r.error().c_str()); return nullptr; }
    return *r;
}

int main() {
    TemplateRegistry reg;
    std::string err;
    // A sphere of radius r, centred at the origin.
    err = reg.add("blob", {"r"}, "sqrt(x*x+y*y+z*z) - r");
    if (!err.empty()) { std::printf("add blob: %s\n", err.c_str()); return 1; }
    // A template that CALLS an earlier template: union (min) of two blobs, the
    // second shifted by +2 in x.
    err = reg.add("two", {"r"},
                  "min(blob(r), sqrt((x-2)*(x-2)+y*y+z*z) - r)");
    if (!err.empty()) { std::printf("add two: %s\n", err.c_str()); return 1; }

    std::unique_ptr<JitEngine> e1, e2, e3;

    std::printf("A. blob(1.5) == |p| - 1.5\n");
    if (auto f = build(e1, reg, "blob(1.5)")) {
        check("blob(1.5) @ (1,0,0)", f(1, 0, 0), 1.0f - 1.5f);
        check("blob(1.5) @ (3,0,0)", f(3, 0, 0), 3.0f - 1.5f);
        check("blob(1.5) @ (0,0,0)", f(0, 0, 0), -1.5f);
        check("blob(1.5) @ (0,4,0)", f(0, 4, 0), 4.0f - 1.5f);
    }

    std::printf("B. same template, different arg: blob(0.5)\n");
    if (auto f = build(e2, reg, "blob(0.5)"))
        check("blob(0.5) @ (2,0,0)", f(2, 0, 0), 2.0f - 0.5f);

    std::printf("C. template calls template: two(1.0)\n");
    if (auto f = build(e3, reg, "two(1.0)")) {
        // (1,0,0): min(|(1,0,0)|-1, |(-1,0,0)|-1) = min(0, 0) = 0
        check("two(1.0) @ (1,0,0)", f(1, 0, 0), 0.0f);
        // (2,0,0): min(2-1, 0-1) = min(1, -1) = -1
        check("two(1.0) @ (2,0,0)", f(2, 0, 0), -1.0f);
    }

    std::printf("D. interpreter (eval_ast) matches hand + JIT\n");
    check("interp blob(1.5) @ (3,0,0)", interp(reg, "blob(1.5)", 3, 0, 0), 1.5f);
    check("interp two(1.0) @ (2,0,0)",  interp(reg, "two(1.0)",  2, 0, 0), -1.0f);
    // Cross-check interpreter vs JIT at an off-axis point.
    if (auto f = build(e1, reg, "blob(1.5)"))
        check("interp==JIT blob(1.5) @ (1,2,2)",
              interp(reg, "blob(1.5)", 1, 2, 2), f(1, 2, 2));

    std::printf("E. SIMD (compile_vec, W=8) matches interpreter, lane by lane\n");
    std::unique_ptr<JitEngine> ev;
    if (auto vf = build_vec(ev, reg, "two(1.0)", 8)) {
        float X[8] = {0, 1, 2, 3, -1, 0.5f, 2, 1};
        float Y[8] = {0, 0, 0, 1,  2, 0.0f, 0, 2};
        float Z[8] = {0, 0, 0, 0,  0, 1.0f, 1, 0};
        float O[8] = {0};
        vf(X, Y, Z, O);
        for (int i : {0, 3, 5, 7}) {
            char lbl[48];
            std::snprintf(lbl, sizeof lbl, "simd two(1.0) lane %d", i);
            check(lbl, O[i], interp(reg, "two(1.0)", X[i], Y[i], Z[i]));
        }
    }

    std::printf("F. interval (compile_interval): template call == inlined expr\n");
    std::unique_ptr<JitEngine> ei1, ei2;
    auto tf = build_ival(ei1, &reg,   "blob(1.5)");
    auto rf = build_ival(ei2, nullptr, "sqrt(x*x+y*y+z*z) - 1.5");
    if (tf && rf) {
        float B[6] = {0, 1, 0, 1, 0, 1};   // box [0,1]^3
        float Ot[2] = {0}, Or[2] = {0};
        tf(B, Ot); rf(B, Or);
        check("interval lo (blob vs inlined)", Ot[0], Or[0]);
        check("interval hi (blob vs inlined)", Ot[1], Or[1]);
    }

    std::printf("G. GLSL emission: template defs + calls well-formed\n");
    {
        std::ostringstream defs;
        CustomExprNode::emit_templates_glsl(defs, reg);
        const std::string d = defs.str();
        auto has = [&](const char* s) { return d.find(s) != std::string::npos ? 1.f : 0.f; };
        check("glsl def frep_tmpl_blob",
              has("float frep_tmpl_blob(float r, float x, float y, float z)"), 1);
        check("glsl def frep_tmpl_two",
              has("float frep_tmpl_two(float r, float x, float y, float z)"), 1);
        check("glsl two() calls blob()", has("frep_tmpl_blob(r, x, y, z)"), 1);

        expr::ParseScope scope;
        for (const auto& t : reg.all())
            scope.templates.push_back({t.name, static_cast<int>(t.params.size())});
        std::ostringstream call;
        CustomExprNode::emit_glsl_ast(call, *expr::parse("blob(1.5)", scope), &reg);
        float ok = call.str().find("frep_tmpl_blob(1.5, x, y, z)") != std::string::npos;
        check("glsl call blob(1.5)", ok, 1);
        if (!ok) std::printf("    call emitted: %s\n", call.str().c_str());
    }

    std::printf("H. end-to-end scene: scene_sdf(template) == scene_sdf(inlined)\n");
    {
        SceneGraph sa;                       // scene using the template
        auto na = std::make_shared<CustomExprNode>("blob(1.5)", "o");
        na->set_templates(&reg);
        sa.add_object(na);

        SceneGraph sb;                       // reference: the inlined expression
        auto nb = std::make_shared<CustomExprNode>("sqrt(x*x+y*y+z*z) - 1.5", "o");
        sb.add_object(nb);

        auto ca = frep::jit::compile_scene_sdf(sa);
        auto cb = frep::jit::compile_scene_sdf(sb);
        if (!ca) std::printf("  compile template scene: %s\n", ca.error().c_str());
        if (!cb) std::printf("  compile inlined scene:  %s\n", cb.error().c_str());
        if (ca && cb) {
            struct P { float x, y, z; } pts[] = {
                {1, 0, 0}, {0, 2, 0}, {1, 1, 1}, {-2, 0.5f, 0.3f}, {0, 0, 0}};
            for (auto p : pts) {
                char lbl[64];
                std::snprintf(lbl, sizeof lbl, "scene_sdf @(%.1f,%.1f,%.1f)",
                              p.x, p.y, p.z);
                check(lbl, ca->fn(p.x, p.y, p.z), cb->fn(p.x, p.y, p.z));
            }
            // FRepNode::eval path (picker / marching cubes).
            check("node.eval template==inlined @(1,1,1)",
                  na->eval(1, 1, 1), nb->eval(1, 1, 1));
        }
    }

    std::printf("I. serialization round-trip: template scene survives JSON\n");
    {
        SceneGraph s;
        std::string e = s.templates().add("blob", {"r"}, "sqrt(x*x+y*y+z*z) - r");
        if (!e.empty()) std::printf("  add: %s\n", e.c_str());
        s.add_object(std::make_shared<CustomExprNode>("blob(1.5)", "o"));
        s.wire_templates();

        std::string js = frep::io::serialize_scene(s);
        check("json carries template blob",
              js.find("\"blob\"") != std::string::npos ? 1.f : 0.f, 1);

        SceneGraph s2 = frep::io::deserialize_scene(js, nullptr, "");
        auto c2 = frep::jit::compile_scene_sdf(s2);
        if (!c2) std::printf("  compile round-trip: %s\n", c2.error().c_str());
        if (c2) {
            check("roundtrip scene_sdf @(1,1,1)", c2->fn(1, 1, 1), std::sqrt(3.f) - 1.5f);
            check("roundtrip scene_sdf @(2,0,0)", c2->fn(2, 0, 0), 0.5f);
        }

        // Dump a template scene and its inlined equivalent for the GPU render
        // parity check (bench_render on real hardware).
        { std::ofstream f("/tmp/claude-1000/tmpl_scene.json");    f << js; }
        SceneGraph si;
        si.add_object(std::make_shared<CustomExprNode>("sqrt(x*x+y*y+z*z) - 1.5", "o"));
        { std::ofstream f("/tmp/claude-1000/inlined_scene.json");
          f << frep::io::serialize_scene(si); }
    }

    std::printf("J. SDF validation (smart-import guard)\n");
    {
        auto mkscene = [](const std::string& expr) {
            SceneGraph s;
            s.add_object(std::make_shared<CustomExprNode>(expr, "o"));
            return s;
        };
        auto r1 = frep::jit::validate_sdf(mkscene("sqrt(x*x+y*y+z*z) - 1"));
        std::printf("   metric sphere : finite=%d metric=%d L=%.2f  %s\n",
                    r1.finite, r1.metric, r1.lipschitz, r1.note.c_str());
        check("sphere finite", r1.finite ? 1.f : 0.f, 1);
        check("sphere metric", r1.metric ? 1.f : 0.f, 1);

        auto r2 = frep::jit::validate_sdf(mkscene("(sqrt(x*x+y*y+z*z) - 1) * 3"));
        std::printf("   x3 field      : finite=%d metric=%d L=%.2f  %s\n",
                    r2.finite, r2.metric, r2.lipschitz, r2.note.c_str());
        check("x3 finite", r2.finite ? 1.f : 0.f, 1);
        check("x3 NOT metric", r2.metric ? 1.f : 0.f, 0);

        auto r3 = frep::jit::validate_sdf(mkscene("sqrt(x) - 0.5"));  // NaN for x<0
        std::printf("   broken sqrt(x): finite=%d nan=%.0f%%  %s\n",
                    r3.finite, r3.nan_frac * 100, r3.note.c_str());
        check("sqrt(x) flagged non-finite", r3.finite ? 1.f : 0.f, 0);
        check("sqrt(x) ~half NaN", r3.nan_frac > 0.3 ? 1.f : 0.f, 1);
    }

    std::printf("L. JIT cost: template (called once) vs inlined monolith\n");
    {
        const std::string HEAVY =
            "sin(3*x)*cos(3*y)+sin(3*y)*cos(3*z)+sin(3*z)*cos(3*x)"
            "+sin(5*x)*cos(5*y)+sin(5*y)*cos(5*z)+sin(5*z)*cos(5*x)";
        const int N = 24;
        auto jit_ms = [](const SceneGraph& s) {
            auto t0 = std::chrono::steady_clock::now();
            auto c = frep::jit::compile_scene_sdf(s);
            auto t1 = std::chrono::steady_clock::now();
            (void)c;
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        // Template: N callers of one shared frep_tmpl_heavy body.
        TemplateRegistry hr;
        hr.add("heavy", {"r"}, HEAVY + " - r");
        SceneGraph st;
        st.templates() = hr;
        for (int i = 0; i < N; ++i) {
            char e[64]; std::snprintf(e, sizeof e, "heavy(%.4f)", 0.05f * i + 0.1f);
            char id[16]; std::snprintf(id, sizeof id, "o%d", i);
            st.add_object(std::make_shared<CustomExprNode>(e, id));
        }
        st.wire_templates();
        // Inlined: N distinct copies of the full heavy body.
        SceneGraph si;
        for (int i = 0; i < N; ++i) {
            char e[256]; std::snprintf(e, sizeof e, "(%s) - %.4f", HEAVY.c_str(), 0.05f * i + 0.1f);
            char id[16]; std::snprintf(id, sizeof id, "o%d", i);
            si.add_object(std::make_shared<CustomExprNode>(e, id));
        }
        double best_t = 1e9, best_i = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            best_t = std::min(best_t, jit_ms(st));
            best_i = std::min(best_i, jit_ms(si));
        }
        std::printf("   %d objects: template JIT %.0f ms  vs inlined %.0f ms  (%.2fx)\n",
                    N, best_t, best_i, best_i / best_t);
        check("template JIT faster than inlined", best_t < best_i ? 1.f : 0.f, 1);
    }

    std::printf("K. factor_instances: repetition recovery, field unchanged\n");
    {
        auto build_grid = []() {
            std::vector<FRepNode::Ptr> sp;
            const int G = 4; const float s = 1.5f;
            for (int i = 0; i < G; ++i)
                for (int j = 0; j < G; ++j) {
                    auto sph = std::make_shared<SphereNode>(0.4f);
                    sp.push_back(std::make_shared<TranslateNode>(
                        sph, (i - 1.5f) * s, 0.f, (j - 1.5f) * s));
                }
            SceneGraph sc; sc.add_object(union_all(std::move(sp))); return sc;
        };
        SceneGraph sc = build_grid();
        auto before = frep::jit::compile_scene_sdf(sc);
        int n = frep::factor_instances(sc);
        auto after = frep::jit::compile_scene_sdf(sc);
        std::printf("   16 spheres -> %d instances of 1 shared prototype\n", n);
        check("created 16 instances", (float)n, 16);
        if (before && after) {
            struct P { float x, y, z; } pts[] = {
                {2.25f, 0, 2.25f}, {0, 0, 0}, {-2.25f, 0.2f, 0.75f}, {1, 0.3f, -1}};
            bool ok = true;
            for (auto p : pts)
                if (std::fabs(before->fn(p.x, p.y, p.z) - after->fn(p.x, p.y, p.z)) > 1e-5f)
                    ok = false;
            check("field identical after factoring", ok ? 1.f : 0.f, 1);
        }
        std::ofstream f("/tmp/claude-1000/factored_scene.json");
        f << frep::io::serialize_scene(sc);   // instances + hidden prototype
    }

    // Dump a union-of-spheres scene (a 4x4 grid) for the multi-BLAS RTX path in
    // the render bench: union_all makes a UnionNode chain that partition_csg_groups
    // splits into 16 groups -> a 16-instance TLAS.
    {
        std::vector<FRepNode::Ptr> spheres;
        const int G = 4; const float spacing = 1.5f;
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                auto s = std::make_shared<SphereNode>(0.4f);
                float x = (i - (G - 1) / 2.0f) * spacing;
                float z = (j - (G - 1) / 2.0f) * spacing;
                spheres.push_back(std::make_shared<TranslateNode>(s, x, 0.f, z));
            }
        SceneGraph su;
        su.add_object(union_all(std::move(spheres)));
        std::ofstream f("/tmp/claude-1000/union_scene.json");
        f << frep::io::serialize_scene(su);
        std::printf("(wrote union_scene.json: 16-sphere grid for multi-BLAS RTX)\n");
    }

    std::printf("\n%s (%d failure%s)\n",
                g_failures ? "TEMPLATE TEST FAILED" : "TEMPLATE TEST PASSED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
