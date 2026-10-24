// Randomized soundness test for the interval twin: for random boxes, the
// compiled {lo,hi} must enclose densely sampled values of the scalar function.
#include "core/frep/custom_expr.hpp"
#include "core/frep/expr_ast.hpp"
#include "core/compiler/jit_engine.hpp"
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

using namespace frep;

using IvFn = void (*)(const float*, float*);

static IvFn build(const char* e, JitEngine& eng) {
    auto ast = expr::fold(expr::parse(e));
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("iv", *ctx);
    CustomExprCompiler c;
    if (!c.compile_interval(*mod, *ctx, "iv", ast)) {
        std::printf("  compile fail: %s\n", c.last_error().c_str());
        return nullptr;
    }
    auto fn = eng.load_as<IvFn>(std::move(mod), std::move(ctx), "iv");
    return fn ? *fn : nullptr;
}

// Dense sampling of the reference inside the box.
template <class Ref>
static bool check(const char* name, const char* expr, Ref ref,
                  float span, int boxes = 400, int samples = 12) {
    JitEngine eng;
    IvFn iv = build(expr, eng);
    if (!iv) return false;
    std::mt19937 g(99);
    std::uniform_real_distribution<float> c(-4.0f, 4.0f), w(0.01f, span);
    long viol = 0;
    double width_sum = 0;
    for (int t = 0; t < boxes; ++t) {
        float cx = c(g), cy = c(g), cz = c(g);
        float hx = w(g), hy = w(g), hz = w(g);
        float B[6] = {cx-hx, cx+hx, cy-hy, cy+hy, cz-hz, cz+hz}, O[2];
        iv(B, O);
        if (!std::isfinite(O[0]) || !std::isfinite(O[1])) { width_sum += 1e9; continue; }
        width_sum += (O[1] - O[0]);
        for (int i = 0; i <= samples; ++i)
        for (int j = 0; j <= samples; ++j)
        for (int k = 0; k <= samples; ++k) {
            float x = B[0] + (B[1]-B[0]) * i / samples;
            float y = B[2] + (B[3]-B[2]) * j / samples;
            float z = B[4] + (B[5]-B[4]) * k / samples;
            float f = ref(x, y, z);
            if (!std::isfinite(f)) continue;
            if (f < O[0] - 2e-4f || f > O[1] + 2e-4f) ++viol;
        }
    }
    std::printf("  %-22s violations=%-6ld mean_width=%.3g\n",
                name, viol, width_sum / boxes);
    return viol == 0;
}

// One box, one claim about the bound: the rotation rules must not only be
// sound (checked above) but actually bite where the term-by-term rule does not.
template <class Pred>
static bool tight(const char* name, const char* expr, const float (&B)[6], Pred pred) {
    JitEngine eng;
    IvFn iv = build(expr, eng);
    if (!iv) return false;
    float O[2];
    iv(B, O);
    const bool ok = pred(O[0], O[1]);
    std::printf("  %-22s [%g, %g] over x[%g,%g] y[%g,%g] z[%g,%g]  %s\n",
                name, O[0], O[1], B[0], B[1], B[2], B[3], B[4], B[5], ok ? "ok" : "LOOSE");
    return ok;
}

int main() {
    bool ok = true;
    std::printf("interval enclosure (random boxes, dense samples):\n");
    ok &= check("sin(x)",        "sin(x)",      [](float x,float,float){ return std::sin(x); }, 2.0f);
    ok &= check("cos(y)",        "cos(y)",      [](float,float y,float){ return std::cos(y); }, 2.0f);
    ok &= check("tan(x)",        "tan(x)",      [](float x,float,float){ return std::tan(x); }, 0.6f);
    ok &= check("atan2(y,x)",    "atan2(y,x)",  [](float x,float y,float){ return std::atan2(y,x); }, 1.5f);
    ok &= check("asin(x*0.2)",   "asin(x*0.2)", [](float x,float,float){ return std::asin(x*0.2f); }, 0.8f);
    ok &= check("x*x - y*y",     "x*x-y*y",     [](float x,float y,float){ return x*x - y*y; }, 1.0f);
    // Rotation pair and rotated box (the twisted bar of the benchmark), in the
    // "0.0 - p" spelling the scene files use for a leading minus.
    const char* kPair  = "x*cos(1.2*z)+y*sin(1.2*z)";
    const char* kPairN = "0.0-x*sin(1.2*z)+y*cos(1.2*z)";
    const char* kTwist = "max(max(abs(x*cos(1.2*z)+y*sin(1.2*z))-0.35, abs(0.0-x*sin(1.2*z)+y*cos(1.2*z))-0.35), abs(z)-1.1)";
    ok &= check("rotation pair",  kPair,  [](float x,float y,float z){ return x*std::cos(1.2f*z)+y*std::sin(1.2f*z); }, 2.0f);
    ok &= check("rotation pair -", kPairN, [](float x,float y,float z){ return -x*std::sin(1.2f*z)+y*std::cos(1.2f*z); }, 2.0f);
    ok &= check("twisted bar",    kTwist, [](float x,float y,float z){
        float c=std::cos(1.2f*z), s=std::sin(1.2f*z);
        return std::max(std::max(std::fabs(x*c+y*s)-0.35f, std::fabs(-x*s+y*c)-0.35f), std::fabs(z)-1.1f); }, 2.0f);
    std::printf("rotation rules bite:\n");
    // θ = 1.2z spans 4.8 rad, so sin and cos each cover [-1,1] and the plain sum
    // gives ±0.4; the pair rule gives ±sqrt(0.2²+0.2²) = ±0.283.
    const float small[6] = {0.0f, 0.2f, 0.0f, 0.2f, 0.0f, 4.0f};
    ok &= tight("pair width",    kPair,  small, [](float lo, float hi){ return lo >= -0.29f && hi <= 0.29f; });
    ok &= tight("pair width -",  kPairN, small, [](float lo, float hi){ return lo >= -0.29f && hi <= 0.29f; });
    // Far from the axis both rotated coordinates straddle 0 over the box, so the
    // plain max is -0.35; the box rule gives sqrt((4+4)/2) - 0.35 > 0: outside.
    const float far[6] = {2.0f, 3.0f, 2.0f, 3.0f, 0.0f, 1.0f};
    ok &= tight("twisted bar far", kTwist, far, [](float lo, float){ return lo > 1.0f; });
    // The cell that decides the inferred box's ±0.5: the bar's corners reach
    // 0.35·√2 = 0.495, so x in [0.5, 1] holds no surface.
    const float edge[6] = {0.5f, 1.0f, -0.5f, 0.0f, 0.0f, 0.5f};
    ok &= tight("twisted bar edge", kTwist, edge, [](float lo, float){ return lo > 0.0f; });
    std::printf("%s\n", ok ? "ALL SOUND" : "UNSOUND");
    return ok ? 0 : 1;
}
