// CPU sphere-tracer benchmark: scalar vs SIMD-8 packet vs SIMD-8 + Lipschitz
// step. Reports wall time, SDF evaluations, and hit count (a parity check —
// the three tracers must agree on which pixels hit).
//
//   bench_trace scene.json [res] [safety]
#include "core/compiler/compile_sdf.hpp"
#include "core/io/scene_io.hpp"
#include "core/render/cpu_trace.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace frep;


template <class F>
static double timed(F&& f, int reps = 3) {
    f();                                             // warmup
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        auto a = std::chrono::steady_clock::now();
        f();
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static long hits(const std::vector<float>& d) {
    long n = 0; for (float v : d) if (v >= 0) ++n; return n;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s scene.json [res] [safety]\n", argv[0]); return 1; }
    const int   R      = argc > 2 ? std::atoi(argv[2]) : 256;
    const float safety = argc > 3 ? float(std::atof(argv[3])) : 0.85f;
    const int   steps  = argc > 4 ? std::atoi(argv[4]) : 256;

    SceneGraph s = io::load_scene(argv[1]);
    auto sc = jit::compile_scene_sdf(s);
    if (!sc) { std::fprintf(stderr, "scalar: %s\n", sc.error().c_str()); return 2; }
    auto sv = jit::compile_scene_sdf_simd(s, 8);
    if (!sv) { std::fprintf(stderr, "simd: %s\n", sv.error().c_str()); return 2; }

    render::Camera  cam;
    render::MarchCfg cfg; cfg.safety = safety; cfg.max_steps = steps;

    auto grid_ms = timed([&]{ (void)render::build_lip_grid(sc->fn, 24, -1.8f, 1.8f); }, 1);
    auto lip = render::build_lip_grid(sc->fn, 24, -1.8f, 1.8f);

    std::vector<float> d0, d1, d2; long s0 = 0, s1 = 0, s2 = 0;
    double t0 = timed([&]{ render::trace_scalar(sc->fn, cam, cfg, R, R, d0, s0); });
    double t1 = timed([&]{ render::trace_simd(sv->fn, sv->width, cam, cfg, R, R, d1, s1); });
    double t2 = timed([&]{ render::trace_simd(sv->fn, sv->width, cam, cfg, R, R, d2, s2, &lip); });

    std::printf("scene=%s res=%d safety=%.2f max_steps=%d  lip_grid_build=%.1fms\n",
                argv[1], R, safety, steps, grid_ms);
    std::printf("  scalar        %8.2f ms  evals=%-10ld hits=%ld\n", t0, s0, hits(d0));
    std::printf("  simd8         %8.2f ms  evals=%-10ld hits=%ld  (%.2fx)\n",
                t1, s1, hits(d1), t0 / t1);
    std::printf("  simd8+lip     %8.2f ms  evals=%-10ld hits=%ld  (%.2fx)  max_L=%.2f%s\n",
                t2, s2, hits(d2), t0 / t2, lip.max_l(),
                lip.worth_using(safety) ? "  [recommended]" : "  [not needed: true SDF]");
    return 0;
}
