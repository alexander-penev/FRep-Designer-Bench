// pathcost — separate BUILD from RUN for every render path, one honest table.
//
// The motivating bug: the paper's render table showed GPU_RTX ~30-38 ms at 512²
// ("setup excluded"), 30x its actual per-frame trace. Every path has a build
// (CPU_IR: LLVM JIT; GPU_GLSL: GLSL->SPIR-V+pipeline; GPU_IR: IR->NVPTX+module;
// GPU_RTX: RT pipeline+SBT+BVH). RTX's build is ~25 ms vs ~1 ms for the others,
// so when build leaks into "render" only RTX looks slow. This measures the two
// separately and consistently for all four paths:
//
//   build = compile_ms of the FIRST render (cold: codegen/JIT/pipeline/BVH).
//   run   = median render_ms of WARM renders (the recurring per-frame dispatch).
//   wall  = median wall-clock of the warm render() call (run + readback + crop),
//           the number a naive timer would report if it timed the whole call.
//
// If "RTX is slow" is a build artifact, run collapses while build stays large.
#include "core/exec/multipath.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/io/scene_io.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace frep;
using namespace frep::exec;

static double med(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <scene.json> [res]\n", argv[0]); return 2; }
    std::string path = argv[1];
    int R = argc > 2 ? std::atoi(argv[2]) : 512;
    if (R <= 0) R = 512;

    SceneGraph scene;
    try { scene = io::load_scene(path); }
    catch (const std::exception& e) { std::fprintf(stderr, "load: %s\n", e.what()); return 1; }

    const PathKind paths[] = { PathKind::CpuIr, PathKind::GpuGlsl,
                               PathKind::GpuIr, PathKind::GpuRtx };
    const int WARM = 3, ITERS = 15;

    std::printf("scene %s   %dx%d   (ms)\n", path.c_str(), R, R);
    std::printf("%-9s | %10s | %10s | %10s | %s\n",
                "path", "build", "run", "wall", "run ns/ray");
    std::printf("----------+------------+------------+------------+-----------\n");

    for (PathKind pk : paths) {
        const char* name = path_kind_name(pk);
        TracerConfig cfg;
        auto exec = make_executor(pk, cfg);
        if (!exec || !exec->available()) {
            std::printf("%-9s | %10s\n", name, "unavailable");
            continue;
        }
        Tile full{0, 0, R, R};

        TileResult first = exec->render(scene, R, R, full);
        if (!first.ok) { std::printf("%-9s | error: %s\n", name, first.error.c_str()); continue; }
        double build_ms = first.compile_ms;

        std::vector<double> run_ms, wall_ms, warmbuild_ms;
        for (int i = 0; i < WARM + ITERS; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            TileResult r = exec->render(scene, R, R, full);
            auto t1 = std::chrono::steady_clock::now();
            if (!r.ok) break;
            if (i >= WARM) {
                run_ms.push_back(r.render_ms);
                warmbuild_ms.push_back(r.compile_ms);  // >0 on warm => cache miss / rebuild
                wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        double run = med(run_ms), wall = med(wall_ms), wb = med(warmbuild_ms);
        double ns = run * 1e6 / ((double)R * R);
        std::printf("%-9s | %10.2f | %10.3f | %10.3f | %9.1f  (warm build %.2f)\n",
                    name, build_ms, run, wall, ns, wb);
        std::fflush(stdout);
    }
    return 0;
}
