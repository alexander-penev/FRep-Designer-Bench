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
#include "core/power/energy_meter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace frep;
using namespace frep::exec;
using namespace frep::power;

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
    // Seconds to spend per path when measuring energy. NVML integrates power on
    // a ~10-100 ms cadence, so a handful of millisecond-scale frames samples
    // nothing: the window has to be seconds, not frames, or the counter returns
    // zero for the fast paths and noise for the slow ones. 0 disables.
    double energy_s = 0;
    for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--energy" && i + 1 < argc)
            energy_s = std::atof(argv[++i]);

    std::string scene_name = path;
    if (auto sl = scene_name.find_last_of('/'); sl != std::string::npos)
        scene_name = scene_name.substr(sl + 1);
    if (auto d = scene_name.rfind(".json"); d != std::string::npos)
        scene_name = scene_name.substr(0, d);

    SceneGraph scene;
    try { scene = io::load_scene(path); }
    catch (const std::exception& e) { std::fprintf(stderr, "load: %s\n", e.what()); return 1; }

    auto cpu_e = make_cpu_energy_counter();
    auto gpu_e = make_gpu_energy_counter();

    // Idle baseline. Both counters measure the whole device, and neither device
    // is off while the other works: this GPU draws ~18 W and this CPU package
    // ~80 W doing nothing. Reporting raw counter deltas therefore credits a
    // CPU-only render with "GPU energy" that is merely the idle draw of a card
    // sitting there. Measure that floor once and subtract it, so the figures are
    // the marginal cost of rendering rather than the cost of owning the machine.
    double idle_cpu_w = 0, idle_gpu_w = 0;
    if (energy_s > 0) {
        const double t_idle = 3.0;
        if (cpu_e && cpu_e->available()) cpu_e->begin();
        if (gpu_e && gpu_e->available()) gpu_e->begin();
        auto i0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - i0).count() < t_idle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - i0).count();
        if (cpu_e && cpu_e->available()) { if (auto v = cpu_e->end()) idle_cpu_w = *v / el; }
        if (gpu_e && gpu_e->available()) { if (auto v = gpu_e->end()) idle_gpu_w = *v / el; }
        std::printf("idle baseline: cpu %.1f W, gpu %.1f W (subtracted below)\n",
                    idle_cpu_w, idle_gpu_w);
    }

    const PathKind paths[] = { PathKind::CpuIr, PathKind::GpuGlsl,
                               PathKind::GpuIr, PathKind::GpuRtx };
    const int WARM = 3, ITERS = 15;

    std::printf("scene %s   %dx%d   (ms)\n", path.c_str(), R, R);
    std::printf("%-9s | %10s | %10s | %10s | %9s | %7s %7s\n",
                "path", "build", "run", "wall", "run ns/ray", "mJ/f", "mJ/f");
    std::printf("          |            |            |            |           |"
                "   (gpu)   (cpu)\n");
    std::printf("----------+------------+------------+------------+-----------+----------------\n");

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
        // Energy over the timed iterations only. CPU package and GPU are kept
        // apart on purpose: on an idle-heavy host the package draws ~80 W
        // regardless, so lumping it in would price a GPU path by how long the
        // CPU sat waiting for it rather than by what the render cost.
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
        // Separate energy pass: render continuously for energy_s seconds and
        // divide by the frames actually completed, so every path is integrated
        // over the same wall-clock window regardless of how fast it renders.
        double j_cpu = -1, j_gpu = -1, frames = 0;
        if (energy_s > 0) {
            if (cpu_e && cpu_e->available()) cpu_e->begin();
            if (gpu_e && gpu_e->available()) gpu_e->begin();
            auto e0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - e0).count() < energy_s) {
                exec->render(scene, R, R, full);
                frames += 1;
            }
            double el = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - e0).count();
            if (cpu_e && cpu_e->available())
                { if (auto v = cpu_e->end()) j_cpu = *v - idle_cpu_w * el; }
            if (gpu_e && gpu_e->available())
                { if (auto v = gpu_e->end()) j_gpu = *v - idle_gpu_w * el; }
            if (j_cpu < 0) j_cpu = 0;   // below the noise floor of the baseline
            if (j_gpu < 0) j_gpu = 0;
        }
        if (frames <= 0) frames = 1;
        double run = med(run_ms), wall = med(wall_ms), wb = med(warmbuild_ms);
        double ns = run * 1e6 / ((double)R * R);
        std::printf("%-9s | %10.2f | %10.3f | %10.3f | %9.1f | %7.2f %7.2f\n",
                    name, build_ms, run, wall, ns,
                    j_gpu >= 0 ? j_gpu / frames * 1000.0 : -1.0,
                    j_cpu >= 0 ? j_cpu / frames * 1000.0 : -1.0);
        (void)wb;
        std::fflush(stdout);
        // Machine-readable copy on stderr, so a driver script can capture the
        // numbers while the table above stays readable. Columns follow the
        // suite's CSV: system,backend,scene,metric,size,ms,throughput,...
        // ms = wall (the per-frame cost a user sees); build and run are the
        // separated halves the paper reports.
        std::fprintf(stderr, "frep4,%s,%s,render,%d,%.3f,%.3f,%.4f,%.4f,%.2f,%.3f\n",
                     name, scene_name.c_str(), R, wall,
                     wall > 0 ? (double)R * R / wall / 1e3 : -1.0,
                     j_gpu >= 0 ? j_gpu / frames : -1.0,
                     j_cpu >= 0 ? j_cpu / frames : -1.0, build_ms, run);
    }
    return 0;
}
