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
#include <cstdlib>
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

// FREP_PATHS=cpu_ir,gpu_rtx restricts a run to the named paths, so one path can
// be profiled or A/B-tested without the others warming the machine first.
static bool path_selected(const char* name) {
    const char* only = std::getenv("FREP_PATHS");
    if (!only || !*only) return true;
    std::string list = only, want = name;
    for (std::size_t b = 0; b <= list.size();) {
        std::size_t e = list.find(',', b);
        if (e == std::string::npos) e = list.size();
        if (list.compare(b, e - b, want) == 0) return true;
        b = e + 1;
    }
    return false;
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
    // Shading level. This exists because the cross-system render comparison is
    // otherwise not like-for-like: libfive's Heightmap::render and mpr's
    // render3D both produce depth + surface normal and stop there, while our
    // default configuration also runs a shadow march, five ambient-occlusion
    // samples and a Cook-Torrance specular term at every hit. Those are extra
    // evaluations of the whole field per pixel, not extra arithmetic on one.
    // "geometry" turns them off so the work per pixel is what the other systems
    // actually do; "full" is the default the interactive app ships with.
    bool geometry_only = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--energy" && i + 1 < argc) energy_s = std::atof(argv[++i]);
        else if (a == "--geometry") geometry_only = true;
    }

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
    const bool have_cpu = cpu_e && cpu_e->available();
    const bool have_gpu = gpu_e && gpu_e->available();
    if (!have_cpu)
        std::fprintf(stderr, "note: CPU energy unavailable "
                     "(needs kernel.perf_event_paranoid <= 0); reported as -1\n");

    // Idle baseline. Both counters measure the whole device, and neither device
    // is off while the other works: this GPU draws ~18 W and this CPU package
    // ~80 W doing nothing. Reporting raw counter deltas therefore credits a
    // CPU-only render with "GPU energy" that is merely the idle draw of a card
    // sitting there. Measure that floor once and subtract it, so the figures are
    // the marginal cost of rendering rather than the cost of owning the machine.
    double idle_cpu_w = 0, idle_gpu_w = 0;
    if (energy_s > 0) {
        // Let the machine settle first. Measuring the floor immediately after
        // the previous scene's run catches a package that is still hot and
        // boosted, which inflates the baseline and then over-subtracts -- far
        // enough to drive a busy CPU path's marginal energy to zero.
        std::this_thread::sleep_for(std::chrono::seconds(3));
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

    std::printf("scene %s   %dx%d   shading=%s   (ms)\n", path.c_str(), R, R,
                geometry_only ? "geometry (depth+normal, matches libfive/mpr)"
                              : "full (shadows+AO+specular)");
    std::printf("%-9s | %10s | %10s | %10s | %9s | %7s %7s\n",
                "path", "build", "run", "wall", "run ns/ray", "mJ/f", "mJ/f");
    std::printf("          |            |            |            |           |"
                "   (gpu)   (cpu)\n");
    std::printf("----------+------------+------------+------------+-----------+----------------\n");

    for (PathKind pk : paths) {
        const char* name = path_kind_name(pk);
        if (!path_selected(name)) continue;
        TracerConfig cfg;
        if (geometry_only) {
            cfg.enable_shadows  = false;
            cfg.enable_ao       = false;
            cfg.enable_specular = false;
        }
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
        // Warm up by time, not by frame count, before timing. The GPU's boost
        // clock takes a few hundred milliseconds to ramp from idle, and three
        // frames of a sub-millisecond path measure the ramp, not the path —
        // the same build came out 20% apart in back-to-back runs before this.
        // FREP_WARM_MS overrides the window (0 keeps only the WARM frames).
        {
            double warm_ms = 500.0;
            if (const char* s = std::getenv("FREP_WARM_MS")) warm_ms = std::atof(s);
            auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - w0).count() < warm_ms) {
                TileResult r = exec->render(scene, R, R, full);
                if (!r.ok) break;
            }
        }
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
            // Clamp only a *measured* value that fell below the baseline's
            // noise floor. A counter that reported nothing stays at -1, which
            // the table prints as a dash: "unavailable" and "zero" are not the
            // same claim, and collapsing them silently reported a busy CPU path
            // as costing no energy when the RAPL counter was simply locked.
            if (have_cpu && j_cpu < 0) j_cpu = 0;
            if (have_gpu && j_gpu < 0) j_gpu = 0;
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
