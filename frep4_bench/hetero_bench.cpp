// Direct measurement of the heterogeneous CPU+GPU frame split — the number the
// paper's additivity argument predicts. For each scene it measures, steady
// state (warm executors, median of 5, setup excluded):
//
//   1. cpu_ir alone and gpu_glsl alone, through the SAME MultiPathExecutor
//      harness (WholeFrame + stitch), so orchestration cost is in every row;
//   2. a weighted two-band split (bands proportional to each path's measured
//      throughput), tile assembly included in the wall time;
//   3. a dynamic work-stealing grid split (no calibration), ditto.
//
// It also reports the additive ideal T = T_cpu*T_gpu/(T_cpu+T_gpu), the
// realized fraction of that ideal, and the parity of the stitched frame
// against the CPU_IR reference (the seam must be invisible).
//
// CSV columns match the suite: system,backend,scene,metric,size,ms,throughput.
#include "core/exec/multipath.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/io/scene_io.hpp"
#include "../common/timing.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace frep;
using namespace frep::exec;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <res> <scene.json>...\n"
            "  measures the CPU+GPU heterogeneous split per scene.\n", argv[0]);
        return 2;
    }
    int R = std::atoi(argv[1]);
    if (R <= 0) R = 800;

    for (int a = 2; a < argc; ++a) {
        std::string path = argv[a];
        std::string name = path;
        auto slash = name.find_last_of('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);
        auto dot = name.rfind(".json");
        if (dot != std::string::npos) name = name.substr(0, dot);

        SceneGraph scene;
        try {
            scene = io::load_scene(path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", path.c_str(), e.what());
            continue;
        }

        TracerConfig cfg;
        auto cpu = make_executor(PathKind::CpuIr, cfg);
        auto gpu = make_executor(PathKind::GpuGlsl, cfg);
        if (!cpu || !cpu->available() || !gpu || !gpu->available()) {
            std::fprintf(stderr, "  %-16s needs both cpu_ir and gpu_glsl — skipped\n",
                         name.c_str());
            continue;
        }

        Tile full{0, 0, R, R};

        // Warm both executors: pays JIT / shader+pipeline setup once, exactly
        // like the steady-state policy of every other table.
        TileResult ref = cpu->render(scene, R, R, full);
        TileResult gw  = gpu->render(scene, R, R, full);
        if (!ref.ok || !gw.ok) {
            std::fprintf(stderr, "  %-16s warmup failed: %s\n", name.c_str(),
                         (!ref.ok ? ref.error : gw.error).c_str());
            continue;
        }

        WholeFrame     whole;
        OnePathPerTile one;
        StitchMerge    stitch;
        MultiPathExecutor solo(whole, one, stitch);

        std::vector<IExecutor*> just_cpu{cpu.get()};
        std::vector<IExecutor*> just_gpu{gpu.get()};
        std::vector<IExecutor*> both{cpu.get(), gpu.get()};

        // Calibration pre-pass: rough per-path medians to derive the band
        // weights (and the initial clock warm-up for both devices).
        double c_cpu = median_ms([&] { cpu->render(scene, R, R, full); }, 3);
        double c_gpu = median_ms([&] { gpu->render(scene, R, R, full); }, 3);

        // During the split the CPU band leaves two hardware threads free for
        // the GPU path's submit/readback thread.
        unsigned hw = std::thread::hardware_concurrency();
        char nthr[16]; std::snprintf(nthr, sizeof nthr, "%u", hw > 4 ? hw - 2 : hw);
        WeightedBands wb({1.0 / c_cpu, 1.0 / c_gpu});
        MultiPathExecutor split(wb, one, stitch);
        int tile_px = R / 4; if (tile_px < 64) tile_px = 64;
        GridDecompose grid(R, tile_px);
        MultiPathExecutor dyn(grid, one, stitch);

        // Interleaved rounds: one sample of every mode per round, medians per
        // mode. GPU clocks drift between thermal states, so measuring the
        // modes in separate blocks minutes apart would bias the comparison;
        // interleaving spreads any drift evenly across all of them.
        const int ROUNDS = 11;
        auto med = [](std::vector<double>& v) {
            std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::vector<double> v_cpu, v_gpu, v_split, v_dyn;
        using clk = std::chrono::steady_clock;
        auto time1 = [&](auto&& f) {
            auto a = clk::now(); f();
            return std::chrono::duration<double, std::milli>(clk::now() - a).count();
        };
        for (int i = 0; i < ROUNDS; ++i) {
            v_cpu.push_back(time1([&] { cpu->render(scene, R, R, full); }));
            v_gpu.push_back(time1([&] { gpu->render(scene, R, R, full); }));
            setenv("FREP4_CPU_THREADS", nthr, 1);
            v_split.push_back(time1([&] {
                split.run(scene, R, R, both, RunMode::Concurrent); }));
            v_dyn.push_back(time1([&] {
                dyn.run(scene, R, R, both, RunMode::DynamicQueue); }));
            unsetenv("FREP4_CPU_THREADS");
        }
        double t_cpu_direct = med(v_cpu), t_gpu_direct = med(v_gpu);
        double t_split = med(v_split);
        double ideal = t_cpu_direct * t_gpu_direct / (t_cpu_direct + t_gpu_direct);

        // One inspection run: band shares, per-band render times, and the
        // stitched-frame parity vs CPU_IR.
        auto insp = split.run(scene, R, R, both, RunMode::Concurrent);
        int cpu_rows = 0, gpu_rows = 0;
        double cpu_band_ms = 0, gpu_band_ms = 0;
        for (const auto& t : insp.tiles) {
            if (t.path == PathKind::CpuIr) { cpu_rows += t.tile.height(); cpu_band_ms = t.render_ms; }
            else                           { gpu_rows += t.tile.height(); gpu_band_ms = t.render_ms; }
        }
        double maxd = 0, sumd = 0; long n = 0;
        if (insp.merged.image.size() == ref.rgba.size()) {
            for (size_t i = 0; i < ref.rgba.size(); ++i) {
                double d = std::fabs((double)insp.merged.image[i] - ref.rgba[i]);
                if (d > maxd) maxd = d;
                sumd += d; ++n;
            }
        }
        double meand = n ? sumd / n : -1;

        double t_dyn = med(v_dyn);
        auto dinsp = dyn.run(scene, R, R, both, RunMode::DynamicQueue);
        int cpu_tiles = 0, gpu_tiles = 0;
        for (const auto& t : dinsp.tiles)
            (t.path == PathKind::CpuIr ? cpu_tiles : gpu_tiles)++;

        double mpix = double(R) * R / 1e6;
        auto thr = [&](double ms) { return ms > 0 ? mpix / (ms / 1e3) : -1; };

        csv_row("frep4", "cpu_ir",         name.c_str(), "render", R, t_cpu_direct, thr(t_cpu_direct));
        csv_row("frep4", "gpu_glsl",       name.c_str(), "render", R, t_gpu_direct, thr(t_gpu_direct));
        csv_row("frep4", "hetero_ideal",   name.c_str(), "render", R, ideal,   thr(ideal));
        csv_row("frep4", "hetero_weighted",name.c_str(), "render", R, t_split, thr(t_split));
        csv_row("frep4", "hetero_dynamic", name.c_str(), "render", R, t_dyn,   thr(t_dyn));

        double best_single = t_cpu_direct < t_gpu_direct ? t_cpu_direct : t_gpu_direct;
        std::fprintf(stderr,
            "  %-16s cpu %7.1f  gpu %6.1f  ideal %6.1f  best-single %6.1f |"
            " split %6.1f (%4.0f%% of ideal, %+5.1f vs best,"
            " rows %d/%d, bands %5.1f/%5.1f ms)"
            "  dyn %6.1f (%4.0f%%, tiles %d/%d) | stitch vs cpu:"
            " mean %.5f max %.3f\n",
            name.c_str(), t_cpu_direct, t_gpu_direct, ideal, best_single,
            t_split, 100.0 * ideal / t_split, t_split - best_single,
            cpu_rows, gpu_rows, cpu_band_ms, gpu_band_ms,
            t_dyn, 100.0 * ideal / t_dyn, cpu_tiles, gpu_tiles,
            meand, maxd);
    }
    return 0;
}
