// tilequeue_bench — heterogeneous *throughput*, measured the way heterogeneity
// actually pays off.
//
// A single frame split across devices and joined at a barrier measures LATENCY,
// and its wall time is bounded below by the slower band: that is the wrong
// question. What a heterogeneous machine buys is sustained THROUGHPUT — more
// megapixels per second from the executors present — and the natural mechanism
// is a work queue of tiles that every executor pulls from until it is empty.
// A faster executor simply takes more tiles; no weights, no calibration, and
// load balances itself as scene cost varies across the image.
//
// Method: render a continuous sequence of frames (an interactive session, not
// one frame). Each frame is a grid of tiles in a shared queue; each executor
// runs on its own thread and pulls the next tile until the frame is done.
// Throughput = total pixels / total wall time over the whole sequence, so the
// per-frame join amortizes exactly as it does in a real viewport.
//
// Reported per scene: each path alone (same harness, same tiling), then the
// combination. `sum` is the arithmetic sum of the single-path throughputs — the
// additive ideal — and `eff` is the fraction of it the combination realizes.
// Energy (RAPL + NVML) is integrated over each run, so Mpix/kWh is comparable
// across single and combined runs.
#include "core/exec/multipath.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/io/scene_io.hpp"
#include "core/power/energy_meter.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace frep;
using namespace frep::exec;
using namespace frep::power;

namespace {

struct Run {
    double ms = 0;          // wall time for the whole sequence
    double mpix_s = 0;      // aggregate throughput
    double joules = -1;     // total energy (-1 = unavailable)
    std::vector<long> tiles_per_exec;
};

// Render `frames` frames of `W x H`, tiles pulled from a shared queue by one
// thread per executor. Returns aggregate throughput over the whole sequence.
Run run_queue(std::vector<IExecutor*> execs, const SceneGraph& scene,
              int W, int H, int tile_px, int frames,
              EnergyCounter* cpu_e, EnergyCounter* gpu_e) {
    // Tile grid (same decomposition for every configuration, so single-path and
    // combined runs do identical work).
    std::vector<Tile> tiles;
    for (int y = 0; y < H; y += tile_px)
        for (int x = 0; x < W; x += tile_px)
            tiles.push_back(Tile{x, y, std::min(x + tile_px, W), std::min(y + tile_px, H)});

    Run out;
    out.tiles_per_exec.assign(execs.size(), 0);
    std::vector<std::atomic<long>> counts(execs.size());
    for (auto& c : counts) c.store(0);

    if (cpu_e && cpu_e->available()) cpu_e->begin();
    if (gpu_e && gpu_e->available()) gpu_e->begin();
    auto t0 = std::chrono::steady_clock::now();

    for (int f = 0; f < frames; ++f) {
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        for (size_t e = 0; e < execs.size(); ++e) {
            pool.emplace_back([&, e] {
                for (;;) {
                    size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= tiles.size()) break;
                    TileResult r = execs[e]->render(scene, W, H, tiles[i]);
                    if (r.ok) counts[e].fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& t : pool) t.join();
    }

    auto t1 = std::chrono::steady_clock::now();
    double j = 0; bool any_e = false;
    if (cpu_e && cpu_e->available()) { if (auto v = cpu_e->end()) { j += *v; any_e = true; } }
    if (gpu_e && gpu_e->available()) { if (auto v = gpu_e->end()) { j += *v; any_e = true; } }

    out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double pixels = (double)W * H * frames;
    out.mpix_s = out.ms > 0 ? pixels / (out.ms / 1e3) / 1e6 : 0;
    out.joules = any_e ? j : -1;
    for (size_t e = 0; e < execs.size(); ++e) out.tiles_per_exec[e] = counts[e].load();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <scene.json>... [--res N] [--tile N] [--frames N]\n"
            "  Heterogeneous throughput via a shared tile queue (CPU_IR + GPU_GLSL).\n"
            "  --rounds N interleaves the three modes N times and takes medians.\n",
            argv[0]);
        return 2;
    }
    int R = 800, tile_px = 128, frames = 10, rounds = 5;
    std::vector<std::string> scenes;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int d) { return i + 1 < argc ? std::atoi(argv[++i]) : d; };
        if      (a == "--res")    R = next(R);
        else if (a == "--tile")   tile_px = next(tile_px);
        else if (a == "--frames") frames = next(frames);
        else if (a == "--rounds") rounds = next(rounds);
        else scenes.push_back(a);
    }

    auto cpu_e = make_cpu_energy_counter();
    auto gpu_e = make_gpu_energy_counter();

    std::printf("tile-queue heterogeneous throughput  %dx%d, %d-px tiles, "
                "%d frames/run, %d interleaved rounds (medians)\n",
                R, R, tile_px, frames, rounds);
    std::printf("energy: cpu=%s gpu=%s\n\n",
                cpu_e && cpu_e->available() ? cpu_e->domain().c_str() : "n/a",
                gpu_e && gpu_e->available() ? gpu_e->domain().c_str() : "n/a");
    std::printf("%-14s %9s %9s %9s %9s %6s  %s\n",
                "scene", "cpu", "gpu", "sum", "cpu+gpu", "eff", "tiles cpu/gpu");
    std::printf("---------------+---------+---------+---------+---------+------+--------------\n");

    for (const auto& path : scenes) {
        std::string name = path;
        if (auto s = name.find_last_of('/'); s != std::string::npos) name = name.substr(s + 1);
        if (auto d = name.rfind(".json"); d != std::string::npos) name = name.substr(0, d);

        SceneGraph scene;
        try { scene = io::load_scene(path); }
        catch (const std::exception& e) { std::fprintf(stderr, "skip %s: %s\n", path.c_str(), e.what()); continue; }

        TracerConfig cfg;
        auto cpu = make_executor(PathKind::CpuIr,   cfg);
        auto gpu = make_executor(PathKind::GpuGlsl, cfg);
        if (!cpu || !cpu->available() || !gpu || !gpu->available()) {
            std::fprintf(stderr, "%s: need both CpuIr and GpuGlsl\n", name.c_str());
            continue;
        }

        // Warm both (JIT / SPIR-V / pipeline) so the measured runs are steady state.
        Tile full{0, 0, R, R};
        cpu->render(scene, R, R, full);
        gpu->render(scene, R, R, full);

        // Interleaved rounds: one sample of every mode per round, medians taken
        // per mode. Measuring each mode to completion in turn lets GPU clock
        // and thermal drift bias whichever mode ran in the hot (or cold) block
        // — which shows up as a combined throughput above the sum of the
        // separately measured parts. Interleaving spreads the drift evenly.
        std::vector<Run> vc, vg, vh;
        for (int r = 0; r < rounds; ++r) {
            vc.push_back(run_queue({cpu.get()},            scene, R, R, tile_px, frames, cpu_e.get(), nullptr));
            vg.push_back(run_queue({gpu.get()},            scene, R, R, tile_px, frames, nullptr, gpu_e.get()));
            vh.push_back(run_queue({cpu.get(), gpu.get()}, scene, R, R, tile_px, frames, cpu_e.get(), gpu_e.get()));
        }
        auto median_run = [](std::vector<Run> v) {
            std::sort(v.begin(), v.end(),
                      [](const Run& a, const Run& b) { return a.mpix_s < b.mpix_s; });
            return v[v.size() / 2];
        };
        Run rc = median_run(vc), rg = median_run(vg), rh = median_run(vh);

        double sum = rc.mpix_s + rg.mpix_s;
        double eff = sum > 0 ? 100.0 * rh.mpix_s / sum : 0;
        std::printf("%-14s %9.2f %9.2f %9.2f %9.2f %5.0f%%  %ld/%ld\n",
                    name.c_str(), rc.mpix_s, rg.mpix_s, sum, rh.mpix_s, eff,
                    rh.tiles_per_exec[0], rh.tiles_per_exec[1]);
        std::fflush(stdout);

        // CSV rows for the results table (throughput in Mpix/s; energy if present).
        auto emit = [&](const char* backend, const Run& r) {
            double pix = (double)R * R * frames;
            double mk = r.joules > 0 ? pixels_per_kwh(pix, r.joules) / 1e6 : -1;
            std::fprintf(stderr, "%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.4f\n",
                         "frep4", backend, name.c_str(), "tilequeue", R,
                         r.ms, r.mpix_s, r.joules, mk);
        };
        emit("cpu_ir", rc); emit("gpu_glsl", rg); emit("hetero_queue", rh);
    }
    return 0;
}
