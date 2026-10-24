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
#include <cstdlib>
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
    long   frames = 0;      // frames actually rendered in the window
    std::vector<long> tiles_per_exec;
};

// Render at least `frames` frames of `W x H`, and keep rendering until at
// least `min_ms` have passed, tiles pulled from a shared queue by one thread
// per executor. Returns aggregate throughput over the whole sequence.
//
// The time floor is for the energy counters: NVML integrates power on a
// 10–100 ms cadence, and ten frames of the GPU path are now under 50 ms of
// GPU work, which it does not resolve above the idle floor. Rendering the
// same thing for half a second and dividing by the frames done does.
Run run_queue(std::vector<IExecutor*> execs, const SceneGraph& scene,
              int W, int H, int tile_px, int frames, double min_ms,
              QueuePolicy policy, EnergyCounter* cpu_e, EnergyCounter* gpu_e) {
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
    auto elapsed_ms = [&] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    long done = 0;
    // One queue for the sequence, reset per frame, so the per-executor cell
    // times it measures carry from frame to frame as they would in a viewport.
    TileQueue queue(tiles, execs.size(), policy);
    while (done < frames || elapsed_ms() < min_ms) {
        // The cells are the same 128-px grid for every policy; under Guided a
        // take merges consecutive cells into a strip or a block, so the count
        // below is of cells rendered, not of render() calls.
        queue.reset();
        std::vector<std::thread> pool;
        for (size_t e = 0; e < execs.size(); ++e) {
            pool.emplace_back([&, e] {
                double last_ms = 0;
                Tile t;
                while (queue.take(e, last_ms, t)) {
                    auto tt0 = std::chrono::steady_clock::now();
                    TileResult r = execs[e]->render(scene, W, H, t);
                    last_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tt0).count();
                    if (r.ok) {
                        const long cells = (long)((t.width() + tile_px - 1) / tile_px) *
                                           (long)((t.height() + tile_px - 1) / tile_px);
                        counts[e].fetch_add(cells, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& t : pool) t.join();
        ++done;
    }

    auto t1 = std::chrono::steady_clock::now();
    double j = 0; bool any_e = false;
    if (cpu_e && cpu_e->available()) { if (auto v = cpu_e->end()) { j += *v; any_e = true; } }
    if (gpu_e && gpu_e->available()) { if (auto v = gpu_e->end()) { j += *v; any_e = true; } }

    out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.frames = done;
    double pixels = (double)W * H * done;
    out.mpix_s = out.ms > 0 ? pixels / (out.ms / 1e3) / 1e6 : 0;
    out.joules = any_e ? j : -1;
    for (size_t e = 0; e < execs.size(); ++e) out.tiles_per_exec[e] = counts[e].load();
    return out;
}


// Keep `n` cores busy for as long as the object lives.
//
// A GPU-only configuration leaves the CPU idle, and an idle CPU drops to a low
// power state that lengthens the latency of every submit and fence wait: on
// this host the GPU-alone baseline measured 24 Mpix/s with the CPU idle and 52
// Mpix/s with cores kept busy, on identical GPU work. Comparing that depressed
// baseline against a combined run (whose CPU band keeps the package awake) is
// what made composition look super-additive. The single-device runs therefore
// carry a synthetic load so every configuration is measured with the machine in
// the same state; the spinner touches no measured device, and CPU energy is not
// attributed to a GPU-only run.
class BusyLoad {
public:
    explicit BusyLoad(int n) {
        if (n <= 0) return;
        stop_.store(false);
        for (int i = 0; i < n; ++i)
            threads_.emplace_back([this] {
                volatile double x = 1.0;
                while (!stop_.load(std::memory_order_relaxed)) x = x * 1.000001 + 1.0;
                (void)x;
            });
    }
    ~BusyLoad() {
        stop_.store(true, std::memory_order_relaxed);
        for (auto& t : threads_) if (t.joinable()) t.join();
    }
private:
    std::atomic<bool> stop_{true};
    std::vector<std::thread> threads_;
};

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
    int R = 800, tile_px = 128, frames = 10, rounds = 5, warmup_s = 4;
    // Each run renders at least `frames` frames and at least min_ms (see
    // run_queue); the queue policy is Auto (the tail rule) unless --policy says otherwise.
    double min_ms = 500.0;
    QueuePolicy policy = QueuePolicy::Auto;
    std::vector<std::string> scenes;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int d) { return i + 1 < argc ? std::atoi(argv[++i]) : d; };
        if      (a == "--res")    R = next(R);
        else if (a == "--tile")   tile_px = next(tile_px);
        else if (a == "--frames") frames = next(frames);
        else if (a == "--rounds") rounds = next(rounds);
        else if (a == "--warmup") warmup_s = next(warmup_s);
        else if (a == "--min-ms") min_ms = next(500);
        else if (a == "--policy" && i + 1 < argc) {
            std::string p = argv[++i];
            policy = p == "plain"  ? QueuePolicy::Plain
                   : p == "tail"   ? QueuePolicy::FastestTail
                   : p == "guided" ? QueuePolicy::Guided : QueuePolicy::Auto;
        }
        else scenes.push_back(a);
    }

    auto cpu_e = make_cpu_energy_counter();
    auto gpu_e = make_gpu_energy_counter();

    std::printf("tile-queue heterogeneous throughput  %dx%d, %d-px cells, "
                "%d frames/run, %d interleaved rounds (medians), %ds warm-up; "
                "Auto = %s\n",
                R, R, tile_px, frames, rounds, warmup_s,
                TileQueue::resolve(policy) == QueuePolicy::Guided ? "guided"
                : TileQueue::resolve(policy) == QueuePolicy::Plain ? "plain" : "tail");
    std::printf("energy: cpu=%s gpu=%s\n\n",
                cpu_e && cpu_e->available() ? cpu_e->domain().c_str() : "n/a",
                gpu_e && gpu_e->available() ? gpu_e->domain().c_str() : "n/a");
    std::printf("%-14s %9s %9s %9s | %9s %5s %-9s | %9s %5s %-9s | %9s %5s %-9s\n",
                "scene", "cpu", "gpu", "sum", "tail", "eff", "cells c/g",
                "plain", "eff", "cells c/g", "guided", "eff", "cells c/g");
    std::printf("---------------+---------+---------+-----------+-----------------------------"
                "+-----------------------------+----------------------------\n");

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

        // Then warm the *hardware*: an idle NVIDIA GPU sits at a low clock
        // (615 MHz against a 2100 MHz maximum on this device) and boosts under
        // sustained load, so the first measured configuration would run on a
        // colder device than the last. That drift is what makes an isolated
        // baseline differ by tens of percent between rounds -- and what can make
        // a combined run look faster than the sum of its parts. Spin both
        // devices for a few seconds first and discard the work.
        {
            auto t_warm = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t_warm).count() < warmup_s) {
                run_queue({cpu.get(), gpu.get()}, scene, R, R, tile_px, 2, 0.0, policy,
                          nullptr, nullptr);
            }
        }

        // Interleaved rounds: one sample of every mode per round, medians taken
        // per mode. Measuring each mode to completion in turn lets GPU clock
        // and thermal drift bias whichever mode ran in the hot (or cold) block
        // — which shows up as a combined throughput above the sum of the
        // separately measured parts. Interleaving spreads the drift evenly.
        // How many cores the CPU executor uses, so the GPU-only run can be given
        // an equivalent synthetic load (see BusyLoad).
        int cpu_threads = 0;
        if (const char* t = std::getenv("FREP4_CPU_THREADS")) cpu_threads = std::atoi(t);
        if (cpu_threads <= 0) cpu_threads = (int)std::thread::hardware_concurrency();

        // Five modes per round: each path alone, then the combined queue under
        // each of the three policies, so the table shows what the tail rule
        // and the guided chunking are each worth. A path alone runs under the
        // guided policy whatever --policy says: with one executor the queue
        // hands it the frame in a few large takes, which is the best that
        // path can do in this harness, and `sum` should be the sum of bests
        // rather than of two paths handicapped by 128-px tiles. (--policy is
        // kept for the Auto resolution the bench prints; every policy is run.)
        const QueuePolicy chosen = QueuePolicy::Guided;
        const QueuePolicy pols[3] = { QueuePolicy::FastestTail, QueuePolicy::Plain,
                                      QueuePolicy::Guided };
        std::vector<Run> vc, vg, vh[3];
        for (int r = 0; r < rounds; ++r) {
            vc.push_back(run_queue({cpu.get()},            scene, R, R, tile_px, frames, min_ms, chosen, cpu_e.get(), nullptr));
            {
                BusyLoad load(cpu_threads);   // match the combined run's CPU state
                vg.push_back(run_queue({gpu.get()},        scene, R, R, tile_px, frames, min_ms, chosen, nullptr, gpu_e.get()));
            }
            for (int p = 0; p < 3; ++p)
                vh[p].push_back(run_queue({cpu.get(), gpu.get()}, scene, R, R, tile_px, frames, min_ms, pols[p], cpu_e.get(), gpu_e.get()));
        }
        auto median_run = [](std::vector<Run> v) {
            std::sort(v.begin(), v.end(),
                      [](const Run& a, const Run& b) { return a.mpix_s < b.mpix_s; });
            return v[v.size() / 2];
        };
        Run rc = median_run(vc), rg = median_run(vg);
        Run rh[3] = { median_run(vh[0]), median_run(vh[1]), median_run(vh[2]) };

        double sum = rc.mpix_s + rg.mpix_s;
        auto eff = [&](const Run& r) { return sum > 0 ? 100.0 * r.mpix_s / sum : 0.0; };
        std::printf("%-14s %9.2f %9.2f %9.2f", name.c_str(), rc.mpix_s, rg.mpix_s, sum);
        for (int p = 0; p < 3; ++p)
            std::printf(" | %9.2f %4.0f%% %4ld/%-4ld", rh[p].mpix_s, eff(rh[p]),
                        rh[p].tiles_per_exec[0], rh[p].tiles_per_exec[1]);
        std::printf("\n");
        std::fflush(stdout);

        // CSV rows for the results table (throughput in Mpix/s; energy if present).
        auto emit = [&](const char* backend, const Run& r) {
            double pix = (double)R * R * r.frames;
            double mk = r.joules > 0 ? pixels_per_kwh(pix, r.joules) / 1e6 : -1;
            std::fprintf(stderr, "%s,%s,%s,%s,%d,%.3f,%.3f,%.3f,%.4f\n",
                         "frep4", backend, name.c_str(), "tilequeue", R,
                         r.ms, r.mpix_s, r.joules, mk);
        };
        // Each combined row is named for the rule it ran, so a table can tell
        // them apart without knowing the default.
        emit("cpu_ir", rc); emit("gpu_glsl", rg);
        emit("hetero_tail", rh[0]); emit("hetero_plain", rh[1]); emit("hetero_guided", rh[2]);
    }
    return 0;
}
