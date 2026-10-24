#pragma once
// core/tracer/tile_scheduler.hpp
//
// Distributes the render across tiles between threads: a persistent pool of
// workers pulls tiles off an atomic counter, and the calling thread works
// alongside them.

#include "core/compiler/jit_engine.hpp"
#include "core/frep/scene.hpp"
#include "core/tracer/cpu_affinity.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace frep {

struct RenderParams {
    int width       = 800;
    int height      = 600;
    // Largest tile edge. The scheduler halves it when a frame would otherwise
    // have too few tiles for the threads (see render()).
    int tile_size   = 64;
    int num_threads = 0;

    // Super-sampling anti-aliasing factor. 1 = no AA; 2 = 4x rays per pixel
    // (2x2 grid), 3 = 9x rays, etc. Internally the scheduler renders the
    // scene at width*ssaa x height*ssaa, then box-filters down to the
    // requested resolution. Quadratic cost in ssaa.
    int ssaa        = 1;

    // Optional render region (in output-pixel coordinates): when width/
    // height of the region are > 0, only this sub-rectangle is rendered and
    // written; pixels outside are left untouched. {0,0,0,0} (the default)
    // means the whole frame. Lets a caller render just one tile of the
    // image — used by the multi-path executor so a CPU path can render
    // exactly its assigned region rather than the whole frame and crop.
    int region_x0 = 0, region_y0 = 0, region_x1 = 0, region_y1 = 0;

    bool has_region() const { return region_x1 > region_x0 && region_y1 > region_y0; }
};

class TileScheduler {
public:
    // Run `job` on `n_threads` of the process pool (the calling thread is one
    // of them) and return when all have finished. For a caller with a frame's
    // worth of per-pixel work outside render(), such as the CPU executor
    // quantising a float frame to bytes, which single-threaded took longer
    // than the render it followed.
    static void run_parallel(int n_threads, const std::function<void()>& job) {
        pool().run(std::max(1, n_threads), job);
    }

    // Renders `cam` into `out_rgba` using the JIT'd render_tile.
    //
    // `params` is an optional float buffer for Incremental compilation
    // mode — the buffer's contents are loaded by render_tile in place of
    // node-parameter constants. Pass nullptr in Constant mode (the
    // buffer is unused in that case; render_tile does not dereference it).
    static void render(RenderTileFn        fn,
                       float*              out_rgba,
                       const Camera&       cam,
                       const RenderParams& p_in,
                       float*              params = nullptr)
    {
        // SSAA: render at higher resolution, then box-filter down.
        const int ssaa = std::max(1, p_in.ssaa);
        const int W_render = p_in.width  * ssaa;
        const int H_render = p_in.height * ssaa;

        // Direct path: no AA → render straight into the caller's buffer.
        // Supersampled path: render to a scratch buffer at W_render x H_render,
        // then average each ssaa x ssaa block into one output pixel.
        std::vector<float> scratch;
        float* render_target = out_rgba;
        if (ssaa > 1) {
            scratch.assign(static_cast<size_t>(W_render) * H_render * 4, 0.0f);
            render_target = scratch.data();
        }

        RenderParams p = p_in;
        p.width  = W_render;
        p.height = H_render;
        int n_threads = p.num_threads > 0
            ? p.num_threads
            : static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;

        // ── Camera basis ────────────────────────────────────────────────────
        auto v3sub = [](auto a, auto b) {
            return std::array<float,3>{a[0]-b[0], a[1]-b[1], a[2]-b[2]};
        };
        auto v3norm = [](std::array<float,3> v) {
            float l = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (l < 1e-8f) return std::array<float,3>{0.0f, 0.0f, -1.0f};
            return std::array<float,3>{v[0]/l, v[1]/l, v[2]/l};
        };
        auto v3cross = [](auto a, auto b) {
            return std::array<float,3>{
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            };
        };

        auto fwd   = v3norm(v3sub(cam.target, cam.position));
        auto right = v3norm(v3cross(fwd, cam.up));
        auto up    = v3cross(right, fwd);
        // Projection-aware scale:
        //   perspective  → positive: half the height of the view frustum at unit distance
        //   orthographic → negative: -half the view-rectangle height in world units
        // The JIT-emitted render_tile distinguishes the two by the sign.
        float view_scale = (cam.projection == Camera::Projection::Orthographic)
            ? -0.5f * cam.ortho_size
            :  std::tan(cam.fov_deg * 3.14159265f / 360.0f);

        struct Tile { int x, y, w, h; };
        std::vector<Tile> tiles;
        // Region bounds in render-space (scaled by ssaa). Default = whole
        // frame. Tiles are clipped to this rectangle so only the requested
        // sub-region is computed and written.
        int rx0 = 0, ry0 = 0, rx1 = p.width, ry1 = p.height;
        if (p_in.has_region()) {
            rx0 = std::max(0, p_in.region_x0 * ssaa);
            ry0 = std::max(0, p_in.region_y0 * ssaa);
            rx1 = std::min(p.width,  p_in.region_x1 * ssaa);
            ry1 = std::min(p.height, p_in.region_y1 * ssaa);
        }
        // Tile granularity. A 512² frame in 64-px tiles is 64 tiles for 56
        // threads: the frame then takes as long as the slowest tile, and the
        // measured scaling collapsed past 16 threads (24–45% of the machine
        // at 56). Halve the tile until there are at least sixteen per thread,
        // so the tail at the end of the frame is one small tile, not a large
        // one. Four per thread was enough while every tile cost about the
        // same; the ray-box clip makes a sky tile free, so the frame's work
        // sits in the handful of tiles the object covers and the frame is
        // once more the single densest tile — and that tile runs on a core
        // the governor has let idle down, since the clip left the rest of the
        // machine with nothing to do (the twisted bar's densest 32-px tile
        // took 3.2 ms on such a core against 1.0 ms on a busy one, and the
        // whole clipped frame was slower than the unclipped one at 56
        // threads). At 16 px that tile is four tiles on four cores: 0.7 ms
        // for the frame instead of 3.6, and every unclipped scene also gains
        // a little. 8 px measured the same as 16, so the floor stays.
        int tile = std::max(1, p.tile_size);
        auto tile_count = [&](int ts) {
            return (long)((rx1 - rx0 + ts - 1) / ts) * ((ry1 - ry0 + ts - 1) / ts);
        };
        while (tile > 16 && tile_count(tile) < 16L * n_threads) tile /= 2;
        for (int ty = ry0; ty < ry1; ty += tile)
            for (int tx = rx0; tx < rx1; tx += tile)
                tiles.push_back({
                    tx, ty,
                    std::min(tile, rx1 - tx),
                    std::min(tile, ry1 - ty)
                });

        std::atomic<int> next{0};
        const int total = static_cast<int>(tiles.size());

        std::function<void()> worker = [&]() {
            while (true) {
                int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= total) break;
                const auto& t = tiles[i];
                fn(render_target,
                   t.x, t.y, t.w, t.h, p.width, p.height,
                   cam.position[0], cam.position[1], cam.position[2],
                   fwd[0],   fwd[1],   fwd[2],
                   right[0], right[1], right[2],
                   up[0],    up[1],    up[2],
                   view_scale,
                   params);
            }
        };
        pool().run(std::min(n_threads, total > 0 ? total : 1), worker);

        // ── Downsample if SSAA was active ────────────────────────────────────
        if (ssaa > 1) {
            const int W = p_in.width;
            const int H = p_in.height;
            const float inv_n = 1.0f / static_cast<float>(ssaa * ssaa);
            // Downsample only the region that was rendered (whole frame by
            // default). Region is in output coordinates.
            int dx0 = 0, dy0 = 0, dx1 = W, dy1 = H;
            if (p_in.has_region()) {
                dx0 = std::max(0, p_in.region_x0); dy0 = std::max(0, p_in.region_y0);
                dx1 = std::min(W, p_in.region_x1); dy1 = std::min(H, p_in.region_y1);
            }
            for (int y = dy0; y < dy1; ++y) {
                for (int x = dx0; x < dx1; ++x) {
                    float r = 0, g = 0, b = 0, a = 0;
                    for (int sy = 0; sy < ssaa; ++sy) {
                        const int src_y = y * ssaa + sy;
                        const float* src_row =
                            render_target + (src_y * W_render + x * ssaa) * 4;
                        for (int sx = 0; sx < ssaa; ++sx) {
                            r += src_row[sx*4 + 0];
                            g += src_row[sx*4 + 1];
                            b += src_row[sx*4 + 2];
                            a += src_row[sx*4 + 3];
                        }
                    }
                    float* dst = out_rgba + (y * W + x) * 4;
                    dst[0] = r * inv_n;
                    dst[1] = g * inv_n;
                    dst[2] = b * inv_n;
                    dst[3] = a * inv_n;
                }
            }
        }
    }

private:
    // One pool for the process, sized to the hardware. Each frame used to
    // spawn its own threads and join them — 56 thread creations per frame on
    // this machine, around a millisecond, which is the whole frame for a
    // simple scene at 512². The workers sleep on a condition variable between
    // frames; run() wakes the first n-1 of them, does its own share of the
    // tiles on the calling thread, and returns when every helper has finished.
    // Frames are serialised through call_mutex_, so two callers rendering at
    // once take turns rather than interleave jobs.
    //
    // Exactly n-1 workers are woken, each through its own condition variable.
    // The single variable this used to broadcast on woke all 55 workers for a
    // frame that wanted 13 of them, and the other 42 ran just long enough to
    // find they were not needed -- on whatever core the kernel chose, which
    // in a heterogeneous render was the core the GPU path's thread was
    // spinning on for its fence: with CPU and GPU sharing one tile queue the
    // combined run measured 25% below the GPU alone from that alone. (A
    // counting semaphore was tried first; libstdc++'s wakes every waiter too,
    // and 55 of them contending for 13 slots on one cache line made the CPU
    // path seven times slower on 128-px tiles.)
    class Pool {
    public:
        Pool() {
            int hw = static_cast<int>(std::thread::hardware_concurrency());
            n_workers_ = std::max(0, hw - 1);
            workers_.reserve(n_workers_);
            for (int i = 0; i < n_workers_; ++i) {
                workers_.emplace_back(std::make_unique<Worker>());
                Worker* w = workers_.back().get();
                w->th = std::thread([this, w] { worker_main(*w); });
            }
        }
        ~Pool() {
            stop_.store(true);
            for (auto& w : workers_) {
                { std::lock_guard<std::mutex> lk(w->m); }
                w->cv.notify_one();
            }
            for (auto& w : workers_) w->th.join();
        }

        void run(int n_threads, const std::function<void()>& job) {
            std::lock_guard<std::mutex> call_lock(call_mutex_);
            const int helpers = std::clamp(n_threads - 1, 0, n_workers_);
            if (helpers > 0) {
                job_ = &job;
                pending_.store(helpers, std::memory_order_relaxed);
                ++generation_;
                for (int i = 0; i < helpers; ++i) {
                    Worker& w = *workers_[i];
                    // The store under the worker's mutex is what makes the
                    // wake reliable; a worker still spinning sees it without
                    // the lock, and notify_one costs no system call then.
                    { std::lock_guard<std::mutex> lk(w.m); w.gen.store(generation_, std::memory_order_release); }
                    w.cv.notify_one();
                }
            }
            job();
            if (helpers > 0) {
                std::unique_lock<std::mutex> lk(m_);
                done_cv_.wait(lk, [&] { return pending_.load(std::memory_order_acquire) == 0; });
                job_ = nullptr;
            }
        }

    private:
        struct Worker {
            std::mutex                 m;
            std::condition_variable    cv;
            std::atomic<std::uint64_t> gen{0};  // generation this worker is to run
            std::thread                th;
        };

        void worker_main(Worker& w) {
            // Whoever created the pool may be pinned to the GPU's socket;
            // the CPU path wants every core it was launched with.
            reset_thread_affinity_to_launch();
            std::uint64_t seen = 0;
            while (true) {
                // Spin briefly before blocking: tiles rendered back to back
                // (a shared tile queue, a frame of sub-tiles) arrive within
                // microseconds of each other, and a futex wake costs tens of
                // microseconds on each of thirteen workers per 128-px tile --
                // a third of such a tile's render. Idle longer than the spin
                // window and the worker sleeps, so a paused editor costs
                // nothing.
                bool ready = false;
                const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
                for (int i = 0; !ready; ++i) {
                    ready = stop_.load(std::memory_order_relaxed) ||
                            w.gen.load(std::memory_order_acquire) != seen;
                    if (!ready && (i & 63) == 63 && std::chrono::steady_clock::now() >= spin_until) break;
                }
                if (!ready) {
                    std::unique_lock<std::mutex> lk(w.m);
                    w.cv.wait(lk, [&] {
                        return stop_.load() || w.gen.load(std::memory_order_acquire) != seen; });
                }
                if (stop_.load()) return;
                seen = w.gen.load(std::memory_order_acquire);
                (*job_)();
                if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(m_);
                    done_cv_.notify_one();
                }
            }
        }

        std::mutex                           call_mutex_;
        std::mutex                           m_;
        std::condition_variable              done_cv_;
        std::vector<std::unique_ptr<Worker>> workers_;
        int                                  n_workers_  = 0;
        std::uint64_t                        generation_ = 0;   // under call_mutex_
        std::atomic<int>                     pending_{0};
        std::atomic<bool>                    stop_{false};
        const std::function<void()>*         job_ = nullptr;
    };

    static Pool& pool() {
        static Pool p;
        return p;
    }
};

} // namespace frep
