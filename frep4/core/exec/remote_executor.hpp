#pragma once

// core/exec/remote_executor.hpp
//
// RemoteExecutor is an IExecutor that does no local rendering: it forwards each
// tile to a worker over a TCP connection (the "lan" path) and returns the
// pixels like any other executor. This unifies distributed rendering with the
// local paths — the compositing ExecutorViewport drives it exactly like cpu_ir
// / gpu_glsl / etc., so "lan" can be selected alone or composited alongside
// local paths, and re-renders on scene changes for free (the viewport already
// re-invokes render() on every change).
//
// Design (PoC, deliberately simple):
//   - One RemoteExecutor owns one PersistentMaster bound to one port, accepting
//     a fixed number of workers. The master endpoint is this executor's
//     identity — which makes a *multi-master* topology a natural future
//     extension (see MULTI_MASTER below): several RemoteExecutors, each its own
//     master+workers, all composited by the viewport.
//   - render() re-serializes the whole scene and sends it only when the scene
//     hash changes (so the per-frame cost is one Scene push, not one per tile).
//     No delta / incremental scene update — out of scope for the PoC.
//   - The viewport calls render() serially per executor (one worker thread, one
//     tile at a time), so the single socket needs no locking against itself.
//
// ─────────────────────────────────────────────────────────────────────────────
// MULTI_MASTER (future work — recorded so it isn't re-derived later)
//
// Because each RemoteExecutor *is* an IExecutor wrapping one master endpoint,
// multiple masters fall out for free: a selection like {cpu_ir, remote@:5900,
// remote@:5901} is local CPU plus two independent LAN clusters, each with its
// own pool of workers, all composited into one frame by the existing
// ExecutorViewport. The coordinating process (the GUI) becomes one-to-many:
//   GUI  ─┬─ master A (port 5900) ─┬─ worker A1
//         │                        └─ worker A2
//         ├─ master B (port 5901) ─── worker B1
//         └─ local cpu_ir
// No new orchestration machinery is needed — "master" stops being global and
// becomes per-executor, and the weighted-strips layout already balances by
// measured throughput, so faster clusters automatically get more of the frame.
//
// What is explicitly OUT OF SCOPE for the PoC (each its own substantial design):
//   - When/how to distribute the scene (on demand vs. ahead-of-time vs. hybrid),
//     and incremental scene updates instead of whole-scene resends.
//   - Dynamic job distribution that accounts for heterogeneous worker resources
//     and keeps them optimally utilized.
//   - Fault tolerance: a worker (or whole master) vanishing mid-frame, with
//     re-assignment of its outstanding tiles.
//   - Back-pressure, batching tiles per round-trip, and latency hiding.
// These turn the simple per-tile request/response here into a much more dynamic
// distribution system; they are real networked-rendering problems beyond a PoC.
// ─────────────────────────────────────────────────────────────────────────────

#include "core/exec/multipath.hpp"
#include "core/dist/master.hpp"
#include "core/io/scene_io.hpp"

#include <functional>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>

namespace frep::exec {

class RemoteExecutor : public IExecutor {
public:
    struct Config {
        int port      = 53900;
        int n_workers = 1;
    };

    explicit RemoteExecutor(Config cfg) : cfg_(cfg) {
        dist::MasterConfig mc;
        mc.port = cfg_.port;
        mc.n_workers = cfg_.n_workers;
        mc.verbose = false;
        mc.cancel = &cancel_;        // interruptible accept (see ~RemoteExecutor)
        master_ = std::make_unique<dist::PersistentMaster>(mc);
        // Open the master (bind + accept workers) on a background thread so the
        // render thread NEVER blocks waiting for a worker to connect. available()
        // reports false until the session is up; the viewport simply skips the
        // lan strip until then. accept() polls the cancel flag, so teardown is
        // immediate even while still waiting for workers.
        open_thread_ = std::thread([this] {
            std::string err = master_->open();
            std::lock_guard<std::mutex> lk(state_mu_);
            open_error_ = err;
            opened_ = err.empty();
        });
    }

    ~RemoteExecutor() override {
        cancel_ = true;                       // interrupt a blocked accept
        if (open_thread_.joinable()) open_thread_.join();
        if (master_) master_->close();
    }

    PathKind path() const override { return PathKind::Lan; }

    bool available() const override {
        // Ready once the background open() succeeded and a worker is connected.
        std::lock_guard<std::mutex> lk(state_mu_);
        return opened_ && master_ && master_->worker_count() > 0;
    }

    TileResult render(const SceneGraph& scene, int W, int H,
                      const Tile& tile) override {
        TileResult out; out.tile = tile;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            if (!opened_) {
                // Master not up yet (still waiting for workers). Return a
                // not-ok tile; available() gates this path so the viewport
                // normally skips it, but guard here too.
                out.ok = false;
                out.error = open_error_.empty() ? "lan: waiting for workers"
                                                : open_error_;
                return out;
            }
        }
        // Re-serialize + diff against the last scene we sent (one Scene push per
        // frame; no delta optimization — PoC scope).
        std::string json = io::serialize_scene(scene, "", /*embed_textures=*/true);
        bool changed = (json != last_scene_json_);
        auto r = master_->render_tile(json, tile, changed);
        if (!r) { out.ok = false; out.error = r.error(); return out; }
        if (changed) last_scene_json_ = std::move(json);
        return std::move(*r);
    }

private:
    Config                                  cfg_;
    std::unique_ptr<dist::PersistentMaster> master_;
    std::string                             last_scene_json_;
    std::thread                             open_thread_;
    std::atomic<bool>                       cancel_{false};
    mutable std::mutex                      state_mu_;
    bool                                    opened_ = false;     // guarded by state_mu_
    std::string                             open_error_;         // guarded by state_mu_
};

}  // namespace frep::exec
