#pragma once
// core/dist/master.hpp
//
// Distributed render master. Binds a listener, accepts N worker connections,
// sends each the scene once, then serves tiles through an IScheduler until
// the frame is complete. Each worker connection is handled on its own thread;
// the scheduler (thread-safe) hands out tile indices, so PullScheduler gives
// work-stealing across machines for free.
//
// The master is transport-agnostic above the socket: it speaks the framed
// protocol from transport.hpp over whatever ITransport each connection is.
// Results are collected into a tile vector and stitched (+ post-processed by
// the caller) exactly like the local multipath path — the network is just
// where the RenderStage happened to run.

#include "core/dist/scheduler.hpp"
#include "core/dist/transport.hpp"
#include "core/dist/tcp_transport.hpp"
#include "core/exec/multipath.hpp"

#include <expected>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <functional>
#include <vector>

namespace frep::dist {

struct MasterConfig {
    int         port      = 53900;
    int         n_workers = 1;       // how many connections to wait for
    int         width     = 400;
    int         height    = 300;
    std::string scene_json;          // serialized scene, sent to each worker
    bool        verbose   = true;
    // Optional progress hook: called once per tile as it arrives, from the
    // serving thread (so the callback must be thread-safe). Lets a GUI display
    // tiles progressively instead of waiting for the whole frame. The tile
    // index is into the original `tiles` decomposition; worker_id is which
    // connection delivered it. Empty by default (CLI path is unaffected).
    std::function<void(int tile_index, int worker_id,
                       const exec::TileResult&)> on_tile;
    // Optional cancel flag. When it becomes true, run() stops waiting for
    // workers (interruptible accept) and stops serving tiles, returning early.
    // Lets a GUI tear down a render that's still waiting for workers.
    const std::atomic<bool>* cancel = nullptr;
};

struct MasterResult {
    std::vector<exec::TileResult> tiles;   // one per rendered tile (in tile order)
    int   width = 0, height = 0;
    int   tiles_done = 0;
    bool  ok = false;
    std::string error;
    // Per-worker tile counts, to show the distribution / balance.
    std::vector<int> per_worker_tiles;
};

// Run the master to completion: accept n_workers, render `tiles` via the
// scheduler, return collected results. `tiles` is the decomposition of the
// W×H frame (e.g. from GridDecompose); the scheduler indexes into it.
class Master {
public:
    Master(MasterConfig cfg, std::vector<exec::Tile> tiles, IScheduler& sched)
        : cfg_(std::move(cfg)), tiles_(std::move(tiles)), sched_(sched) {}

    MasterResult run() {
        MasterResult res;
        res.width = cfg_.width; res.height = cfg_.height;
        res.tiles.resize(tiles_.size());
        res.per_worker_tiles.assign(cfg_.n_workers, 0);

        auto listener = TcpListener::bind(cfg_.port);
        if (!listener) { res.error = "bind: " + listener.error(); return res; }
        if (cfg_.verbose)
            std::printf("  master: listening on port %d for %d worker(s)\n",
                        cfg_.port, cfg_.n_workers);

        // Accept all workers first (simple; could overlap with serving).
        std::vector<std::unique_ptr<TcpBinaryTransport>> conns;
        for (int i = 0; i < cfg_.n_workers; ++i) {
            auto c = listener->accept(cfg_.cancel);
            if (!c) {
                // Cancelled while waiting for workers, or a real accept error.
                res.error = (cfg_.cancel && cfg_.cancel->load())
                                ? "cancelled" : ("accept: " + c.error());
                return res;
            }
            if (cfg_.verbose)
                std::printf("  master: worker %d connected\n", i);
            conns.push_back(std::move(*c));
        }

        // One serving thread per worker.
        std::atomic<int> done{0};
        std::vector<std::thread> threads;
        std::string thread_err;
        std::mutex err_mu;

        for (int i = 0; i < cfg_.n_workers; ++i) {
            threads.emplace_back([&, i] {
                auto& t = *conns[i];
                auto fail = [&](const std::string& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (thread_err.empty()) thread_err = e;
                };
                // Expect HELLO, then send the scene once.
                auto hello = recv_msg(t);
                if (!hello) { fail("worker " + std::to_string(i) + ": " + hello.error()); return; }
                if (hello->first != MsgType::Hello) { fail("expected Hello"); return; }
                std::vector<std::uint8_t> scene_bytes(cfg_.scene_json.begin(),
                                                      cfg_.scene_json.end());
                if (auto s = send_msg(t, MsgType::Scene, scene_bytes); !s) {
                    fail("send scene: " + s.error()); return;
                }

                // Serve tiles until this worker is told there's no more work.
                for (;;) {
                    if (cfg_.cancel && cfg_.cancel->load()) {
                        send_msg(t, MsgType::NoMoreWork, {});
                        break;
                    }
                    auto req = recv_msg(t);
                    if (!req) { fail("worker " + std::to_string(i) + ": " + req.error()); return; }
                    if (req->first != MsgType::TileRequest) { fail("expected TileRequest"); return; }

                    long idx = sched_.next_for(i);
                    if (idx < 0) {
                        send_msg(t, MsgType::NoMoreWork, {});
                        break;
                    }
                    const exec::Tile& tl = tiles_[static_cast<std::size_t>(idx)];
                    WireTile w{tl.x0, tl.y0, tl.x1, tl.y1, -1};
                    if (auto a = send_msg(t, MsgType::TileAssign, encode_tile(w)); !a) {
                        fail("send assign: " + a.error()); return;
                    }
                    // Receive the rendered result.
                    auto rr = recv_msg(t);
                    if (!rr) { fail("worker " + std::to_string(i) + ": " + rr.error()); return; }
                    if (rr->first != MsgType::TileResult) { fail("expected TileResult"); return; }
                    auto dec = decode_tile_result(rr->second);

                    exec::TileResult tr;
                    tr.tile = tl;
                    tr.rgba = std::move(dec.rgba);
                    tr.ok = true;
                    // Fire the progress hook before moving the result into the
                    // collection, so a GUI can paint this tile immediately.
                    if (cfg_.on_tile) cfg_.on_tile((int)idx, i, tr);
                    res.tiles[static_cast<std::size_t>(idx)] = std::move(tr);
                    res.per_worker_tiles[i]++;
                    done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) th.join();

        res.tiles_done = done.load();
        if (!thread_err.empty()) { res.error = thread_err; return res; }
        res.ok = (res.tiles_done == static_cast<int>(tiles_.size()));
        if (!res.ok && res.error.empty())
            res.error = "incomplete: " + std::to_string(res.tiles_done) +
                        "/" + std::to_string(tiles_.size()) + " tiles";
        return res;
    }

private:
    MasterConfig          cfg_;
    std::vector<exec::Tile>     tiles_;
    IScheduler&           sched_;
};

// PersistentMaster keeps workers connected across many frames for an
// interactive LAN render. Bind + accept happen once; then render_frame() is
// called repeatedly (e.g. whenever the scene changes). Each call sends the full
// scene to every worker, distributes the tiles via a fresh scheduler, collects
// results (firing on_tile progressively), and tells workers EndFrame so they
// stay connected for the next one. The whole frame re-sends the scene — no
// delta optimization, by design for now.
//
// Threading: open() and render_frame() are meant to be called from one driver
// thread (the GUI's render thread). cancel (a std::atomic<bool>*) interrupts a
// blocked accept or an in-flight frame so teardown never hangs.
class PersistentMaster {
public:
    struct Frame {
        bool ok = false;
        int  tiles_done = 0;
        std::string error;
        std::vector<int> per_worker_tiles;
    };

    explicit PersistentMaster(MasterConfig cfg) : cfg_(std::move(cfg)) {}

    // Bind the port and accept n_workers. Returns an error string on failure
    // (empty on success). After this, workers are connected and have completed
    // their Hello; each is waiting for the first Scene.
    std::string open() {
        auto listener = TcpListener::bind(cfg_.port);
        if (!listener) return "bind: " + listener.error();
        listener_ = std::make_unique<TcpListener>(std::move(*listener));
        for (int i = 0; i < cfg_.n_workers; ++i) {
            auto c = listener_->accept(cfg_.cancel);
            if (!c)
                return (cfg_.cancel && cfg_.cancel->load())
                           ? "cancelled" : ("accept: " + c.error());
            // Expect Hello up front (once per connection).
            auto hello = recv_msg(**c);
            if (!hello) return "worker " + std::to_string(i) + ": " + hello.error();
            if (hello->first != MsgType::Hello) return "expected Hello";
            conns_.push_back(std::move(*c));
        }
        return {};
    }

    // Render one frame of the given scene at W×H decomposed into `tiles`, using
    // `sched` (which must be sized for tiles.size() and the worker count).
    // on_tile(tile_index, worker_id, result) fires per tile as it arrives.
    Frame render_frame(const std::string& scene_json,
                       const std::vector<exec::Tile>& tiles,
                       IScheduler& sched,
                       const std::function<void(int,int,const exec::TileResult&)>& on_tile) {
        Frame f;
        f.per_worker_tiles.assign(cfg_.n_workers, 0);
        if (conns_.empty()) { f.error = "no workers"; return f; }

        std::atomic<int> done{0};
        std::vector<std::thread> threads;
        std::string thread_err;
        std::mutex err_mu, tile_mu;

        for (int i = 0; i < cfg_.n_workers; ++i) {
            threads.emplace_back([&, i] {
                auto& t = *conns_[i];
                auto fail = [&](const std::string& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (thread_err.empty()) thread_err = e;
                };
                // Send this frame's scene.
                std::vector<std::uint8_t> scene_bytes(scene_json.begin(),
                                                      scene_json.end());
                if (auto s = send_msg(t, MsgType::Scene, scene_bytes); !s) {
                    fail("send scene: " + s.error()); return;
                }
                // Serve tiles until the queue drains, then EndFrame (keep alive).
                for (;;) {
                    if (cfg_.cancel && cfg_.cancel->load()) {
                        send_msg(t, MsgType::NoMoreWork, {}); return;
                    }
                    auto req = recv_msg(t);
                    if (!req) { fail("worker " + std::to_string(i) + ": " + req.error()); return; }
                    if (req->first != MsgType::TileRequest) { fail("expected TileRequest"); return; }

                    long idx = sched.next_for(i);
                    if (idx < 0) { send_msg(t, MsgType::EndFrame, {}); break; }

                    const exec::Tile& tl = tiles[static_cast<std::size_t>(idx)];
                    WireTile w{tl.x0, tl.y0, tl.x1, tl.y1, -1};
                    if (auto a = send_msg(t, MsgType::TileAssign, encode_tile(w)); !a) {
                        fail("send assign: " + a.error()); return;
                    }
                    auto rr = recv_msg(t);
                    if (!rr) { fail("worker " + std::to_string(i) + ": " + rr.error()); return; }
                    if (rr->first != MsgType::TileResult) { fail("expected TileResult"); return; }
                    auto dec = decode_tile_result(rr->second);

                    exec::TileResult tres;
                    tres.tile = tl; tres.rgba = std::move(dec.rgba); tres.ok = true;
                    if (on_tile) {
                        std::lock_guard<std::mutex> lk(tile_mu);
                        on_tile((int)idx, i, tres);
                    }
                    f.per_worker_tiles[i]++;
                    done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) th.join();
        f.tiles_done = done.load();
        if (!thread_err.empty()) { f.error = thread_err; return f; }
        f.ok = (f.tiles_done == static_cast<int>(tiles.size()));
        return f;
    }

    // Per-tile render against a single worker (worker 0), for use by an
    // IExecutor wrapper. `scene_changed` makes the master push the scene before
    // this tile (otherwise the worker reuses the scene from a previous tile of
    // the same frame). Returns the rendered tile, or an error.
    //
    // Protocol note: the worker loop is `recv Scene → {send TileRequest → recv
    // TileAssign|EndFrame}*`. To drive it one tile at a time we consume the
    // worker's TileRequest, answer with one TileAssign, and read the TileResult.
    // A scene change ends the current frame (EndFrame) so the worker loops back
    // to await a fresh Scene.
    std::expected<exec::TileResult, std::string>
    render_tile(const std::string& scene_json, const exec::Tile& tile,
                bool scene_changed) {
        if (conns_.empty()) return std::unexpected("no workers");
        auto& t = *conns_[0];

        if (scene_changed) {
            // If a frame was in progress, close it so the worker awaits a Scene.
            if (frame_open_) {
                // Drain the worker's pending TileRequest, answer EndFrame.
                auto req = recv_msg(t);
                if (!req) return std::unexpected("endframe req: " + req.error());
                send_msg(t, MsgType::EndFrame, {});
                frame_open_ = false;
            }
            std::vector<std::uint8_t> sb(scene_json.begin(), scene_json.end());
            if (auto s = send_msg(t, MsgType::Scene, sb); !s)
                return std::unexpected("send scene: " + s.error());
            frame_open_ = true;
        }
        // Consume the worker's TileRequest, assign this tile, read the result.
        auto req = recv_msg(t);
        if (!req) return std::unexpected("tile req: " + req.error());
        if (req->first != MsgType::TileRequest)
            return std::unexpected("expected TileRequest");
        WireTile w{tile.x0, tile.y0, tile.x1, tile.y1, -1};
        if (auto a = send_msg(t, MsgType::TileAssign, encode_tile(w)); !a)
            return std::unexpected("send assign: " + a.error());
        auto rr = recv_msg(t);
        if (!rr) return std::unexpected("recv result: " + rr.error());
        if (rr->first != MsgType::TileResult)
            return std::unexpected("expected TileResult");
        auto dec = decode_tile_result(rr->second);
        exec::TileResult res;
        res.tile = tile; res.rgba = std::move(dec.rgba); res.ok = true;
        return res;
    }

    int worker_count() const { return (int)conns_.size(); }

    // Tell all workers the session is over (NoMoreWork) and drop connections.
    void close() {
        for (auto& c : conns_) {
            if (!c) continue;
            // If mid-frame, close it cleanly first.
            if (frame_open_) { recv_msg(*c); send_msg(*c, MsgType::EndFrame, {}); }
            send_msg(*c, MsgType::NoMoreWork, {});
        }
        frame_open_ = false;
        conns_.clear();
        listener_.reset();
    }

    ~PersistentMaster() { close(); }

private:
    MasterConfig cfg_;
    std::unique_ptr<TcpListener> listener_;
    std::vector<std::unique_ptr<TcpBinaryTransport>> conns_;
    bool frame_open_ = false;   // a Scene has been sent, frame not yet ended
};

} // namespace frep::dist
