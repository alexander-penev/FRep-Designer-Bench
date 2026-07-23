#pragma once
// core/dist/worker.hpp
//
// Distributed render worker. Connects to the master, receives the scene once,
// builds its configured executor (CPU_IR / GPU_GLSL / GPU_IR — a worker is a
// full frep executor and compiles locally), then loops: ask for a tile,
// render it, send the pixels back, until the master says NoMoreWork.
//
// A worker is deliberately a thin shell around an exec::IExecutor: the same
// executors used by the local multipath path render here too. Which executor
// a worker runs is its choice, so a heterogeneous cluster (one CPU worker,
// one GPU worker) falls out naturally — each renders the tiles it pulls with
// its own path, and because the paths are visually equivalent (~0.0008) the
// stitched frame is seamless regardless of which worker rendered which tile.

#include "core/dist/transport.hpp"
#include "core/dist/tcp_transport.hpp"
#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/io/scene_io.hpp"

#include <cstdio>
#include <functional>
#include <memory>
#include <string>

namespace frep::dist {

struct WorkerConfig {
    std::string host = "127.0.0.1";
    int         port = 53900;
    int         width = 400, height = 300;   // full frame dims (for ray setup)
    double      retry_secs = 0.0;            // retry connect for N s (LAN startup race)
    bool        verbose = true;
    // Optional per-tile hook, fired after each tile this worker renders (before
    // it's sent back). Debug-only — lets a GUI preview the worker's last result.
    // Called on the worker's own thread, so the callback must be thread-safe.
    // Empty by default; carries no cost when unset.
    std::function<void(const exec::Tile&, const exec::TileResult&)> on_tile;
};

// `make_executor` builds the executor this worker uses (lets the caller pick
// CPU_IR / GPU_GLSL / GPU_IR). Returns false-y on construction failure.
using ExecutorFactory = std::function<std::unique_ptr<exec::IExecutor>()>;

class Worker {
public:
    Worker(WorkerConfig cfg, ExecutorFactory make_exec)
        : cfg_(std::move(cfg)), make_exec_(std::move(make_exec)) {}

    // Run to completion. Returns the number of tiles rendered, or an error.
    std::expected<int, std::string> run() {
        auto conn = tcp_connect(cfg_.host, cfg_.port, cfg_.retry_secs);
        if (!conn) return std::unexpected("connect: " + conn.error());
        auto& t = **conn;
        if (cfg_.verbose)
            std::printf("  worker: connected to %s:%d\n", cfg_.host.c_str(), cfg_.port);

        // Handshake once.
        if (auto h = send_msg(t, MsgType::Hello, {}); !h)
            return std::unexpected("hello: " + h.error());

        int total_rendered = 0;
        // Frame loop. The master sends a Scene at the start of every frame; in
        // one-shot mode there's exactly one. After a frame's tiles drain, the
        // master sends NoMoreWork (you may exit) or EndFrame (stay connected,
        // a new Scene for the next frame is coming) — the persistent LAN mode.
        for (;;) {
            auto scene_msg = recv_msg(t);
            if (!scene_msg) break;   // master closed the connection → done
            if (scene_msg->first == MsgType::NoMoreWork) break;
            if (scene_msg->first != MsgType::Scene)
                return std::unexpected("expected Scene message");

            std::string scene_json(scene_msg->second.begin(),
                                   scene_msg->second.end());
            SceneGraph scene;
            try {
                scene = io::deserialize_scene(scene_json, nullptr);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("scene parse: ") + e.what());
            }

            // Rebuild the executor each frame (the scene is re-sent whole; no
            // delta optimization for now, per the interactive-mode design).
            auto executor = make_exec_();
            if (!executor) return std::unexpected("executor construction failed");
            if (cfg_.verbose)
                std::printf("  worker: frame scene loaded, executor=%s\n",
                            exec::path_kind_name(executor->path()));

            // Tile loop: request → render → result, until EndFrame/NoMoreWork.
            bool done_session = false;
            for (;;) {
                if (auto r = send_msg(t, MsgType::TileRequest, {}); !r)
                    return std::unexpected("request: " + r.error());
                auto msg = recv_msg(t);
                if (!msg) { done_session = true; break; }   // connection gone
                if (msg->first == MsgType::EndFrame) break;  // frame done, await next Scene
                if (msg->first == MsgType::NoMoreWork) { done_session = true; break; }
                if (msg->first != MsgType::TileAssign)
                    return std::unexpected("expected TileAssign");

                WireTile w = decode_tile(msg->second);
                exec::Tile tile{w.x0, w.y0, w.x1, w.y1};
                auto rr = executor->render(scene, cfg_.width, cfg_.height, tile);
                if (!rr.ok)
                    return std::unexpected("render: " + rr.error);

                // Debug preview hook (no-op unless a GUI set it).
                if (cfg_.on_tile) cfg_.on_tile(tile, rr);

                auto payload = encode_tile_result(w, rr.rgba);
                if (auto s = send_msg(t, MsgType::TileResult, payload); !s)
                    return std::unexpected("send result: " + s.error());
                ++total_rendered;
            }
            if (done_session) break;
        }
        if (cfg_.verbose)
            std::printf("  worker: done, rendered %d tile(s) total\n", total_rendered);
        return total_rendered;
    }

private:
    WorkerConfig    cfg_;
    ExecutorFactory make_exec_;
};

} // namespace frep::dist
