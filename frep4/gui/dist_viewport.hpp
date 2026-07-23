#pragma once

// gui/dist_viewport.hpp
//
// An IViewport that renders a frame across LAN workers and paints tiles as they
// arrive. It runs a dist::Master on a background thread; the master's per-tile
// progress hook marshals each finished tile onto the GUI thread, where it's
// blitted into the composited image — the same progressive-display feel as the
// local ExecutorViewport, but the pixels come from remote workers.
//
// This viewport is master-side only: it owns the listener + scheduler and shows
// the result. Starting a *worker* is a separate action (it has no local frame
// to show) handled by the LAN tab directly.

#include "gui/iviewport.hpp"
#include "core/frep/scene.hpp"
#include "core/exec/multipath.hpp"   // Tile, TileResult

#include <QImage>
#include <QWidget>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace frep::gui {

class DistCanvas;  // inner painted surface

// Master-side render config the LAN tab fills in.
struct DistRenderConfig {
    int         port      = 53900;
    int         n_workers = 1;
    int         width     = 400;
    int         height    = 300;
    int         tile      = 64;     // grid tile size (px)
    bool        push      = false;  // pull (default) vs push scheduler
};

class DistViewport : public IViewport {
    Q_OBJECT
    friend class DistCanvas;  // paint reads image_ under image_mu_
public:
    DistViewport(SceneGraph* scene, DistRenderConfig cfg, QWidget* parent = nullptr);
    ~DistViewport() override;

    QWidget* widget() override;
    void invalidate() override;     // (re)start the distributed render
    void set_tracer_config(const TracerConfig&) override {}
    void set_ssaa(int) override {}
    void set_camera_control_config(const CameraControlConfig&) override {}
    QImage capture_image() override;
    QString status_text() const override;

    // Called by the LAN tab to (re)configure before invalidate().
    void set_dist_config(const DistRenderConfig& c) { cfg_ = c; }
    // Public stop — interrupts and joins the master thread now (the LAN tab
    // calls this before tearing the viewport down so the join is synchronous).
    void stop() { stop_render(); }

private:
    void start_render();            // launch the master thread (one-shot)
    void stop_render();             // signal + join
    void on_tile_ready();           // GUI-thread: repaint after a tile lands

    SceneGraph*      scene_;
    DistRenderConfig cfg_;
    DistCanvas*      canvas_ = nullptr;

    std::thread          master_thread_;
    std::atomic<bool>    running_{false};
    std::atomic<bool>    cancel_{false};   // set by stop_render() → master returns
    std::atomic<int>     tiles_done_{0};
    std::atomic<int>     tiles_total_{0};
    mutable std::mutex   image_mu_;
    QImage               image_;          // composited frame (under image_mu_)
    std::string          status_;         // last status line (under image_mu_)
    std::vector<int>     per_worker_;      // tiles per worker (under image_mu_)
};

}  // namespace frep::gui
