#pragma once

// gui/executor_viewport.hpp
//
// ExecutorViewport — an IViewport that renders through the exec::IExecutor
// abstraction, so it can drive ANY of the four retargeting paths (CPU_IR,
// GPU_GLSL, GPU_IR, GPU_RTX) — including the RT path the old Viewport never
// exposed — and can render with SEVERAL paths at once for a multipath view.
//
// Single path: the chosen executor renders the whole frame.
// Multiple paths: the frame is partitioned among the active paths by the
// selected layout (side-by-side strips, weighted strips, or interleaved
// tiles), each path rendering its share — a live picture of the heterogeneous
// split. Rendering happens on a worker thread; the widget repaints
// progressively as regions complete.
//
// This is deliberately separate from the legacy Viewport (which stays the
// CPU/CUDA realtime-ish path) so it can't regress the existing modes.

#include "core/compiler/codegen.hpp"      // TracerConfig
#include "core/exec/multipath.hpp"        // PathKind
#include "core/frep/scene.hpp"
#include "gui/camera_control_config.hpp"
#include "gui/iviewport.hpp"

#include <QImage>
#include <QWidget>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace frep::gui {

// PathKind lives in frep::exec; alias it here for brevity in this file.
using frep::exec::PathKind;

// How the frame is divided among multiple active paths.
enum class MultiLayout {
    Strips,         // equal vertical strips, one per path, left→right
    WeightedStrips, // strips sized by each path's recent throughput
    Tiles,          // interleaved checkerboard of tiles round-robin'd to paths
};

// The inner painted surface. Kept as a nested QWidget so ExecutorViewport (a
// QObject, not a QWidget) can hand MainWindow a real widget via widget().
class ExecutorCanvas;

class ExecutorViewport : public IViewport {
    Q_OBJECT
public:
    ExecutorViewport(SceneGraph* scene, QObject* parent = nullptr);
    ~ExecutorViewport() override;

    // ── IViewport ───────────────────────────────────────────────────────────
    QWidget* widget() override;
    void invalidate() override;
    void set_tracer_config(const TracerConfig& cfg) override;
    void set_ssaa(int n) override;
    void set_camera_control_config(const CameraControlConfig& c) override;
    QImage capture_image() override;
    QString status_text() const override;
    QString metrics_text() const override;

    // ── multipath controls (driven by the MainWindow panel) ──────────────────
    // Set which paths are active. Order defines strip order. Empty → CPU_IR.
    void set_active_paths(const std::vector<PathKind>& paths);
    std::vector<PathKind> active_paths() const;
    void set_layout(MultiLayout layout);
    MultiLayout layout() const { return layout_; }
    // Endpoint for the "lan" path's RemoteExecutor (port + worker count). Set
    // before activating the lan path; takes effect on the next executor rebuild.
    void set_remote_config(int port, int n_workers);

    // Per-path last-frame timing (ms), parallel to active_paths(); -1 if the
    // path didn't run or failed. For the status bar / overlay.
    std::vector<double> last_timings_ms() const;

    // Per-path overlay info for the canvas to label each region. The region
    // rect is in full-frame image pixels (the canvas maps it to widget space).
    struct PathOverlay {
        PathKind path;
        QRect    region;       // image-space rect this path rendered
        double   mpix_s = -1;  // throughput this frame (-1 if unknown/failed)
        bool     ok = false;
    };
    std::vector<PathOverlay> path_overlays() const;
    // Aggregate throughput across active paths (sum of per-path Mpix/s), the
    // heterogeneous number; -1 until a frame with timing has completed.
    double aggregate_mpix_s() const;

private:
    friend class ExecutorCanvas;
    void start_render();                 // kick a worker render of the frame
    void render_worker(int W, int H);    // body run off the GUI thread
    void on_region_done();               // marshal back to GUI thread → repaint
    // (Re)build the persistent per-path executors when the path set changes, so
    // stateful backends (e.g. the RT pipeline cache) persist across frames.
    void ensure_executors();

    SceneGraph*    scene_ = nullptr;
    ExecutorCanvas* canvas_ = nullptr;
    TracerConfig   cfg_{};
    int            ssaa_ = 1;
    CameraControlConfig cam_cfg_{};   // orbit/zoom sensitivity + clamps

    std::vector<PathKind> paths_ = { PathKind::CpuIr };

    // Persistent executors, parallel to paths_, rebuilt by ensure_executors()
    // only when the path set changes. Holds stateful backends across frames.
    std::vector<std::unique_ptr<exec::IExecutor>> executors_;
    std::vector<PathKind> executors_for_paths_;   // path set executors_ was built for
    int remote_port_ = 53900;     // "lan" path RemoteExecutor endpoint
    int remote_workers_ = 1;
    MultiLayout    layout_ = MultiLayout::Strips;

    // Composited image, written by the worker under image_mu_, read by paint.
    mutable std::mutex image_mu_;
    QImage         image_;
    // Last completed supersampled frame, reused as the base for the next pass
    // so a strip that doesn't finish this generation (e.g. a slow CPU path that
    // gets pre-empted by rapid camera moves) keeps its previous content instead
    // of flickering through the dark fill. Guarded by size/layout/path-set.
    QImage                last_frame_;
    MultiLayout           last_frame_layout_ = MultiLayout::Strips;
    std::vector<PathKind> last_frame_paths_;
    std::vector<double> timings_ms_;     // per active path, last frame
    std::vector<double> ema_throughput_; // pixels/ms EMA for WeightedStrips
    // Per-path accounting for the overlay (under image_mu_):
    std::vector<QRect>  path_regions_;   // bounding region each path rendered
    std::vector<double> path_pixels_;    // pixels each path rendered this frame
    std::vector<bool>   path_ok_;        // did the path produce pixels
    std::vector<std::string> path_error_; // first render error per path (empty = none)

    std::thread    worker_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> rendering_{false};
    std::atomic<int>  generation_{0};    // bumps on each invalidate to drop stale
    std::atomic<bool> pending_retry_{false};  // a path was unavailable → retry soon
    std::atomic<int>  my_gen_for_worker_{0};  // generation the worker is serving
};

}  // namespace frep::gui
