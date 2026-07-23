// gui/executor_viewport.cpp
#include "gui/executor_viewport.hpp"

#include "core/exec/executor_factory.hpp"
#include "core/exec/remote_executor.hpp"

#include <QImage>
#include <QTimer>
#include <QPainter>
#include <QFontMetrics>
#include <QFont>
#include <QPen>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>
#include <chrono>

namespace frep::gui {

// ── Inner canvas widget: paints the composited image, drives camera orbit ───
class ExecutorCanvas : public QWidget {
public:
    ExecutorCanvas(ExecutorViewport* owner, SceneGraph* scene, QWidget* parent)
        : QWidget(parent), owner_(owner), scene_(scene) {
        setMinimumSize(160, 120);
        setMouseTracking(false);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.fillRect(rect(), QColor(18, 18, 20));
        QImage img;
        {
            std::lock_guard<std::mutex> lk(owner_->image_mu_);
            img = owner_->image_;  // implicit-shared copy, cheap
        }
        if (img.isNull()) {
            p.setPen(QColor(150, 150, 150));
            p.drawText(rect(), Qt::AlignCenter, "rendering…");
            return;
        }
        // Letterbox-fit the rendered image into the widget.
        QSize ts = img.size().scaled(size(), Qt::KeepAspectRatio);
        QRect dst(QPoint((width() - ts.width()) / 2,
                         (height() - ts.height()) / 2), ts);
        p.drawImage(dst, img);

        // Map an image-space rect to widget space (same fit transform).
        const double sx = (double)dst.width()  / std::max(1, img.width());
        const double sy = (double)dst.height() / std::max(1, img.height());
        auto to_widget = [&](const QRect& r) {
            return QRect(dst.left() + (int)(r.left() * sx),
                         dst.top()  + (int)(r.top()  * sy),
                         (int)(r.width()  * sx),
                         (int)(r.height() * sy));
        };

        auto overlays = owner_->path_overlays();
        // Only annotate when more than one path is active (single path needs
        // no per-region label; the status bar already shows its timing).
        if (overlays.size() > 1) {
            p.setRenderHint(QPainter::Antialiasing, true);
            for (const auto& o : overlays) {
                if (o.region.isNull()) continue;
                QRect wr = to_widget(o.region);
                // Faint border around the path's region.
                p.setPen(QPen(QColor(255, 255, 255, 60), 1));
                p.drawRect(wr.adjusted(0, 0, -1, -1));
                // Label chip: "path  NN.N Mpix/s".
                QString label = exec::path_kind_name(o.path);
                if (o.mpix_s >= 0)
                    label += QString("  %1 Mpix/s").arg(o.mpix_s, 0, 'f', 1);
                else if (!o.ok)
                    label += "  (n/a)";
                QFont f = p.font(); f.setPointSizeF(8.5); p.setFont(f);
                QRect tb = p.fontMetrics().boundingRect(label);
                QRect chip(wr.left() + 4, wr.top() + 4,
                           tb.width() + 10, tb.height() + 6);
                p.fillRect(chip, QColor(0, 0, 0, 150));
                p.setPen(QColor(235, 235, 235));
                p.drawText(chip, Qt::AlignCenter, label);
            }
        }

        // Aggregate throughput banner: the heterogeneous sum. Placed a row
        // BELOW the per-path label chips (which sit at each region's top-left at
        // y≈4) so it can't overlap a strip's own label — with several vertical
        // strips a per-path chip can land near the horizontal center where a
        // top-aligned banner would otherwise sit.
        double agg = owner_->aggregate_mpix_s();
        if (agg >= 0 && overlays.size() > 1) {
            QString s = QString("Σ %1 Mpix/s  (%2 paths)")
                            .arg(agg, 0, 'f', 1).arg(overlays.size());
            QFont f = p.font(); f.setPointSizeF(9.5); f.setBold(true);
            p.setFont(f);
            QRect tb = p.fontMetrics().boundingRect(s);
            // Estimate the per-path chip height to offset below it.
            int chip_h = QFontMetrics(QFont(p.font().family(), 9)).height() + 6;
            int by = 4 + chip_h + 6;     // one row down from the per-path chips
            QRect banner((width() - tb.width()) / 2 - 8, by,
                         tb.width() + 16, tb.height() + 6);
            p.fillRect(banner, QColor(20, 60, 30, 190));
            p.setPen(QColor(180, 255, 200));
            p.drawText(banner, Qt::AlignCenter, s);
        }
    }

    void resizeEvent(QResizeEvent*) override { owner_->invalidate(); }

    void mousePressEvent(QMouseEvent* e) override {
        last_pos_ = e->pos(); dragging_ = true;
    }
    void mouseReleaseEvent(QMouseEvent*) override { dragging_ = false; }
    void mouseMoveEvent(QMouseEvent* e) override {
        if (!dragging_ || !scene_) return;
        QPoint d = e->pos() - last_pos_;
        last_pos_ = e->pos();
        const auto& cc = owner_->cam_cfg_;
        // Orbit yaw/pitch around the target at the configured sensitivity.
        auto& cam = scene_->camera();
        float* pos = cam.position.data();
        const float* tgt = cam.target.data();
        float vx = pos[0] - tgt[0], vy = pos[1] - tgt[1], vz = pos[2] - tgt[2];
        float r = std::sqrt(vx*vx + vy*vy + vz*vz);
        float yaw = std::atan2(vx, vz) - d.x() * cc.mouse_sensitivity;
        float pitch = std::asin(std::clamp(vy / (r + 1e-6f), -0.999f, 0.999f))
                      + d.y() * cc.mouse_sensitivity;
        pitch = std::clamp(pitch, -cc.max_pitch, cc.max_pitch);
        pos[0] = tgt[0] + r * std::cos(pitch) * std::sin(yaw);
        pos[1] = tgt[1] + r * std::sin(pitch);
        pos[2] = tgt[2] + r * std::cos(pitch) * std::cos(yaw);
        owner_->invalidate();
    }
    void wheelEvent(QWheelEvent* e) override {
        if (!scene_) return;
        const auto& cc = owner_->cam_cfg_;
        auto& cam = scene_->camera();
        float* pos = cam.position.data();
        const float* tgt = cam.target.data();
        float vx = pos[0]-tgt[0], vy = pos[1]-tgt[1], vz = pos[2]-tgt[2];
        float r = std::sqrt(vx*vx + vy*vy + vz*vz);
        float notches = e->angleDelta().y() / 120.0f;
        float s = std::pow(1.0f / cc.zoom_step, notches);
        float nr = std::clamp(r * s, cc.min_distance, cc.max_distance);
        float k = (r > 1e-6f) ? nr / r : 1.0f;
        pos[0] = tgt[0] + vx*k; pos[1] = tgt[1] + vy*k; pos[2] = tgt[2] + vz*k;
        owner_->invalidate();
    }

private:
    ExecutorViewport* owner_;
    SceneGraph*       scene_;
    QPoint last_pos_;
    bool   dragging_ = false;
};

// ── ExecutorViewport ────────────────────────────────────────────────────────

ExecutorViewport::ExecutorViewport(SceneGraph* scene, QObject* parent)
    : IViewport(parent), scene_(scene) {
    canvas_ = new ExecutorCanvas(this, scene_, nullptr);
    start_render();
}

ExecutorViewport::~ExecutorViewport() {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
    // Worker is stopped → safe to release the executors (and their GPU state,
    // e.g. the RT pipeline cache) before the rest of the object tears down.
    executors_.clear();
    // Delete the canvas explicitly (the LAN/path teardown now deletes the
    // viewport synchronously). Detach from any Qt parent first so the parent
    // doesn't also try to delete it (double-free).
    if (canvas_) {
        canvas_->setParent(nullptr);
        delete canvas_;
        canvas_ = nullptr;
    }
}

QWidget* ExecutorViewport::widget() { return canvas_; }

void ExecutorViewport::set_tracer_config(const TracerConfig& cfg) {
    cfg_ = cfg;
    executors_.clear();        // cfg affects emitted shaders → rebuild executors
    executors_for_paths_.clear();
    invalidate();
}
void ExecutorViewport::set_ssaa(int n) { ssaa_ = std::max(1, n); invalidate(); }
void ExecutorViewport::set_camera_control_config(const CameraControlConfig& c) {
    cam_cfg_ = c;
}

QImage ExecutorViewport::capture_image() {
    std::lock_guard<std::mutex> lk(image_mu_);
    return image_;
}

void ExecutorViewport::set_active_paths(const std::vector<PathKind>& paths) {
    // Stop the render worker BEFORE mutating shared state — it may be inside
    // ex->render() reading paths_ (e.g. a RemoteExecutor blocked on a TCP
    // round-trip). Mutating paths_ under it is a use-after-free.
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
    rendering_ = false;
    paths_ = paths.empty() ? std::vector<PathKind>{ PathKind::CpuIr } : paths;
    ema_throughput_.assign(paths_.size(), 0.0);
    invalidate();
}
std::vector<PathKind> ExecutorViewport::active_paths() const { return paths_; }
void ExecutorViewport::set_layout(MultiLayout l) {
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
    rendering_ = false;
    layout_ = l;
    invalidate();
}

void ExecutorViewport::set_remote_config(int port, int n_workers) {
    // Stop the render worker before clearing executors_ — it may be inside
    // ex->render() on the RemoteExecutor we're about to destroy. Clearing it
    // (or its endpoint) under the worker is a use-after-free.
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
    rendering_ = false;
    remote_port_ = port; remote_workers_ = n_workers;
    executors_.clear();   // force rebuild so the new endpoint is used
    executors_for_paths_.clear();
}

std::vector<double> ExecutorViewport::last_timings_ms() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    return timings_ms_;
}

std::vector<ExecutorViewport::PathOverlay>
ExecutorViewport::path_overlays() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    std::vector<PathOverlay> out;
    for (std::size_t i = 0; i < paths_.size(); ++i) {
        PathOverlay o;
        o.path = paths_[i];
        o.region = i < path_regions_.size() ? path_regions_[i] : QRect();
        o.ok = i < path_ok_.size() && path_ok_[i];
        double ms = i < timings_ms_.size() ? timings_ms_[i] : -1;
        double px = i < path_pixels_.size() ? path_pixels_[i] : 0;
        // Mpix/s = (pixels / 1e6) / (ms / 1000) = pixels / (ms * 1000).
        o.mpix_s = (o.ok && ms > 0) ? px / (ms * 1000.0) : -1.0;
        out.push_back(o);
    }
    return out;
}

double ExecutorViewport::aggregate_mpix_s() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    double sum = 0; bool any = false;
    for (std::size_t i = 0; i < paths_.size(); ++i) {
        double ms = i < timings_ms_.size() ? timings_ms_[i] : -1;
        double px = i < path_pixels_.size() ? path_pixels_[i] : 0;
        bool ok = i < path_ok_.size() && path_ok_[i];
        if (ok && ms > 0) { sum += px / (ms * 1000.0); any = true; }
    }
    return any ? sum : -1.0;
}

void ExecutorViewport::invalidate() {
    generation_++;          // any in-flight worker result will be discarded
    start_render();
}

void ExecutorViewport::start_render() {
    if (!canvas_) return;
    // If a worker is already running, leave it; the generation bump from
    // invalidate() makes it kick a fresh render for the new state on return.
    if (rendering_.exchange(true)) return;
    if (worker_.joinable()) worker_.join();
    int W = std::max(1, canvas_->width());
    int H = std::max(1, canvas_->height());
    cancel_ = false;
    int my_gen = generation_.load();
    worker_ = std::thread([this, W, H, my_gen] {
        my_gen_for_worker_ = my_gen;
        pending_retry_ = false;
        render_worker(W, H);
        rendering_ = false;
        if (my_gen != generation_.load()) {
            // Invalidated mid-render → render again for the latest state.
            QMetaObject::invokeMethod(this, [this] { start_render(); },
                                      Qt::QueuedConnection);
        } else if (pending_retry_.load() && !cancel_.load()) {
            // A path (e.g. lan) wasn't ready. Re-render shortly so the strip
            // fills in as soon as its worker connects — no camera move needed.
            QMetaObject::invokeMethod(this, [this] {
                QTimer::singleShot(400, this, [this] {
                    if (!cancel_.load()) { generation_++; start_render(); }
                });
            }, Qt::QueuedConnection);
            on_region_done();
        } else {
            on_region_done();
        }
    });
}

namespace {
// Convert an executor's float-RGBA tile into the destination QImage region.
void blit_tile(QImage& dst, const exec::TileResult& r) {
    const auto& t = r.tile;
    int tw = t.width(), th = t.height();
    for (int y = 0; y < th; ++y) {
        int dy = t.y0 + y;
        if (dy < 0 || dy >= dst.height()) continue;
        auto* row = reinterpret_cast<QRgb*>(dst.scanLine(dy));
        for (int x = 0; x < tw; ++x) {
            int dx = t.x0 + x;
            if (dx < 0 || dx >= dst.width()) continue;
            const float* px = &r.rgba[(std::size_t)(y * tw + x) * 4];
            auto clamp8 = [](float v) {
                int i = (int)(v * 255.0f + 0.5f);
                return (unsigned char)std::clamp(i, 0, 255);
            };
            row[dx] = qRgb(clamp8(px[0]), clamp8(px[1]), clamp8(px[2]));
        }
    }
}
}  // namespace

void ExecutorViewport::ensure_executors() {
    // Rebuild only if the active path set differs from what executors_ holds.
    // (Called from the worker thread; start_render() joins any prior worker
    // before relaunching, so paths_ isn't being mutated concurrently here.)
    if (executors_for_paths_ == paths_ && executors_.size() == paths_.size())
        return;
    executors_.clear();
    executors_.resize(paths_.size());
    for (std::size_t i = 0; i < paths_.size(); ++i) {
        if (paths_[i] == PathKind::Lan) {
            // The "lan" path is a RemoteExecutor over a TCP master endpoint,
            // not a local executor — build it from the remote config.
            exec::RemoteExecutor::Config rc;
            rc.port = remote_port_; rc.n_workers = remote_workers_;
            executors_[i] = std::make_unique<exec::RemoteExecutor>(rc);
        } else {
            executors_[i] = exec::make_executor(paths_[i], cfg_);
        }
    }
    executors_for_paths_ = paths_;
}

void ExecutorViewport::render_worker(int W_out, int H_out) {
    // Supersample: render at ssaa× the widget size, then box-downscale to the
    // output. ssaa==1 is a no-op. Clamp so a large window at high SSAA doesn't
    // explode the tile count.
    int ss = std::clamp(ssaa_, 1, 4);
    int W = W_out * ss, H = H_out * ss;
    QImage frame(W, H, QImage::Format_RGB32);
    {
        // Reuse the previous frame as the base so strips that don't finish this
        // pass keep their last content (no flicker through black). Only when the
        // resolution, layout and path set match — otherwise strips would land in
        // the wrong place.
        std::lock_guard<std::mutex> lk(image_mu_);
        if (!last_frame_.isNull() && last_frame_.size() == frame.size()
            && last_frame_layout_ == layout_ && last_frame_paths_ == paths_)
            frame = last_frame_.copy();
        else
            frame.fill(QColor(18, 18, 20));
    }

    std::vector<double> timings(paths_.size(), -1.0);

    // Build the tiles each path is responsible for, per layout.
    struct Job { std::size_t path_idx; exec::Tile tile; };
    std::vector<Job> jobs;
    const std::size_t np = paths_.size();

    if (np == 1) {
        jobs.push_back({0, exec::Tile{0, 0, W, H}});
    } else if (layout_ == MultiLayout::Tiles) {
        const int TS = 64;
        std::size_t k = 0;
        for (int y = 0; y < H; y += TS)
            for (int x = 0; x < W; x += TS, ++k)
                jobs.push_back({k % np,
                    exec::Tile{x, y, std::min(x + TS, W), std::min(y + TS, H)}});
    } else {
        // Strips / WeightedStrips: vertical bands left→right.
        std::vector<double> w(np, 1.0);
        if (layout_ == MultiLayout::WeightedStrips) {
            double tot = 0;
            for (std::size_t i = 0; i < np; ++i) {
                w[i] = ema_throughput_.size() > i && ema_throughput_[i] > 0
                           ? ema_throughput_[i] : 1.0;
                tot += w[i];
            }
            for (auto& v : w) v /= (tot > 0 ? tot : 1.0);
        } else {
            for (auto& v : w) v = 1.0 / (double)np;
        }
        int x = 0;
        for (std::size_t i = 0; i < np; ++i) {
            int xw = (i + 1 == np) ? (W - x) : (int)(w[i] * W + 0.5);
            xw = std::max(0, std::min(xw, W - x));
            if (xw > 0) jobs.push_back({i, exec::Tile{x, 0, x + xw, H}});
            x += xw;
        }
    }

    // Persistent executors (rebuilt only when the path set changes) so stateful
    // backends — notably RtxExecutor's pipeline cache — survive across frames.
    // A camera move then reuses the warm cache instead of rebuilding the RT
    // pipeline every frame.
    ensure_executors();
    auto& execs = executors_;

    // Per-path accounting for the overlay: bounding region, total pixels, ok.
    std::vector<QRect>  regions(np);
    std::vector<double> px_count(np, 0.0);
    std::vector<bool>   ok_flags(np, false);
    if (path_error_.size() != np) path_error_.assign(np, std::string());

    // Render each path on its own thread. In the sequential version a fast path
    // (e.g. gpu_glsl) was blitted only after a slower path ahead of it in path
    // order (e.g. cpu_ir) had finished, so the quicker strip visibly lagged.
    // Here the strips render concurrently and whichever finishes first publishes
    // first. Each path owns disjoint image regions and disjoint slots in the
    // accounting vectors, so the threads don't synchronise on those; only the
    // shared frame + the image_ publish is serialised by image_mu_. Executors are
    // one-per-path (no sharing) and the scene is read-only during render.
    if (ema_throughput_.size() < np) ema_throughput_.resize(np, 0.0);
    std::vector<std::vector<exec::Tile>> path_tiles(np);
    for (const auto& j : jobs)
        if (j.path_idx < np) path_tiles[j.path_idx].push_back(j.tile);

    std::vector<std::thread> threads;
    threads.reserve(np);
    for (std::size_t p = 0; p < np; ++p) {
        if (path_tiles[p].empty() || p >= execs.size() || !execs[p]) continue;
        threads.emplace_back([&, p] {
            auto& ex = execs[p];
            // Skip a path that isn't ready yet (e.g. the lan path before a worker
            // has connected). Flag a retry so the frame re-renders once it comes
            // up, without needing a camera move.
            if (!ex->available()) { pending_retry_ = true; return; }
            for (const auto& tile : path_tiles[p]) {
                if (cancel_.load() || generation_.load() != my_gen_for_worker_)
                    return;
                auto t0 = std::chrono::steady_clock::now();
                exec::TileResult r = ex->render(*scene_, W, H, tile);
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (timings[p] < 0) timings[p] = 0;
                timings[p] += ms;
                if (!r.ok) {
                    // Surface the first failure per path instead of silently
                    // keeping the previous frame (which looks like "the toggle
                    // did nothing"): record it so the overlay/log can report a
                    // path that is failing to render, e.g. an unavailable GPU_IR
                    // backend, rather than showing stale pixels as if current.
                    std::lock_guard<std::mutex> lk(image_mu_);
                    if (path_error_[p].empty()) path_error_[p] = r.error;
                    continue;
                }
                { std::lock_guard<std::mutex> lk(image_mu_); path_error_[p].clear(); }
                {
                    std::lock_guard<std::mutex> lk(image_mu_);
                    blit_tile(frame, r);
                    image_ = frame;   // progressive: publish after each tile
                }
                // Accumulate this path's region + pixel count for the overlay.
                QRect tr(tile.x0, tile.y0, tile.width(), tile.height());
                regions[p] = regions[p].isNull() ? tr : regions[p].united(tr);
                px_count[p] += (double)tile.width() * tile.height();
                ok_flags[p] = true;
                // Update throughput EMA for the weighted layout.
                if (ms > 0) {
                    double px = (double)tile.width() * tile.height();
                    double tp = px / ms;  // pixels/ms
                    double& e = ema_throughput_[p];
                    e = (e <= 0) ? tp : 0.7 * e + 0.3 * tp;
                }
                // Ask the canvas to repaint with what we have so far.
                QMetaObject::invokeMethod(this, [this] {
                    if (canvas_) canvas_->update();
                }, Qt::QueuedConnection);
            }
        });
    }
    for (auto& t : threads) t.join();

    if (cancel_.load() || generation_.load() != my_gen_for_worker_)
        return;   // a newer generation is pending — discard this partial frame

    std::lock_guard<std::mutex> lk(image_mu_);
    // Downscale the supersampled frame to the output size (smooth = box-ish).
    if (ss > 1) {
        image_ = frame.scaled(W_out, H_out, Qt::IgnoreAspectRatio,
                              Qt::SmoothTransformation);
        // Bring per-path regions back to output coordinates for the overlay.
        for (auto& r : regions)
            if (!r.isNull())
                r = QRect(r.x() / ss, r.y() / ss, r.width() / ss, r.height() / ss);
    } else {
        image_ = frame;
    }
    // Save the supersampled frame as the base for the next pass (preserves
    // strips that don't re-render this generation).
    last_frame_        = frame;
    last_frame_layout_ = layout_;
    last_frame_paths_  = paths_;
    timings_ms_ = timings;
    path_regions_ = regions;
    path_pixels_  = px_count;
    path_ok_      = ok_flags;
}

void ExecutorViewport::on_region_done() {
    // Called from the worker thread. Marshal the repaint + signal onto the GUI
    // thread via a queued invocation (touching widgets off-thread is UB in Qt).
    QMetaObject::invokeMethod(this, [this] {
        if (canvas_) canvas_->update();
        double total = 0;
        {
            std::lock_guard<std::mutex> lk(image_mu_);
            for (double t : timings_ms_) if (t > 0) total = std::max(total, t);
        }
        Q_EMIT render_completed((int)total);
    }, Qt::QueuedConnection);
}

QString ExecutorViewport::metrics_text() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    QString s;
    for (std::size_t i = 0; i < paths_.size(); ++i) {
        double t = i < timings_ms_.size() ? timings_ms_[i] : -1;
        QString line;
        if (i < path_error_.size() && !path_error_[i].empty()) {
            line = QString("%1  ERROR").arg(exec::path_kind_name(paths_[i]));
        } else {
            double fps = (t > 0) ? 1000.0 / t : 0.0;
            line = QString("%1  %2 ms  (%3 fps)")
                       .arg(exec::path_kind_name(paths_[i]))
                       .arg(t >= 0 ? QString::number(t, 'f', 1) : "—")
                       .arg(fps > 0 ? QString::number(fps, 'f', 0) : "—");
        }
        s += line;
        if (i + 1 < paths_.size()) s += "\n";
    }
    return s;
}

QString ExecutorViewport::status_text() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    QString s;
    for (std::size_t i = 0; i < paths_.size(); ++i) {
        if (i) s += "  ";
        double t = i < timings_ms_.size() ? timings_ms_[i] : -1;
        // A path that errored shows the reason instead of a timing, so a
        // silently-failing backend (e.g. GPU_IR without CUDA) is visible rather
        // than looking like a working render that ignores settings.
        if (i < path_error_.size() && !path_error_[i].empty()) {
            s += QString("%1: ERROR (%2)")
                     .arg(exec::path_kind_name(paths_[i]))
                     .arg(QString::fromStdString(path_error_[i]).left(60));
        } else {
            s += QString("%1: %2")
                     .arg(exec::path_kind_name(paths_[i]))
                     .arg(t >= 0 ? QString("%1ms").arg(t, 0, 'f', 1) : "—");
        }
    }
    return s;
}

}  // namespace frep::gui
