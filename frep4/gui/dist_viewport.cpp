// gui/dist_viewport.cpp
#include "gui/dist_viewport.hpp"

#include "core/dist/master.hpp"
#include "core/dist/scheduler.hpp"
#include "core/exec/multipath.hpp"   // GridDecompose
#include "core/io/scene_io.hpp"

#include <QPainter>
#include <QMetaObject>

#include <algorithm>
#include <thread>
#include <chrono>

namespace frep::gui {

// ── inner painted surface ───────────────────────────────────────────────────
class DistCanvas : public QWidget {
public:
    explicit DistCanvas(DistViewport* owner) : owner_(owner) {
        setMinimumSize(160, 120);
    }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    DistViewport* owner_;
};

}  // namespace frep::gui

// paint reads the owner's composited image under its mutex.
#include "gui/dist_viewport_paint.inc"

namespace frep::gui {

DistViewport::DistViewport(SceneGraph* scene, DistRenderConfig cfg,
                           QWidget* parent)
    : scene_(scene), cfg_(cfg) {
    // canvas_ is owned solely by this DistViewport (deleted in the destructor).
    // We deliberately do NOT give it a Qt parent: the LAN tab adds it to a
    // layout (which would reparent it), and if it also had a parent QObject,
    // teardown could delete it twice — once via the parent chain, once here.
    // The destructor reparents to null before deleting to neutralize whatever
    // layout it ended up in.
    canvas_ = new DistCanvas(this);
    image_ = QImage(cfg_.width, cfg_.height, QImage::Format_ARGB32);
    image_.fill(Qt::black);
    status_ = "LAN: idle";
}

DistViewport::~DistViewport() {
    // Stop the master thread first (joins it) so no more queued repaint /
    // render_completed events can be posted referencing this object.
    stop_render();
    // Delete the canvas now, while this owner is still alive (its paintEvent
    // reads owner_->image_mu_). Detach from any Qt parent first so the parent
    // doesn't also try to delete it.
    if (canvas_) {
        canvas_->setParent(nullptr);
        delete canvas_;
        canvas_ = nullptr;
    }
}

QWidget* DistViewport::widget() { return canvas_; }

void DistViewport::invalidate() { start_render(); }

QImage DistViewport::capture_image() {
    std::lock_guard<std::mutex> lk(image_mu_);
    return image_.copy();
}

QString DistViewport::status_text() const {
    std::lock_guard<std::mutex> lk(image_mu_);
    return QString::fromStdString(status_);
}

void DistViewport::stop_render() {
    cancel_ = true;            // interrupt accept() / serving loop
    running_ = false;
    if (master_thread_.joinable()) master_thread_.join();
    cancel_ = false;           // ready for the next start
}

void DistViewport::start_render() {
    stop_render();  // cancel any in-flight render first

    // Reset the composite to the configured size.
    {
        std::lock_guard<std::mutex> lk(image_mu_);
        image_ = QImage(cfg_.width, cfg_.height, QImage::Format_ARGB32);
        image_.fill(Qt::black);
        per_worker_.assign(cfg_.n_workers, 0);
        status_ = "LAN: waiting for " + std::to_string(cfg_.n_workers) +
                  " worker(s) on port " + std::to_string(cfg_.port);
    }
    on_tile_ready();

    tiles_done_ = 0;
    running_ = true;

    // Serialize the scene once (with embedded textures — remote workers may not
    // have the files). Capture by value into the thread.
    std::string scene_json =
        io::serialize_scene(*scene_, "", /*embed_textures=*/true);
    DistRenderConfig cfg = cfg_;

    master_thread_ = std::thread([this, scene_json, cfg] {
        using namespace frep::dist;

        // Decompose the frame into a grid of tiles.
        exec::GridDecompose dec(cfg.tile, cfg.tile);
        auto tiles = dec.decompose(*scene_, cfg.width, cfg.height);
        tiles_total_ = (int)tiles.size();

        std::unique_ptr<IScheduler> sched;
        if (cfg.push) sched = std::make_unique<PushScheduler>(tiles.size(),
                                                              cfg.n_workers);
        else          sched = std::make_unique<PullScheduler>(tiles.size());

        MasterConfig mc;
        mc.port = cfg.port; mc.n_workers = cfg.n_workers;
        mc.width = cfg.width; mc.height = cfg.height;
        mc.scene_json = scene_json;
        mc.verbose = false;
        mc.cancel = &cancel_;   // stop_render() sets cancel_ → master returns
        // Progress hook — runs on a serving thread; blit the tile into the
        // composite and marshal a repaint onto the GUI thread.
        mc.on_tile = [this](int /*idx*/, int worker_id,
                            const exec::TileResult& tr) {
            if (!running_) return;
            {
                std::lock_guard<std::mutex> lk(image_mu_);
                const auto& t = tr.tile;
                for (int y = t.y0; y < t.y1 && y < image_.height(); ++y) {
                    auto* row = reinterpret_cast<QRgb*>(image_.scanLine(y));
                    for (int x = t.x0; x < t.x1 && x < image_.width(); ++x) {
                        std::size_t p = ((std::size_t)(y - t.y0) * (t.x1 - t.x0)
                                         + (x - t.x0)) * 4;
                        if (p + 3 >= tr.rgba.size()) continue;
                        auto f = [&](float v) {
                            return (int)std::clamp(v * 255.0f, 0.0f, 255.0f);
                        };
                        row[x] = qRgba(f(tr.rgba[p]), f(tr.rgba[p+1]),
                                       f(tr.rgba[p+2]), 255);
                    }
                }
                if (worker_id >= 0 && worker_id < (int)per_worker_.size())
                    per_worker_[worker_id]++;
                int done = tiles_done_.fetch_add(1) + 1;
                std::string dist;
                for (std::size_t i = 0; i < per_worker_.size(); ++i)
                    dist += " w" + std::to_string(i) + "=" +
                            std::to_string(per_worker_[i]);
                status_ = "LAN: " + std::to_string(done) + "/" +
                          std::to_string(tiles_total_.load()) +
                          " tiles |" + dist;
            }
            if (!cancel_.load())
                QMetaObject::invokeMethod(this, [this] { on_tile_ready(); },
                                          Qt::QueuedConnection);
        };

        Master master(mc, std::move(tiles), *sched);
        auto t0 = std::chrono::steady_clock::now();
        auto res = master.run();
        double wall = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

        {
            std::lock_guard<std::mutex> lk(image_mu_);
            if (!res.ok)
                status_ = "LAN: error — " + res.error;
            else {
                std::string dist;
                for (std::size_t i = 0; i < res.per_worker_tiles.size(); ++i)
                    dist += " w" + std::to_string(i) + "=" +
                            std::to_string(res.per_worker_tiles[i]);
                char buf[64];
                std::snprintf(buf, sizeof buf, " in %.0f ms", wall);
                status_ = "LAN: done " + std::to_string(res.tiles_done) +
                          " tiles" + buf + " |" + dist;
            }
        }
        running_ = false;
        // If we were cancelled (the viewport is being torn down), don't queue a
        // callback into a possibly-dying object — stop_render() is joining us and
        // the viewport will be gone. Only marshal the completion on a clean end.
        if (!cancel_.load()) {
            QMetaObject::invokeMethod(this, [this, ms = (int)wall] {
                on_tile_ready();
                Q_EMIT render_completed(ms);
            }, Qt::QueuedConnection);
        }
    });
}

void DistViewport::on_tile_ready() {
    if (canvas_) canvas_->update();
}

}  // namespace frep::gui
