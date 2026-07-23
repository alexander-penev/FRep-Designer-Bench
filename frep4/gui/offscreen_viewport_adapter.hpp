#pragma once

// gui/offscreen_viewport_adapter.hpp
//
// Wraps the existing offscreen `Viewport` (QWidget) as an IViewport.
//
// We can't make Viewport itself inherit IViewport because both
// inherit QObject and Qt forbids multiple-QObject inheritance. So we
// give MainWindow an IViewport handle, and this adapter forwards
// signals and exposes the underlying QWidget.
//
// MainWindow still keeps a `Viewport*` pointer for the CPU-specific
// API (compiler(), set_gpu_mode(), set_ssaa(), current_image(), ...),
// available only when the offscreen path is active. The IViewport
// pointer is the universal handle that works in both modes.

#include "gui/iviewport.hpp"
#include "gui/viewport.hpp"
#include "core/compiler/incremental.hpp"  // IncrementalCompiler full type

namespace frep { class SceneGraph; }

namespace frep::gui {

class OffscreenViewportAdapter : public IViewport {
    Q_OBJECT
public:
    OffscreenViewportAdapter(Viewport* vp, SceneGraph* scene,
                             QObject* parent = nullptr)
        : IViewport(parent), vp_(vp), scene_(scene)
    {
        // Adapt Viewport's 4-arg render_completed signal (render_ms,
        // total_ms, was_cached, structure_unchanged) to IViewport's
        // simpler 1-arg form (just the render time in milliseconds).
        // The other fields are still available for offscreen-specific
        // status bar logic that needs more detail — that path bypasses
        // the IViewport interface and reads from MainWindow's
        // dedicated Viewport pointer.
        connect(vp_, &Viewport::render_completed,
                this, [this](double render_ms,
                             double /*total_ms*/,
                             bool   /*was_cached*/,
                             bool   /*structure_unchanged*/) {
                    Q_EMIT render_completed(static_cast<int>(render_ms));
                });
        connect(vp_, &Viewport::object_picked,
                this, [this](const QString& id) {
                    Q_EMIT object_picked(id);
                });
    }

    QWidget* widget() override { return vp_; }
    void     invalidate() override { vp_->invalidate(); }
    QString  status_text() const override { return vp_->gpu_status(); }

    void set_tracer_config(const TracerConfig& cfg) override {
        // Forward to the IncrementalCompiler living inside the offscreen
        // Viewport. Both the CPU JIT path (uses compiler.tracer_config()
        // directly) and the offscreen GPU path (passes the same config
        // through to GlslEmitter::emit) read from this single source.
        // force_recompile rebuilds on the next paint event.
        vp_->compiler().tracer_config() = cfg;
        if (scene_)
            vp_->compiler().force_recompile(*scene_);
        vp_->invalidate();
    }

    void set_ssaa(int n) override {
        // Offscreen path: forward directly to the underlying Viewport's
        // SSAA setter. The viewport scales its internal render buffer
        // and re-renders on the next paint event.
        vp_->set_ssaa(n);
    }

    void set_camera_control_config(const CameraControlConfig& c) override {
        vp_->set_camera_control_config(c);
    }

    void set_spatial_guards_enabled(bool on) override {
        // Forward to the CPU compiler and force a recompile so the change
        // takes effect on the next paint. No-op-cheap if unchanged.
        if (vp_->compiler().spatial_guards_enabled() == on) return;
        vp_->compiler().set_spatial_guards_enabled(on);
        if (scene_)
            vp_->compiler().force_recompile(*scene_);
        vp_->invalidate();
    }

    QImage capture_image() override {
        // Offscreen path already has the rendered frame on the CPU side
        // (the CPU JIT writes it directly; the offscreen GPU path reads
        // it back after dispatch). Just hand it over.
        return vp_->current_image();
    }

    // Access for MainWindow when it knows the offscreen path is active.
    // Returns the wrapped Viewport for offscreen-specific operations
    // (compiler config, SSAA, screenshots). nullptr when the active
    // viewport is the real-time one — but in that case MainWindow won't
    // hold an OffscreenViewportAdapter to begin with.
    Viewport* offscreen() const { return vp_; }

private:
    Viewport*    vp_;     // not owned — MainWindow owns the QWidget lifetime
    SceneGraph*  scene_;  // not owned — for tracer-config-driven recompiles
};

} // namespace frep::gui
