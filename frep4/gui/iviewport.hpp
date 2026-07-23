#pragma once

// gui/iviewport.hpp
//
// IViewport — the minimal interface MainWindow needs from "the thing
// that renders the scene". Both the offscreen QPainter-based Viewport
// and the QVulkanWindow-backed real-time renderer implement it,
// letting MainWindow swap between them at startup without conditional
// code in the signal-wiring path.
//
// Deliberately narrow scope: this interface covers only what's
// rendering-mode-independent. CPU-side concepts that only make sense
// for the offscreen path — the JIT compiler, SSAA, CPU/GPU mode
// toggle, current image readback for screenshots — stay as
// Viewport-specific methods, accessed via downcasting only when the
// caller knows the offscreen path is active.
//
// Why a QObject interface rather than a templated abstraction: Qt's
// signal/slot system needs metaobject registration, which is
// incompatible with C++ abstract templates. A plain interface class
// inheriting QObject gives us the cleanest path to a polymorphic
// signal source without ditching the meta-object machinery.

#include "core/compiler/codegen.hpp"   // TracerConfig
#include "gui/camera_control_config.hpp"

#include <QObject>
#include <QImage>
#include <QString>

class QWidget;

namespace frep::gui {

class IViewport : public QObject {
    Q_OBJECT
public:
    explicit IViewport(QObject* parent = nullptr) : QObject(parent) {}
    ~IViewport() override = default;

    // The Qt widget that renders the scene. The viewport owns it;
    // callers add it to a layout but must NOT delete it. Lifetime
    // matches the IViewport instance itself.
    virtual QWidget* widget() = 0;

    // Mark the current render as stale. Implementations are free to
    // batch — a single invalidate() may schedule rather than render
    // immediately. For continuous-render viewports (real-time) this
    // is a no-op since they re-render every frame anyway.
    virtual void invalidate() = 0;

    // Push a new TracerConfig to the renderer. Both implementations
    // honour shading_model, enable_shadows, shadow_softness,
    // enable_ao, and ao_strength identically — the offscreen path
    // forwards to its IncrementalCompiler (which triggers a JIT
    // recompile), the real-time path stashes the config and lets the
    // per-frame scene-hash check trigger a GLSL re-emit + pipeline
    // rebuild on the next frame.
    //
    // Implementations are expected to no-op if the config is bit-
    // identical to what they already hold, so MainWindow can call
    // this freely without worrying about needless recompiles.
    virtual void set_tracer_config(const TracerConfig& cfg) = 0;

    // Set the super-sampling AA factor. 1 = native resolution, 2 = 2x2
    // (4 samples per output pixel), 3 = 3x3. Higher values quadratic-
    // ally increase shader work but smooth out edge aliasing. Both
    // backends now implement this — the offscreen path scales its
    // internal render buffer; the real-time path resizes its
    // storage image and uses VK_FILTER_LINEAR during the swapchain
    // blit to do the downsample on the GPU.
    virtual void set_ssaa(int n) = 0;

    // Push orbit-camera control tuning (mouse sensitivity, zoom range,
    // pitch clamp). Default no-op so backends that don't drive their own
    // camera (if any) needn't implement it. Both real backends override.
    virtual void set_camera_control_config(const CameraControlConfig&) {}

    // Enable/disable adaptive build-time spatial guards (BVH approach 1).
    // On by default in the CPU compiler; only the offscreen-CPU backend
    // honours it (the GPU path doesn't use the JIT scene_sdf). Default
    // no-op so other backends can ignore it.
    virtual void set_spatial_guards_enabled(bool) {}

    // Toggle an on-viewport metrics HUD (ms/frame, cull method, cull rate).
    // Default no-op; the real-time viewport overrides it (task 2).
    virtual void set_metrics_overlay(bool) {}

    // Current metrics as a short multi-line string for the HUD overlay
    // (ms/frame, and — where the backend has it — cull method and cull rate).
    // Empty string means "no metrics available"; the HUD hides itself then.
    virtual QString metrics_text() const { return {}; }

    // Capture the current rendered frame as a CPU-side QImage for export.
    // Default returns a null image (caller treats that as "nothing to
    // export"). The offscreen backend returns its last readback image;
    // the real-time backend does a true GPU framebuffer grab
    // (QVulkanWindow::grab → vkCmdCopyImage internally), which is more
    // faithful than a screen-coordinate grab and unaffected by
    // overlapping windows or compositor effects.
    virtual QImage capture_image() { return {}; }

    // Human-readable string describing render-time state — backend
    // name, mode, last frame time, cache hit. Used by the status bar.
    // Empty string is allowed (means "nothing to show").
    virtual QString status_text() const { return {}; }

Q_SIGNALS:
    // Emitted every time a frame completes rendering. `ms` is the
    // wall-clock time for that frame (whatever the backend chooses
    // to measure — for offscreen, the full CPU/GPU dispatch; for
    // real-time, the per-frame command-buffer build time). The
    // status bar uses this to display "Render NNNms" without caring
    // about which backend is active.
    void render_completed(int ms);

    // Emitted when the user clicks an object in the viewport and the
    // raycast resolves to a specific scene object ID. The inspector
    // connects to this to sync its selection.
    void object_picked(const QString& object_id);
};

} // namespace frep::gui
