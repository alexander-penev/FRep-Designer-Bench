#pragma once
// gui/frep_vulkan_window.hpp
//
// QVulkanWindow subclass used by VulkanViewport. Carries:
//   - A pending tracer config + SSAA factor buffer (applied when Qt
//     finally builds the renderer)
//   - A QTimer that samples the renderer's per-frame GPU timestamp at
//     ~10 Hz and emits `render_time_sampled` for the status bar
//   - Mouse handlers that translate drag/wheel into orbital camera
//     updates on scene_->camera()
//
// Lives in its own header (rather than the .cpp where it logically
// belongs) so AutoMoc's heuristic — which only scans headers for
// Q_OBJECT classes — finds it. The manual qt6_wrap_cpp dance in
// CMakeLists.txt for the other interface headers covers this one
// too. See the CHANGELOG entry for v4.0.4 explaining why Qt 6.4.2
// needs the manual-moc workaround.

#include "core/compiler/codegen.hpp"   // TracerConfig
#include "gui/camera_control_config.hpp"

#include <QObject>
#include <QPoint>
#include <QTimer>
#include <QVulkanWindow>

#include <algorithm>
#include <memory>

namespace frep {
class SceneGraph;
class ScenePicker;
}

namespace frep::gui {

class ComputeBlitRenderer;   // defined in vulkan_viewport.cpp

class FRepVulkanWindow : public QVulkanWindow {
    Q_OBJECT
public:
    explicit FRepVulkanWindow(SceneGraph* scene);
    ~FRepVulkanWindow() override;

    QVulkanWindowRenderer* createRenderer() override;

    // Push a new TracerConfig in from the outside. If the renderer
    // has already been constructed, forward to it directly. Otherwise
    // buffer in `pending_cfg_` so createRenderer() applies it when
    // Qt finally gets around to building the renderer.
    void set_tracer_config(const TracerConfig& cfg);

    // Push a new SSAA factor. Same buffer-then-forward dance.
    void set_ssaa(int n);

    // Orbit-camera control tuning, shared with the offscreen path.
    void set_camera_control_config(const CameraControlConfig& c) { cam_cfg_ = c; }

Q_SIGNALS:
    // Emitted at ~10 Hz with the most recent GPU-side per-frame time
    // (milliseconds). VulkanViewport forwards this to IViewport's
    // render_completed signal, which the status bar listens on.
    void render_time_sampled(int ms);

    // Emitted when a left-click (with no drag) hit an object. id is
    // the SceneGraph object name, or empty for a miss. additive is
    // true when Ctrl was held — consumers (MainWindow) interpret
    // additive as "add to existing inspector selection" rather than
    // "replace it". Wired through VulkanViewport::object_picked.
    void object_picked(const QString& id, bool additive);

protected:
    void mousePressEvent  (QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseMoveEvent   (QMouseEvent* e) override;
    void wheelEvent       (QWheelEvent* e) override;

private Q_SLOTS:
    void sample_timing();

private:
    void apply_camera();
    void try_pick(const QPoint& pos, bool additive);

    SceneGraph*              scene_           = nullptr;
    ComputeBlitRenderer*     active_renderer_ = nullptr;  // not owned — Qt owns the renderer lifetime
    TracerConfig             pending_cfg_;                // applied at createRenderer time
    int                      pending_ssaa_    = 1;        // applied at createRenderer time
    QTimer                   sample_timer_;               // ~10 Hz GPU time poll
    QPoint                   press_pos_;                  // start of current drag — used for click-vs-drag distinction
    Qt::KeyboardModifiers    press_modifiers_ = Qt::NoModifier;
    QPoint                   last_pos_;
    float                    cam_yaw_         = 0.0f;
    float                    cam_pitch_       = 0.0f;
    float                    cam_dist_        = 5.0f;
    bool                     dragging_        = false;
    bool                     orbit_init_      = false;  // lazy: read scene_->camera() on first interaction
    CameraControlConfig      cam_cfg_;                  // orbit/zoom tuning (shared struct)

    // CPU-side picker. Real-time GPU shader doesn't have a separate
    // ID pass yet; we re-use the same ScenePicker the offscreen path
    // builds and run it CPU-side on click. Builds once on first use,
    // rebuilds when scene structure changes (valid_for() check).
    std::unique_ptr<ScenePicker> picker_;
};

} // namespace frep::gui
