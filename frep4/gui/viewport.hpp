#pragma once
// gui/viewport.hpp
//
// Viewport — a QWidget that shows the rendered SDF scene.
// Has orbit camera controls (mouse drag, wheel) and an async render thread.

#include <QImage>
#include <QPoint>
#include <QTimer>
#include <QWidget>

#include "gui/camera_control_config.hpp"

#include <atomic>
#include <memory>
#include <vector>

// Forward declarations — keep the header lean for MOC.
class QMouseEvent;
class QWheelEvent;
class QResizeEvent;
class QPaintEvent;
class QString;

namespace frep::gpu { class VulkanCtx; }
namespace frep::gpu { class CudaCtx; }

namespace frep {
class SceneGraph;
class IncrementalCompiler;
class ScenePicker;
class IncrementalParams;
}

namespace frep::gui {

class Viewport : public QWidget {
    Q_OBJECT

public:
    explicit Viewport(SceneGraph* scene, QWidget* parent = nullptr);
    ~Viewport() override;

    IncrementalCompiler& compiler();

    // Super-sampling anti-aliasing factor (1 = none, 2 = 2x2, 3 = 3x3).
    // Higher values quadruple render time; 2 is usually the sweet spot.
    void set_ssaa(int n);
    int  ssaa() const { return ssaa_; }

    // Orbit-camera control tuning (mouse sensitivity, zoom range, ...).
    // Shared struct so the offscreen and real-time paths stay identical;
    // mouse_sensitivity is surfaced in the Render tab.
    void set_camera_control_config(const CameraControlConfig& c) { cam_cfg_ = c; }
    const CameraControlConfig& camera_control_config() const { return cam_cfg_; }

    // Which GPU retargeting path the GPU mode uses:
    //   Cuda   — Model → IR → PTX → CUDA → GPU raymarch  (GPU-IR path)
    //   Vulkan — Model → GLSL → SPIR-V → Vulkan → GPU raymarch (GPU-GLSL)
    // Both end in the same offscreen-readback → QImage presentation + the
    // same CPU box-downsample SSAA, so they differ only in the executor
    // (the path's render stage), keeping them comparable.
    enum class GpuBackend { Cuda, Vulkan };

    // Toggle between the CPU JIT path (default) and a GPU compute path.
    // On enable, the next render emits for the chosen backend, builds its
    // context (cached + reused while the scene structure is unchanged), and
    // renders a frame read back into a QImage. Returns false if the chosen
    // backend is unavailable; the viewport stays on CPU in that case.
    bool set_gpu_mode(bool on, GpuBackend backend = GpuBackend::Cuda);
    bool gpu_mode() const { return gpu_mode_; }
    // Brief one-line message about the last GPU init / render outcome.
    QString gpu_status() const { return gpu_status_; }

    // Returns the most recently rendered image. Empty if the viewport has
    // never produced a frame (e.g. before show()). The image is in
    // QImage::Format_RGBA8888 and lives until the next render.
    const QImage& current_image() const { return image_; }

Q_SIGNALS:
    // structure_unchanged: true when the scene changed but only parameter
    // values moved (the FRepNode tree structure was identical to last frame).
    // Currently informational only; a future per-parameter incremental path
    // would use this to skip full recompilation.
    void render_completed(double render_ms, double total_ms,
                          bool was_cached, bool structure_unchanged);
    // Emitted on a click on an object (or with an empty string on a click in empty space).
    void object_picked(const QString& object_id);

public Q_SLOTS:
    void invalidate();

protected:
    void paintEvent       (QPaintEvent*  e) override;
    void mousePressEvent  (QMouseEvent*  e) override;
    void mouseReleaseEvent(QMouseEvent*  e) override;
    void mouseMoveEvent   (QMouseEvent*  e) override;
    void wheelEvent       (QWheelEvent*  e) override;
    void resizeEvent      (QResizeEvent* e) override;

private Q_SLOTS:
    void on_render_tick();

private:
    void recompute_camera();
    void do_render();
    void try_pick(const QPoint& pos);

    // GPU-IR (CUDA) helpers — implemented behind FREP_HAS_CUDA.
    bool cuda_available() const;
    void reset_cuda_ctx();
    bool render_gpu_cuda(int hi_w, int hi_h, std::size_t scene_hash,
                         std::vector<std::uint8_t>& rgba);
    void refresh_cuda_params();   // pull live param values into the buffer

    SceneGraph*                          scene_;
    std::unique_ptr<IncrementalCompiler> compiler_;
    std::unique_ptr<ScenePicker>         picker_;
    QImage                               image_;
    std::vector<float>                   pixels_;
    QTimer*                              render_timer_ = nullptr;
    int                                  ssaa_ = 1;

    // Camera orbit state
    float cam_yaw_   = 0.0f;
    float cam_pitch_ = 0.3f;
    float cam_dist_  = 9.0f;
    CameraControlConfig cam_cfg_;   // orbit/zoom tuning (shared struct)

    // Mouse state
    bool   dragging_ = false;
    bool   moved_since_press_ = false;   // distinguishes a click from a drag
    QPoint last_pos_;
    QPoint press_pos_;

    std::atomic<bool> dirty_     {true};
    std::atomic<bool> rendering_ {false};

    // ── GPU compute mode (optional) ─────────────────────────────────────────
    // We hold the VulkanCtx (and the GlslEmitResult that produced it)
    // behind unique_ptrs so the header doesn't have to pull in Vulkan
    // headers. Created on demand in do_render_gpu().
    bool                       gpu_mode_ = false;
    GpuBackend                 gpu_backend_ = GpuBackend::Cuda;
    QString                    gpu_status_;
    std::unique_ptr<gpu::VulkanCtx>      gpu_ctx_;
#ifdef FREP_HAS_CUDA
    std::unique_ptr<gpu::CudaCtx>        cuda_ctx_;
    // CUDA compile amortization: the CUDA module is keyed on the scene
    // *structure* (topology), not the full scene hash. Parameter-only edits
    // refresh the runtime params buffer and re-launch — no codegen / NVPTX /
    // module reload. Built when incremental codegen runs.
    std::unique_ptr<frep::IncrementalParams> cuda_params_;
    std::size_t                cuda_structure_hash_ = 0;
    bool                       cuda_incremental_ = false;
#endif
    std::size_t                gpu_scene_hash_ = 0;  // detect structure changes

    bool do_render_gpu();    // returns true on success
};

} // namespace frep::gui
