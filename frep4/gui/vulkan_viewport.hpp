#pragma once
// gui/vulkan_viewport.hpp
//
// QVulkanWindow-based real-time viewport — the F4 "true real-time" path.
//
// ARCHITECTURE
// ────────────
// The existing `Viewport` (QWidget + offscreen Vulkan compute + QPainter
// blit) is fast enough for editing but the host-side readback of every
// frame imposes a 5-15 ms floor. For genuinely interactive editing
// (60 FPS at 1080p), the rendered image needs to live in a swapchain
// image directly, with QVulkanWindow handling the present.
//
// The integration plan, in order:
//   1. QVulkanWindow creates VkInstance + VkDevice + VkSwapchainKHR
//   2. Our compute shader binds the current swapchain image as a
//      storage image (output) and dispatches compute writing RGBA8 to it
//   3. A pipeline barrier transitions the image from
//      STORAGE → PRESENT_SRC layout before vkQueuePresentKHR
//   4. Camera/scene state is read from a shared `SceneGraph*` (same
//      ownership as Viewport)
//
// CURRENT STATUS
// ──────────────
// This header declares the interface; the .cpp provides:
//   - A working VulkanViewport that opens a QVulkanWindow and renders
//     the live SceneGraph via a compute shader dispatched into a
//     private storage image, then blitted into the swapchain image
//     via vkCmdBlitImage every frame at the display's refresh rate.
//     Scene-structure changes trigger a pipeline rebuild on the next
//     frame.
//   - Full scene support, including MeshSDF nodes and image-based
//     texture maps. The renderer allocates HOST_VISIBLE storage
//     buffers (bindings 1 / 2) for mesh voxel grids and texture pixel
//     arrays respectively, with the same layout as the offscreen
//     VulkanCtx path so both code paths share the emitted GLSL/SPV.
//     Buffer uploads happen as part of pipeline rebuild — typical
//     sizes (256 KB voxel grids, 1 MB textures) re-upload in well
//     under a millisecond on PCIe 3.0+ hardware.
//   - Mouse interaction (left-drag orbit, wheel zoom) wired directly
//     to scene_->camera() — camera-only changes don't trigger
//     pipeline rebuilds since the camera is push-constant data.
//   - Graceful fallback: vulkan_viewport_available() returns false on
//     environments without proper Vulkan swapchain support (most
//     notably the Mesa llvmpipe used in CI/sandbox), in which case
//     callers should construct the offscreen `Viewport` instead.
//   - A secondary fallback: if pipeline construction fails for any
//     reason (GLSL emit failure, SPV compile failure, descriptor
//     allocation failure), the renderer clears the swapchain to a
//     recognisable dark-teal so the user sees the window is alive
//     but pipeline build was rejected, rather than crashing.
//
// The development sandbox exposes only Mesa's `llvmpipe` software
// Vulkan driver, which `vulkan_viewport_available()` deliberately
// rejects — QVulkanWindow assumes a real swapchain capable of
// presenting to a window surface. The renderer implementation
// therefore compiles and links cleanly but its rendering output is
// not live-verifiable here; real hardware (or a desktop Mesa stack
// with a visible display) is needed to validate the actual frames.
//
// USAGE
// ─────
//   if (vulkan_viewport_available() && user_wants_realtime) {
//       // Get a QWidget container for the QVulkanWindow
//       auto* vp = new VulkanViewport(scene, parent);
//       // Connect signals the same as Viewport
//   } else {
//       auto* vp = new Viewport(scene, parent);  // existing fallback
//   }

#include "gui/iviewport.hpp"

#include <QObject>
#include <QString>

class QWidget;
class QVulkanInstance;

namespace frep {
class SceneGraph;
}

namespace frep::gui {

// Forward declaration for the QVulkanWindow subclass that owns the
// compute renderer. Declared in gui/frep_vulkan_window.hpp — public
// consumers of this header still don't pull in Vulkan headers since
// they only need a `FRepVulkanWindow*` member.
class FRepVulkanWindow;

// Returns true when this build can construct a working VulkanViewport
// at runtime. The check is conservative — it must be safe to call from
// MainWindow's ctor before any Vulkan resource has been created.
//
// Specifically, returns false when ANY of:
//   - Qt was built without Vulkan support
//   - No VkInstance can be created (no driver, no loader)
//   - The Vulkan device doesn't expose VK_KHR_swapchain
//   - The detected device is Mesa llvmpipe (software path — not what
//     QVulkanWindow is designed for; the existing offscreen Viewport
//     covers this case at no extra cost)
//
// The check itself creates a transient VkInstance to probe; this
// takes ~10 ms but only runs once at startup.
bool vulkan_viewport_available();

// A real-time viewport backed by QVulkanWindow + a compute/blit
// renderer. Implements the IViewport interface so MainWindow can treat
// it identically to the offscreen Viewport when wiring signals.
//
// On environments where vulkan_viewport_available() is false, callers
// should not construct this — instead create the offscreen Viewport
// and wrap it in OffscreenViewportAdapter.
class VulkanViewport : public IViewport {
    Q_OBJECT
public:
    // Factory: returns a heap-allocated VulkanViewport that owns its
    // QVulkanWindow + container widget. The caller (MainWindow) takes
    // ownership; pass `parent` for Qt's object-tree cleanup.
    // `scene` must outlive the returned object.
    //
    // The legacy create() returning QWidget* is kept for backward
    // compatibility with callers that don't use the IViewport
    // interface — it's implemented by extracting widget() from a
    // freshly-constructed IViewport instance.
    static IViewport* create_iv(SceneGraph* scene, QWidget* parent);
    static QWidget*   create   (SceneGraph* scene, QWidget* parent);

    // IViewport implementation.
    QWidget* widget() override;
    void     invalidate() override;
    void     set_tracer_config(const TracerConfig& cfg) override;
    void     set_ssaa(int n) override;
    void     set_camera_control_config(const CameraControlConfig& c) override;
    QImage   capture_image() override;
    QString  status_text() const override { return status_; }
    QString  metrics_text() const override;

    // Status message — populated during initialization, useful for the
    // GUI status bar to display "Real-time GPU viewport active" or
    // similar. Empty string means "no status to report".
    static QString last_status();

private:
    explicit VulkanViewport(QWidget* container,
                            FRepVulkanWindow* window,
                            QString status);
    QWidget*          container_ = nullptr;  // owned via Qt parent-tree
    FRepVulkanWindow* window_    = nullptr;  // weak ref (Qt-owned), used for config forwarding
    QString  status_;
    QString  cull_method_;   // last-configured cull method, for the metrics HUD
    int      last_gpu_ms_ = -1;   // most recent GPU frame time (timestamp query)
};

} // namespace frep::gui
