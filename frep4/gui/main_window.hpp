#pragma once
// gui/main_window.hpp
//
// MainWindow — viewport + property panel + status bar + node graph editor.

#include <QMainWindow>
#include <QStringList>

#include "core/compiler/codegen.hpp"   // TracerConfig
#include "core/exec/multipath.hpp"        // exec::PathKind
#include "gui/camera_control_config.hpp"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

class QListWidget;
class QDoubleSpinBox;
class QSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTimer;
class QAction;
class QComboBox;
class QCheckBox;
class QSplitter;

namespace frep {
class SceneGraph;
namespace plugin { class PluginRegistry; }
namespace undo { class UndoStack; }
}

namespace frep::gui {

class IViewport;
class Viewport;
class SceneInspector;
class ExpressionEditor;
class MaterialEditor;
class NodeGraphEditor;

// Which rendering backend the viewport uses. The three are mutually
// exclusive and runtime-switchable via the Render-tab selector / Render
// menu. Switching tears down the previous backend completely (its
// timers, GPU contexts, and the widget itself) before building the new
// one — see MainWindow::switch_render_mode — so no inactive backend
// keeps doing work under the surface.
enum class RenderMode {
    OffscreenCPU = 0,   // Model→IR→JIT→CPU raymarch→offscreen readback→image
    OffscreenGPU = 1,   // Model→IR→PTX→CUDA→GPU raymarch→offscreen readback→image
    Realtime     = 2,   // Model→GLSL→SPIR-V→Vulkan→GPU raymarch→swapchain→screen
    Rtx          = 3,   // Model→GLSL RT shaders→Vulkan ray tracing (single path,
                        //   via ExecutorViewport with one GPU_RTX executor)
    Multipath    = 4,   // ExecutorViewport: any/all of the 4 paths via IExecutor,
                        //   single or split across paths (strips/weighted/tiles)
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(SceneGraph* scene, plugin::PluginRegistry* reg,
               bool realtime_viewport = false,
               QWidget* parent = nullptr);
    ~MainWindow() override;

    // Public render-mode access, primarily for smoke-testing the
    // teardown/rebuild path (FREP_SMOKE_CYCLE_MODES) and for any future
    // scripting hook. request_render_mode routes through the same
    // switch_render_mode used by the UI, including the hardware
    // availability guard.
    void       request_render_mode(RenderMode m) { switch_render_mode(m); }
    RenderMode current_render_mode() const { return current_mode_; }

private Q_SLOTS:
    void on_render_completed(double render_ms, double total_ms,
                             bool was_cached, bool structure_unchanged);
    void on_scene_changed();
    void on_add_primitive(const QString& plugin_name);
    void on_open_scene();
    void on_save_scene();
    void on_import_mesh();
    void on_export_image();
    void on_export_mesh();
    // Node graph editor → scene: builds the tree and reloads the scene.
    void on_graph_changed();
    // Inspector selection changed → reload the node graph view with the
    // newly-selected object's geometry. Keeps the graph "looking at" a
    // single object the way Houdini's network view does.
    void on_inspector_selection_changed(const QString& object_id);

private:
    void build_ui();
    void build_menu();
    void build_toolbar();
    QWidget* build_side_panel();
    QWidget* build_lights_panel();
    QWidget* build_plugins_panel();      // registered plugin list (own tab)
    QWidget* build_lan_panel();          // distributed (LAN) render controls

    // LAN tab state. The master renders into the main viewport via a
    // DistViewport backend; the worker runs on its own thread serving a master.
    void start_lan_master();
    void stop_lan_master();
    void start_lan_worker();
    void stop_lan_worker();
    void sync_path_menu_to_combo();
    // LAN tab widgets. Defined inline so the unique_ptr<LanWidgets> member sees
    // a complete type at every destruction site.
    struct LanWidgets {
        QSpinBox*    port = nullptr;
        QSpinBox*    workers = nullptr;
        QSpinBox*    tile = nullptr;
        QComboBox*   scheduler = nullptr;
        QPushButton* start_master = nullptr;
        QPushButton* stop_master = nullptr;
        QLabel*      master_status = nullptr;
        QLineEdit*   host = nullptr;
        QSpinBox*    wport = nullptr;
        QComboBox*   wpath = nullptr;
        QPushButton* start_worker = nullptr;
        QPushButton* stop_worker = nullptr;
        QLabel*      worker_status = nullptr;
        QCheckBox*   worker_preview = nullptr;   // debug: show rendered tiles
        QListWidget* worker_preview_list = nullptr;  // chronological tile thumbnails
        class DistViewport* dist_vp = nullptr;
        QTimer* master_poll = nullptr;   // status poller while master runs
    };
    std::unique_ptr<LanWidgets> lan_;
    std::thread             lan_worker_thread_;
    std::atomic<bool>       lan_worker_running_{false};

    // Render-mode lifecycle (Stage 1: unified render-mode selection).
    //
    // build_viewport creates the IViewport for the given mode, wires its
    // universal signals, applies the current tracer config / SSAA, and
    // places its widget into viewport_container_. It does NOT touch any
    // previously-built viewport — switch_render_mode handles teardown.
    //
    // switch_render_mode tears the current viewport down (delete the
    // widget → its destructor stops timers and releases GPU contexts),
    // then calls build_viewport for the new mode and updates
    // current_mode_ + the selector UI. A request for Realtime when
    // hardware Vulkan is unavailable falls back to OffscreenCPU.
    void build_viewport(RenderMode mode);
    void switch_render_mode(RenderMode mode);

    // Rebuilds the lights table from scene_->lights() (call after add/remove
    // or on scene reload).
    void refresh_lights_panel();

    // Two-way sync between SceneGraph and the node graph view:
    //   * sync_graph_to_scene() rebuilds the active object's geometry
    //     from the current node graph. Leaves OTHER scene objects alone.
    //   * sync_scene_to_graph() rebuilds the node graph view from the
    //     active scene object. Used when the active object's geometry
    //     was changed elsewhere (toolbar, inspector, JSON load).
    // A `syncing_` flag prevents one direction from triggering the other
    // and looping forever — both functions set it true while running.
    void sync_graph_to_scene();
    void sync_scene_to_graph();
    bool syncing_ = false;
    // Which scene object the graph view currently represents. Empty when
    // the scene has no objects, or none has been selected yet. Updated
    // by on_inspector_selection_changed and when the scene mutates.
    std::string active_object_id_;

    // Updates Undo/Redo action enable-state and tooltips after the stack
    // changes (push/undo/redo/clear).
    void refresh_undo_redo_actions();

    // Clones the currently-selected objects in place (nudged slightly so
    // the copies are visible), selects the clones, and pushes one undo
    // entry per clone. Shared by Duplicate (Ctrl+D) and Paste.
    void duplicate_selection();

    // Clones whatever ids are on the internal clipboard (populated by
    // Copy). No-op if the clipboard is empty or the source objects were
    // deleted in the meantime.
    void paste_clipboard();

    // Internal object clipboard — ids captured by Edit→Copy. Cleared and
    // repopulated on each Copy; consumed (but not cleared) by Paste so
    // repeated pastes keep working.
    QStringList clipboard_ids_;

    // Forces a JIT recompile of the active scene on the offscreen
    // path. No-op on the real-time path (where the compute pipeline
    // rebuilds itself on its own schedule via the renderer's hash
    // check). Saves us from sprinkling `if (viewport_)` over every
    // editor callback.
    void force_recompile_if_offscreen();

    // Pushes the current `render_config_` to the active viewport, so
    // a Render-tab slider edit makes its way into either path
    // (offscreen → IncrementalCompiler::tracer_config(), real-time →
    // ComputeBlitRenderer::tracer_cfg_). Cheap; safe to call from any
    // Render-tab callback.
    void apply_render_config();

    SceneGraph*              scene_;
    plugin::PluginRegistry*  registry_;

    // The single source of truth for the ray tracer's visual settings
    // — shading model, shadow / AO toggles, softness, strength. The
    // Render-tab widgets bind both ways to this struct: edits mutate
    // it and call apply_render_config(); a scene reload or "Reset"
    // resets it and refreshes the widgets. Both rendering paths read
    // from this single struct (offscreen path forwards it into its
    // IncrementalCompiler; real-time path forwards into the renderer).
    //
    // Default-constructed = TracerConfig defaults (CookTorrance,
    // shadows on, AO on) so a freshly-launched application picks up
    // the same look on both paths.
    TracerConfig             render_config_;
    bool                     show_metrics_overlay_ = false;
    QLabel*                  metrics_label_ = nullptr;   // on-viewport HUD (task 2)
    QTimer*                  metrics_timer_ = nullptr;   // refreshes the HUD text

    // Universal viewport handle: works for both offscreen and
    // real-time paths. Signal connections (render_completed,
    // object_picked), invalidate(), status_text() all go through this.
    // Pointed to either an OffscreenViewportAdapter or a
    // VulkanViewport instance, depending on the active backend.
    IViewport*               viewport_iv_ = nullptr;

    // Container that holds the active viewport widget. Its layout has
    // exactly one child — the current backend's widget. switch_render_mode
    // clears the layout (destroying the old widget, which stops its
    // timers / releases its GPU context via the destructor) and inserts
    // the new backend's widget. Wrapping the viewport in a container lets
    // us swap QWidget (offscreen) and QVulkanWindow-wrapper (real-time)
    // freely, since they share no common concrete base beyond QWidget.
    QWidget*                 viewport_container_ = nullptr;
    QSplitter*               main_splitter_ = nullptr;

    // The active render mode. Initialised from the constructor (which
    // honours the --realtime CLI flag) and changed at runtime by
    // switch_render_mode.
    RenderMode               current_mode_ = RenderMode::OffscreenCPU;

    // Render-tab selector mirroring the three modes. Kept in sync with
    // current_mode_ (QSignalBlocker guards programmatic updates).
    // Path selection UI. The Render-tab combo is a checkable-dropdown picker
    // (tick 1+ paths); the Render menu mirrors it with checkable path actions.
    // A layout combo / submenu picks the multi-view split. All drive the single
    // ExecutorViewport via apply_multipath_selection().
    // Which concrete viewport backend is currently installed. The path
    // selection picks the most efficient one: a single GLSL path uses the
    // real-time Vulkan swapchain, single CPU/GPU-IR use the offscreen Viewport
    // (CPU or CUDA), single RTX and any multi-path selection use the
    // ExecutorViewport (which composites). Tracked so we only tear down and
    // rebuild when the selection crosses a backend boundary.
    enum class Backend { None, Executor, RealtimeGlsl, OffscreenCpu, OffscreenCuda };

    // The paths shown in the picker: the local executors plus the special "lan"
    // orchestration path (which has no local executor — it drives the LAN tab's
    // master and re-renders interactively).
    static std::vector<frep::exec::PathKind> picker_paths();
    Backend                  active_backend_ = Backend::None;
    int                      last_remote_port_ = -1;     // lan endpoint last pushed
    int                      last_remote_workers_ = -1;
    Backend desired_backend_for(const std::vector<frep::exec::PathKind>& paths) const;
    // Tear down the current viewport and install the backend for `paths`.
    void install_backend(Backend backend,
                         const std::vector<frep::exec::PathKind>& paths);

    class ExecutorViewport*  executor_vp_ = nullptr;
    class CheckableComboBox* path_combo_ = nullptr;
    QComboBox*               multi_layout_combo_ = nullptr;
    std::vector<::QAction*>  path_menu_actions_;    // parallel to exec::local_paths()
    std::vector<::QAction*>  layout_menu_actions_;  // split / weighted / tiles
    void apply_multipath_selection();

    // Current SSAA factor, mirrored from the SSAA combo. Re-applied to a
    // freshly-built viewport after a mode switch so the setting survives.
    int                      current_ssaa_ = 1;

    // Orbit-camera control tuning (mouse sensitivity, zoom, ...). Held
    // here so it survives mode switches and is re-applied by
    // build_viewport to whichever backend it creates.
    CameraControlConfig      current_cam_cfg_;

    // Offscreen-only handle. Non-null exactly when an offscreen backend
    // (CPU or GPU) is active; nullptr in real-time mode. Provides access
    // to offscreen-specific APIs (IncrementalCompiler config, GPU-mode
    // toggle, screenshot image readback). Re-pointed by build_viewport on
    // every mode switch. The value is the same QWidget that
    // viewport_iv_->widget() returns, just with a more specific type.
    Viewport*                viewport_    = nullptr;

    // Initial-mode hint from the --realtime CLI flag. Only consulted in
    // the constructor to pick the starting RenderMode (and only honoured
    // if hardware Vulkan is present); runtime mode changes go through
    // switch_render_mode and update current_mode_ instead.
    bool                     realtime_viewport_ = false;
    SceneInspector*          inspector_   = nullptr;
    ExpressionEditor*        expression_editor_ = nullptr;
    MaterialEditor*          material_editor_   = nullptr;
    // Dropdown that selects which scene object's tree the node graph
    // edits. Populated from scene_->objects() whenever the scene
    // changes. Setting the current text routes through
    // on_inspector_selection_changed.
    QComboBox*               graph_object_picker_ = nullptr;
    // When unchecked, the graph stays locked to its current object
    // even when selection changes elsewhere. See B1 in the design notes.
    QCheckBox*               graph_follow_selection_ = nullptr;
    NodeGraphEditor*         graph_editor_ = nullptr;

    // ── Undo/redo ────────────────────────────────────────────────────────────
    // unique_ptr lets us forward-declare UndoStack in this header — the
    // include chain through core/undo/undo_stack.hpp pulls scene.hpp +
    // node.hpp + STL machinery that confuses Qt's MOC (results in an empty
    // moc_main_window.cpp and an undefined-vtable link error). Defining the
    // member out-of-line in the .cpp avoids that.
    std::unique_ptr<undo::UndoStack> undo_stack_;
    ::QAction*               undo_action_ = nullptr;
    ::QAction*               redo_action_ = nullptr;
    // Render-menu mode actions (exclusive group), mirroring
    // (path_menu_actions_ / layout_menu_actions_ declared above mirror the
    // Render-tab path combo + layout combo into the Render menu.)

    // ── Lights tab state ────────────────────────────────────────────────────
    ::QListWidget*           lights_list_      = nullptr;
    ::QDoubleSpinBox*        light_pos_x_      = nullptr;
    ::QDoubleSpinBox*        light_pos_y_      = nullptr;
    ::QDoubleSpinBox*        light_pos_z_      = nullptr;
    ::QDoubleSpinBox*        light_intensity_  = nullptr;
    ::QPushButton*           light_color_btn_  = nullptr;
    int                      lights_selected_  = -1;
};

} // namespace frep::gui
