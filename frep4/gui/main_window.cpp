// gui/main_window.cpp

#include "main_window.hpp"
#include <algorithm>
#include <QSplitter>

#include <chrono>
#include "core/compiler/incremental.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/transforms.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/instance.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/plugin/plugin_api.hpp"
#include "gui/node_graph.hpp"
#include "gui/scene_inspector.hpp"
#include "gui/expression_editor.hpp"
#include "gui/material_editor.hpp"
#include "gui/iviewport.hpp"
#include "gui/offscreen_viewport_adapter.hpp"
#include "gui/executor_viewport.hpp"
#include "gui/checkable_combo_box.hpp"
#include "gui/dist_viewport.hpp"
#include "core/dist/worker.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/exec/executor_factory.hpp"
#include "gui/vulkan_viewport.hpp"
#include "core/undo/undo_stack.hpp"
#include "gui/viewport.hpp"

#include <QAction>
#include <QActionGroup>
#include <QStandardItemModel>
#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QInputDialog>
#include <QFormLayout>
#include <QFuture>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QIcon>
#include <QListView>
#include <QListWidgetItem>
#include <QPixmap>
#include <QProgressDialog>
#include <QtConcurrent>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStatusBar>
#include <QTabWidget>
#include <QLineEdit>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>

#include <random>

namespace frep::gui {

MainWindow::MainWindow(SceneGraph* scene, plugin::PluginRegistry* reg,
                       bool realtime_viewport, QWidget* parent)
    : QMainWindow(parent), scene_(scene), registry_(reg),
      realtime_viewport_(realtime_viewport),
      undo_stack_(std::make_unique<undo::UndoStack>())
{
    setWindowTitle("F-Rep Designer");
    resize(1280, 800);
    build_ui();
    build_menu();
    build_toolbar();
}

// MainWindow::~MainWindow() is defined at the end of this file, after the
// LanWidgets struct is complete (it's held by unique_ptr).

void MainWindow::force_recompile_if_offscreen() {
    // Offscreen path: re-run the JIT and invalidate the viewport so
    // the next paint event re-renders. On the real-time path the
    // viewport's own scene-hash check detects the change and rebuilds
    // the GPU pipeline automatically, so this becomes a no-op.
    if (viewport_) {
        viewport_->compiler().force_recompile(*scene_);
    }
    if (viewport_iv_) {
        viewport_iv_->invalidate();
    }
}

void MainWindow::apply_render_config() {
    // Single fan-out point for Render-tab changes. The IViewport
    // implementation knows how to route the config — offscreen
    // adapter forwards into the compiler and triggers a JIT
    // recompile; real-time adapter stashes it on the renderer and
    // the per-frame scene-hash check picks it up on the next frame.
    //
    // Cheap to call back-to-back: both implementations are no-op
    // when the config is bit-equal to the previous one.
    if (viewport_iv_) viewport_iv_->set_tracer_config(render_config_);
}

void MainWindow::build_viewport(RenderMode /*mode*/) {
    // Initial bootstrap: install the backend for the current path selection.
    // After this, apply_multipath_selection() swaps backends as the selection
    // changes. Default selection is CPU_IR (set on the path combo at build).
    viewport_    = nullptr;
    viewport_iv_ = nullptr;
    executor_vp_ = nullptr;
    active_backend_ = Backend::None;
    install_backend(Backend::OffscreenCpu, { frep::exec::PathKind::CpuIr });
    current_mode_ = RenderMode::Multipath;  // single unified UI now
}

void MainWindow::install_backend(
    Backend backend, const std::vector<frep::exec::PathKind>& paths) {
    // Tear down the existing viewport (its dtor stops timers / releases GPU).
    if (viewport_iv_) {
        QWidget* old_w = viewport_iv_->widget();
        if (old_w && viewport_container_ && viewport_container_->layout())
            viewport_container_->layout()->removeWidget(old_w);
        // Delete synchronously, not deleteLater. The ExecutorViewport may have a
        // pending retry QTimer (scheduled while a path like lan was waiting for a
        // worker) that would otherwise fire on a half-torn-down object before
        // deleteLater runs. A synchronous delete runs ~ExecutorViewport now —
        // which cancels + joins its render worker and drops posted events
        // atomically — so no timer/callback can land on freed memory.
        IViewport* old = viewport_iv_;
        viewport_iv_ = nullptr;
        viewport_    = nullptr;
        executor_vp_ = nullptr;
        delete old;
    }

    // Build the requested backend. Each path's most efficient surface:
    //   RealtimeGlsl → VulkanViewport (swapchain); OffscreenCpu/Cuda → Viewport
    //   (offscreen readback); Executor → ExecutorViewport (composites N paths).
    switch (backend) {
        case Backend::RealtimeGlsl: {
            IViewport* iv = VulkanViewport::create_iv(scene_, this);
            if (iv) { viewport_iv_ = iv; break; }
            backend = Backend::Executor;  // hardware vanished → fall through
            [[fallthrough]];
        }
        case Backend::Executor:
        default: {
            auto* ev = new ExecutorViewport(scene_, this);
            executor_vp_ = ev;
            viewport_iv_ = ev;
            // New viewport has default endpoint → force a re-push on next apply.
            last_remote_port_ = -1; last_remote_workers_ = -1;
            ev->set_active_paths(paths.empty()
                                     ? std::vector{ frep::exec::PathKind::CpuIr }
                                     : paths);
            backend = Backend::Executor;
            break;
        }
        case Backend::OffscreenCpu:
        case Backend::OffscreenCuda: {
            viewport_    = new Viewport(scene_, this);
            viewport_iv_ = new OffscreenViewportAdapter(viewport_, scene_, this);
            if (backend == Backend::OffscreenCuda) {
                if (!viewport_->set_gpu_mode(true, Viewport::GpuBackend::Cuda)) {
                    viewport_->set_gpu_mode(false);   // CUDA absent → CPU
                    backend = Backend::OffscreenCpu;
                }
            } else {
                viewport_->set_gpu_mode(false);
            }
            break;
        }
    }
    active_backend_ = backend;

    // ── universal signal wiring (re-established for every install) ──────────
    connect(viewport_iv_, &IViewport::render_completed,
            this, [this](int ms) {
                on_render_completed(static_cast<double>(ms),
                                    static_cast<double>(ms),
                                    /*was_cached=*/false,
                                    /*structure_unchanged=*/false);
            });
    connect(viewport_iv_, &IViewport::object_picked,
            this, [this](const QString& id) {
                bool additive =
                    QGuiApplication::keyboardModifiers() & Qt::ControlModifier;
                if (inspector_) inspector_->select_object(id, additive);
            });

    viewport_iv_->set_tracer_config(render_config_);
    viewport_iv_->set_ssaa(current_ssaa_);
    viewport_iv_->set_camera_control_config(current_cam_cfg_);

    if (viewport_container_ && viewport_container_->layout())
        viewport_container_->layout()->addWidget(viewport_iv_->widget());
    viewport_iv_->invalidate();
}

void MainWindow::switch_render_mode(RenderMode /*mode*/) {
    // Retired: there's a single ExecutorViewport now and the path set is chosen
    // via the Render-tab combo / Render menu (apply_multipath_selection). Kept
    // as a no-op so any stray callers compile; path changes don't rebuild the
    // viewport, they just re-drive it.
}

void MainWindow::build_ui() {
    // ── viewport construction ────────────────────────────────────────────────
    //
    // The viewport backend is chosen by current_mode_ and built by
    // build_viewport() into viewport_container_. Stage 1 unified the
    // three modes (offscreen CPU, offscreen GPU, real-time) behind a
    // runtime selector; the --realtime CLI flag now just picks the
    // initial mode. build_viewport wires the universal signals
    // (render_completed, object_picked) every time it runs, so a
    // runtime switch re-establishes them for the new backend.
    //
    // Decide the initial mode: honour --realtime if hardware Vulkan is
    // present, otherwise fall back to offscreen CPU.
    if (realtime_viewport_ && vulkan_viewport_available()) {
        current_mode_ = RenderMode::Realtime;
    } else {
        if (realtime_viewport_) {
            // Requested but unavailable — leave a breadcrumb for the
            // status bar and fall back.
            realtime_viewport_ = false;
        }
        current_mode_ = RenderMode::OffscreenCPU;
    }

    viewport_container_ = new QWidget;
    {
        auto* vlay = new QVBoxLayout(viewport_container_);
        vlay->setContentsMargins(0, 0, 0, 0);
        vlay->setSpacing(0);
    }
    // NOTE: build_viewport() is called below, after inspector_ exists,
    // because it wires the viewport's object_picked signal into the
    // inspector.

    inspector_ = new SceneInspector(scene_, this);
    inspector_->set_undo_stack(undo_stack_.get());
    connect(inspector_, &SceneInspector::scene_changed,
            this, &MainWindow::on_scene_changed);
    // Keep the node graph view following the inspector's active object.
    // The new signal gives us the full QStringList of selected ids;
    // the node graph + material editor both key on the *primary*
    // (ids[0]) selection, so we adapt with a lambda.
    connect(inspector_, &SceneInspector::selection_changed,
            this, [this](const QStringList& ids) {
                on_inspector_selection_changed(ids.isEmpty() ? QString() : ids.first());
            });

    // Now that inspector_ exists, build the initial viewport. This
    // wires render_completed + object_picked for the chosen backend and
    // places its widget into viewport_container_.
    build_viewport(current_mode_);

    // Node graph editor — visual editing of the FRepNode tree for the
    // currently active scene object. See sync_graph_to_scene /
    // sync_scene_to_graph for the two-way binding.
    graph_editor_ = new NodeGraphEditor(this);
    graph_editor_->graph_scene()->set_registry(registry_);
    connect(graph_editor_->graph_scene(), &NodeGraphScene::graph_changed,
            this, &MainWindow::on_graph_changed);
    // Initial population — pick the first object (if any) as the active
    // one and load its tree into the graph.
    if (!scene_->objects().empty()) {
        active_object_id_ = scene_->objects().begin()->first;
        sync_scene_to_graph();
    }

    auto* central = new QWidget;
    auto* main_layout = new QHBoxLayout(central);
    main_layout->setContentsMargins(4, 4, 4, 4);

    // A horizontal splitter lets the user drag the divide between the viewport
    // and the settings tabs — important in Multipath mode, where the path panel
    // makes the tabs column want more width than the render needs.
    main_splitter_ = new QSplitter(Qt::Horizontal, central);
    main_layout->addWidget(main_splitter_);

    // The viewport container holds the active backend's widget;
    // switch_render_mode repopulates it without touching this layout.
    main_splitter_->addWidget(viewport_container_);

    // Tabs for inspector + tracer settings + plugin list + node graph + lights
    auto* tabs = new QTabWidget;
    tabs->setMinimumWidth(340);
    tabs->addTab(inspector_, "Scene");
    tabs->addTab(build_side_panel(), "Render");
    expression_editor_ = new ExpressionEditor(scene_, this);
    expression_editor_->set_undo_stack(undo_stack_.get());
    connect(expression_editor_, &ExpressionEditor::scene_changed, this, [this]{
        // Refresh the inspector + viewport, same as
        // SceneInspector::scene_changed does.
        inspector_->refresh();
        viewport_iv_->invalidate();
    });
    tabs->addTab(expression_editor_, "Expression");

    material_editor_ = new MaterialEditor(scene_, this);
    material_editor_->set_undo_stack(undo_stack_.get());
    // Link inspector selection → material editor's "current object".
    // The material editor now broadcasts its scalar/enum edits
    // (pattern, scale, roughness, metallic, reflectivity) to every
    // selected object; the primary (ids[0]) populates the widgets.
    // Albedo multi-edit still goes through the Inspector's colour
    // button. Feed the editor the full selection list.
    connect(inspector_, &SceneInspector::selection_changed,
            this, [this](const QStringList& ids) {
                material_editor_->on_selection_changed_multi(ids);
            });
    // Any material edit triggers a re-render.
    connect(material_editor_, &MaterialEditor::material_changed, this, [this]{
        // A material edit only changes appearance, not the object list,
        // so we just recompile. We deliberately do NOT refresh the
        // inspector here: that would rebuild the list widget and — even
        // with selection preservation — risk disturbing the in-progress
        // multi-edit. The inspector's info label is refreshed by the
        // editor's own selection handling when needed.
        force_recompile_if_offscreen();
    });
    tabs->addTab(material_editor_, "Material");

    tabs->addTab(build_lights_panel(), "Lights");
    tabs->addTab(build_plugins_panel(), "Plugins");
    tabs->addTab(build_lan_panel(), "LAN");

    // Graph tab — wrapper widget with a top toolbar (object picker +
    // fit-to-view) plus the actual editor below.
    auto* graph_panel = new QWidget(this);
    auto* graph_layout = new QVBoxLayout(graph_panel);
    graph_layout->setContentsMargins(0, 0, 0, 0);
    auto* graph_toolbar = new QHBoxLayout;
    graph_toolbar->setContentsMargins(4, 4, 4, 4);
    auto* lbl_editing = new QLabel("Editing:");
    graph_object_picker_ = new QComboBox(this);
    graph_object_picker_->setMinimumWidth(160);
    auto* btn_fit = new QPushButton("Fit", this);
    btn_fit->setToolTip("Fit all nodes into the visible area");
    // "Follow selection" — when checked (the default), clicking an
    // object in the viewport or the inspector list switches the graph
    // view to that object's tree. When unchecked, the graph stays
    // locked to its current object so the user can keep editing
    // without losing context when they click somewhere else.
    graph_follow_selection_ = new QCheckBox("Follow", this);
    graph_follow_selection_->setChecked(true);
    graph_follow_selection_->setToolTip(
        "When on, selection changes (in viewport or inspector) switch the "
        "graph view. When off, the graph stays locked to its current object.");
    graph_toolbar->addWidget(lbl_editing);
    graph_toolbar->addWidget(graph_object_picker_, 1);
    graph_toolbar->addWidget(graph_follow_selection_);
    graph_toolbar->addWidget(btn_fit);
    graph_layout->addLayout(graph_toolbar);
    graph_layout->addWidget(graph_editor_, 1);

    // When the dropdown's selection changes, propagate through the
    // existing selection pathway so the inspector + graph stay in sync.
    connect(graph_object_picker_,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){
                if (syncing_) return;
                QString id = graph_object_picker_->currentText();
                inspector_->select_object(id);
            });
    connect(btn_fit, &QPushButton::clicked, this, [this]{
        graph_editor_->fit_all();
    });

    tabs->addTab(graph_panel, "Node Graph");
    main_splitter_->addWidget(tabs);

    // Give the viewport the lion's share by default, but both panes are
    // user-resizable and the tabs pane won't collapse to nothing.
    main_splitter_->setStretchFactor(0, 1);   // viewport grows
    main_splitter_->setStretchFactor(1, 0);   // tabs keep their size
    main_splitter_->setChildrenCollapsible(false);
    main_splitter_->setSizes({ 900, 460 });

    setCentralWidget(central);
    statusBar()->showMessage("Ready");

    // Push the default TracerConfig into whichever backend is active.
    // Without this, the offscreen path would use its internal default
    // (which happens to match render_config_'s default, but only by
    // coincidence) and the real-time path would never see the config
    // until the user first moves a Render-tab slider. Apply once at
    // startup so both paths begin with a known-good state.
    apply_render_config();
}

void MainWindow::build_menu() {
    auto* file_menu = menuBar()->addMenu("&File");

    auto* open = file_menu->addAction("Open scene...");
    open->setShortcut(QKeySequence::Open);
    connect(open, &QAction::triggered, this, &MainWindow::on_open_scene);

    auto* save = file_menu->addAction("Save scene...");
    save->setShortcut(QKeySequence::SaveAs);
    connect(save, &QAction::triggered, this, &MainWindow::on_save_scene);

    file_menu->addSeparator();

    auto* import_mesh = file_menu->addAction("Import mesh (OBJ/STL) as SDF...");
    connect(import_mesh, &QAction::triggered, this, &MainWindow::on_import_mesh);

    auto* export_img = file_menu->addAction("Export rendered image...");
    connect(export_img, &QAction::triggered, this, &MainWindow::on_export_image);

    auto* export_mesh = file_menu->addAction("Export mesh (OBJ/STL)...");
    connect(export_mesh, &QAction::triggered, this, &MainWindow::on_export_mesh);

    file_menu->addSeparator();

    auto* quit = file_menu->addAction("Quit");
    connect(quit, &QAction::triggered, qApp, &QApplication::quit);

    // ── Edit menu ────────────────────────────────────────────────────────────
    auto* edit_menu = menuBar()->addMenu("&Edit");
    undo_action_ = edit_menu->addAction("Undo");
    undo_action_->setShortcut(QKeySequence::Undo);
    connect(undo_action_, &QAction::triggered, this, [this]() {
        undo_stack_->undo();
        on_scene_changed();
    });
    redo_action_ = edit_menu->addAction("Redo");
    redo_action_->setShortcut(QKeySequence::Redo);
    connect(redo_action_, &QAction::triggered, this, [this]() {
        undo_stack_->redo();
        on_scene_changed();
    });

    // Refresh enable-state and labels whenever the stack changes.
    undo_stack_->set_change_observer([this]{ refresh_undo_redo_actions(); });
    refresh_undo_redo_actions();

    edit_menu->addSeparator();

    // ── Duplicate / Copy / Paste ──────────────────────────────────────────────
    // Duplicate clones the selected objects in place (with a small
    // positional nudge so the copy is visible), selects the clones, and
    // pushes one undo entry per clone. Copy stashes the selected ids on
    // an internal clipboard; Paste clones whatever was copied. All three
    // route through clone_node (JSON round-trip) so plugin nodes work.
    auto* dup_action = edit_menu->addAction("Duplicate");
    dup_action->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_D));
    connect(dup_action, &QAction::triggered, this,
            [this]{ duplicate_selection(); });

    auto* copy_action = edit_menu->addAction("Copy");
    copy_action->setShortcut(QKeySequence::Copy);
    connect(copy_action, &QAction::triggered, this, [this]{
        clipboard_ids_.clear();
        for (const QString& id : inspector_->selected_ids())
            clipboard_ids_.push_back(id);
    });

    auto* paste_action = edit_menu->addAction("Paste");
    paste_action->setShortcut(QKeySequence::Paste);
    connect(paste_action, &QAction::triggered, this,
            [this]{ paste_clipboard(); });

    // ── Render menu — path selection mirroring the Render-tab combo ──────────
    // "Render mode" lists the four paths as independently-checkable actions
    // (tick 1+ to render); a "Multi view" submenu picks how the frame splits
    // when several are ticked. Both stay in sync with the tab's path combo.
    auto* render_menu = menuBar()->addMenu("&Render");
    auto* mode_menu   = render_menu->addMenu("Render mode");

    const auto menu_paths = picker_paths();
    for (std::size_t i = 0; i < menu_paths.size(); ++i) {
        QAction* act = mode_menu->addAction(
            frep::exec::path_kind_name(menu_paths[i]));
        act->setCheckable(true);
        act->setChecked(menu_paths[i] == frep::exec::PathKind::CpuIr);
        const int idx = (int)i;
        connect(act, &QAction::triggered, this, [this, idx](bool on) {
            // Mirror into the tab combo, then re-apply (combo is the source of
            // truth; this keeps menu + combo identical).
            if (path_combo_) path_combo_->set_checked(idx, on);
            apply_multipath_selection();
        });
        path_menu_actions_.push_back(act);
    }

    // Multi-view submenu: exclusive choice of split / weighted / tiles.
    auto* layout_menu  = mode_menu->addMenu("Multi view");
    auto* layout_group = new QActionGroup(this);
    layout_group->setExclusive(true);
    const char* layout_labels[] = { "Split", "Weighted", "Tiles" };
    for (int i = 0; i < 3; ++i) {
        QAction* act = layout_menu->addAction(layout_labels[i]);
        act->setCheckable(true);
        act->setActionGroup(layout_group);
        act->setChecked(i == 0);  // Split default
        connect(act, &QAction::triggered, this, [this, i](bool on) {
            if (on && multi_layout_combo_) multi_layout_combo_->setCurrentIndex(i);
        });
        layout_menu_actions_.push_back(act);
    }


    // ── Adaptive spatial guards toggle ──────────────────────────────────────
    // On by default (the calibration is conservative). Lets the user turn
    // off the build-time AABB prune if desired. Only affects the offscreen
    // CPU backend; other backends no-op the setter.
    render_menu->addSeparator();
    auto* guards_act = render_menu->addAction("Adaptive spatial guards");
    guards_act->setCheckable(true);
    guards_act->setChecked(true);   // matches IncrementalCompiler default
    guards_act->setToolTip(
        "Skip distant objects via a calibrated AABB test during ray-march. "
        "Speeds up scenes with many expensive (CSG/deformation) objects; "
        "no effect on simple primitives. CPU render only.");
    connect(guards_act, &QAction::triggered, this, [this](bool on) {
        if (viewport_iv_) viewport_iv_->set_spatial_guards_enabled(on);
    });

    auto* help_menu = menuBar()->addMenu("&Help");
    auto* about = help_menu->addAction("About");
    connect(about, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, "F-Rep Designer 4.0",
            "F-Rep Designer 4.0 — PoC geometric modeling\n"
            "based on F-Rep + LLVM JIT compilation.\n\n"
            "(c) 2026 University of Plovdiv\n"
            "Project FP17-FMI-008");
    });
}

void MainWindow::on_open_scene() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open scene", QString(), "F-Rep scene (*.frep *.json)");
    if (path.isEmpty()) return;
    try {
        // Pass the plugin registry so plugin-based primitives (Torus,
        // Octahedron, Capsule, ...) deserialize correctly.
        *scene_ = io::load_scene(path.toStdString(), registry_);
        // The loaded scene starts a fresh edit history — old undo entries
        // would refer to objects that no longer exist.
        undo_stack_->clear();
        inspector_->refresh();
        refresh_lights_panel();
        on_scene_changed();
        statusBar()->showMessage("Loaded: " + path);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load error", e.what());
    }
}

void MainWindow::on_save_scene() {
    QString path = QFileDialog::getSaveFileName(
        this, "Save scene", "scene.frep", "F-Rep scene (*.frep *.json)");
    if (path.isEmpty()) return;
    if (io::save_scene(*scene_, path.toStdString())) {
        statusBar()->showMessage("Saved: " + path);
    } else {
        QMessageBox::critical(this, "Error", "Failed to write the file.");
    }
}

void MainWindow::on_export_image() {
    // Unified capture across all three backends. The offscreen paths
    // hand back their last CPU-side readback; the real-time path does a
    // true GPU framebuffer grab (QVulkanWindow::grab → vkCmdCopyImage),
    // which is faithful to what the GPU rendered and immune to
    // overlapping windows / compositor effects — unlike the old
    // QScreen::grabWindow screen-coordinate capture.
    QImage img;
    if (viewport_iv_) img = viewport_iv_->capture_image();
    if (img.isNull()) {
        QMessageBox::information(this, "Nothing to export",
            "The viewport has not produced a rendered image yet.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export rendered image", "render.png",
        "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp);;"
        "PPM image (*.ppm)");
    if (path.isEmpty()) return;

    // Qt picks the format from the file extension. PPM is supported by the
    // built-in image writer (handy for round-tripping with the CLI tools,
    // which also write PPM).
    if (img.save(path)) {
        statusBar()->showMessage(QString("Exported: %1 (%2x%3)")
            .arg(path).arg(img.width()).arg(img.height()));
    } else {
        QMessageBox::critical(this, "Export error",
            "Failed to write the image. Check the file extension matches "
            "a supported format (PNG, JPG, BMP, PPM).");
    }
}

void MainWindow::on_import_mesh() {
    QString path = QFileDialog::getOpenFileName(
        this, "Import mesh as SDF", QString(),
        "Mesh files (*.obj *.stl);;Wavefront OBJ (*.obj);;STL (*.stl)");
    if (path.isEmpty()) return;

    // Load.
    mesh::Mesh m;
    if (path.endsWith(".stl", Qt::CaseInsensitive))
        m = mesh::load_stl(path.toStdString());
    else
        m = mesh::load_obj(path.toStdString());

    if (m.vertices.empty() || m.indices.size() < 3) {
        QMessageBox::warning(this, "Import error",
            "The file could not be loaded or contained no triangles.");
        return;
    }

    // Ask for resolution — controls accuracy vs. voxelization time.
    bool ok = false;
    int res = QInputDialog::getInt(this, "Voxelization resolution",
        QString("Loaded %1 vertices, %2 triangles.\n"
                "Pick the voxel grid resolution (32-256). Higher = sharper "
                "but slower to import and more RAM.")
            .arg(m.vertices.size())
            .arg(m.indices.size() / 3),
        48, 8, 256, 8, &ok);
    if (!ok) return;

    // Optional sparse-octree compression. Saves RAM at the cost of some
    // surface accuracy; 0 disables.
    double tol = QInputDialog::getDouble(this, "Sparse compression",
        "Sparse-octree tolerance (0 = dense, 0.05–0.15 = good trade-off,\n"
        "0.2+ = aggressive). RAM savings grow with tolerance × resolution.",
        0.0, 0.0, 1.0, 3, &ok);
    if (!ok) return;

    // Voxelize asynchronously so the GUI stays responsive. We show
    // an indeterminate progress dialog (the voxelization itself doesn't
    // expose progress events — it's a one-shot BVH traversal — but we
    // can at least keep the window painting and let the user cancel by
    // closing the dialog, which simply hides it; the work continues
    // and the result is discarded if `cancelled` was set).
    QProgressDialog progress(
        QString("Voxelizing %1 at %2³ ...").arg(QFileInfo(path).fileName()).arg(res),
        "Cancel (run in background)", 0, 0, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);  // show immediately
    progress.show();

    auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<bool> cancelled = std::make_shared<bool>(false);
    connect(&progress, &QProgressDialog::canceled, this, [cancelled]{
        *cancelled = true;
    });

    // QtConcurrent::run lets us push the heavy work to a worker thread;
    // QFutureWatcher delivers the result back on the GUI thread.
    auto future = QtConcurrent::run([m, res, tol]() {
        std::string id_unused = "tmp";
        return std::make_shared<MeshSDFNode>(
            m, res, id_unused, static_cast<float>(tol));
    });

    // Pump the event loop until the future completes. This keeps the
    // viewport repainting and lets QProgressDialog update its UI. We
    // also refresh the dialog label every 200 ms to show elapsed time,
    // since the underlying voxelization can't report progress —
    // showing a live counter at least confirms the GUI isn't stuck.
    QString base_label = progress.labelText();
    auto last_tick = t0;
    while (!future.isFinished()) {
        QApplication::processEvents(QEventLoop::AllEvents, 30);
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_tick).count() > 200) {
            double el = std::chrono::duration<double>(now - t0).count();
            progress.setLabelText(base_label +
                QString("\n(elapsed: %1 s)").arg(el, 0, 'f', 1));
            last_tick = now;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    progress.close();

    if (*cancelled) {
        statusBar()->showMessage("Mesh import cancelled.");
        return;
    }

    auto temp_node = future.result();
    // Re-id with the real scene-local name.
    std::string id = "mesh_" + std::to_string(scene_->objects().size());
    auto node = std::make_shared<MeshSDFNode>(*temp_node);
    node->id = id;

    // Add to scene as a normal object — colour by hash of file name so
    // repeated imports of the same file get the same colour.
    std::mt19937 rng(std::hash<std::string>{}(path.toStdString()));
    std::uniform_real_distribution<float> dist(0.3f, 0.9f);
    Material mat{{dist(rng), dist(rng), dist(rng)}};

    undo_stack_->push_apply(std::make_unique<undo::AddObjectCommand>(
        *scene_, id, node, mat));
    inspector_->refresh();
    on_scene_changed();

    QString sparse_info;
    if (node->uses_sparse()) {
        sparse_info = QString(" → sparse %1 KB (%2x)")
            .arg(node->sparse_bytes() / 1024)
            .arg(node->sparse_ratio(), 0, 'f', 2);
    }
    statusBar()->showMessage(
        QString("Imported %1: %2 verts, %3 tris → %4^3 SDF (%5 KB)%6 in %7 ms")
            .arg(path)
            .arg(m.vertices.size())
            .arg(m.indices.size() / 3)
            .arg(res)
            .arg(node->grid_bytes() / 1024)
            .arg(sparse_info)
            .arg(ms, 0, 'f', 1));
}

void MainWindow::on_export_mesh() {
    if (scene_->objects().empty()) {
        QMessageBox::information(this, "Nothing to export",
            "The scene has no objects to mesh.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export mesh", "mesh.obj",
        "Wavefront OBJ (*.obj);;ASCII STL (*.stl)");
    if (path.isEmpty()) return;

    // Ask user for sampling resolution. Marching cubes cost scales as
    // O(N³), so 32³≈30k samples is instant, 128³≈2M takes a few sec.
    // Higher resolutions catch fine surface detail but bloat the file.
    bool ok_res = false;
    int res = QInputDialog::getInt(
        this, "Mesh resolution",
        "Sampling resolution per axis (32–256). Higher = finer mesh, slower:",
        64, 32, 256, 16, &ok_res);
    if (!ok_res) return;

    mesh::MarchingCubesParams params;
    params.rx = params.ry = params.rz = res;

    // Run extraction off-thread so the GUI stays responsive. For a
    // 128³ extraction on a typical scene this is ~2-5 seconds; without
    // async the window would freeze and the OS would mark it Not
    // Responding. QtConcurrent::run posts the work to a thread pool
    // and we pump events until QFuture::isFinished() — same pattern as
    // on_import_mesh.
    QProgressDialog progress(
        QString("Extracting mesh at %1³ (%2 cells) ...")
            .arg(res).arg(res * res * res),
        "Cancel (continue in background)", 0, 0, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.show();

    std::shared_ptr<bool> cancelled = std::make_shared<bool>(false);
    connect(&progress, &QProgressDialog::canceled, this,
            [cancelled]{ *cancelled = true; });

    auto t0 = std::chrono::steady_clock::now();
    SceneGraph scene_copy = *scene_;  // snapshot
    auto future = QtConcurrent::run([scene_copy, params]() {
        return mesh::extract_iso_mesh(scene_copy, params);
    });
    while (!future.isFinished()) {
        QApplication::processEvents(QEventLoop::AllEvents, 30);
    }
    auto t1 = std::chrono::steady_clock::now();
    progress.close();

    if (*cancelled) {
        statusBar()->showMessage("Mesh export cancelled.");
        return;
    }

    auto m = future.result();
    if (m.vertices.empty()) {
        QMessageBox::warning(this, "Empty mesh",
            "Marching cubes produced no triangles — the surface does not "
            "intersect the sampling region.");
        statusBar()->clearMessage();
        return;
    }

    bool wrote = false;
    if (path.endsWith(".stl", Qt::CaseInsensitive)) {
        wrote = mesh::save_stl(m, path.toStdString());
    } else {
        // Default to OBJ for any unknown extension.
        wrote = mesh::save_obj(m, path.toStdString());
    }

    if (wrote) {
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        statusBar()->showMessage(
            QString("Exported %1: %2 verts, %3 tris in %4 ms")
                .arg(QFileInfo(path).fileName())
                .arg(m.vertices.size())
                .arg(m.indices.size() / 3)
                .arg(static_cast<int>(ms)));
    } else {
        QMessageBox::critical(this, "Export error",
            "Failed to write the mesh file.");
    }
}

std::vector<frep::exec::PathKind> MainWindow::picker_paths() {
    auto v = frep::exec::local_paths();        // cpu_ir, gpu_glsl, gpu_ir, gpu_rtx
    v.push_back(frep::exec::PathKind::Lan);     // + the LAN orchestration path
    return v;
}

MainWindow::Backend MainWindow::desired_backend_for(
    const std::vector<frep::exec::PathKind>& paths) const {
    using PK = frep::exec::PathKind;
    // "lan" is now a normal RemoteExecutor path: it composites through the
    // ExecutorViewport like any other, alone or alongside local paths. So any
    // selection containing lan (or more than one path) uses the Executor
    // backend. A single local path still gets its dedicated efficient backend.
    for (auto p : paths) if (p == PK::Lan) return Backend::Executor;
    if (paths.size() != 1) return Backend::Executor;
    switch (paths[0]) {
        case PK::GpuGlsl:
            return vulkan_viewport_available() ? Backend::RealtimeGlsl
                                               : Backend::Executor;
        case PK::CpuIr:  return Backend::OffscreenCpu;
        case PK::GpuIr:  return Backend::OffscreenCuda;
        case PK::GpuRtx: return Backend::Executor;
        default:         return Backend::Executor;
    }
}

void MainWindow::apply_multipath_selection() {
    if (!path_combo_) return;
    // Read the checked paths (default to CPU_IR if the user unticked all).
    const auto paths = picker_paths();
    std::vector<frep::exec::PathKind> active;
    for (int idx : path_combo_->checked_indices())
        if (idx >= 0 && idx < (int)paths.size()) active.push_back(paths[idx]);
    if (active.empty()) active.push_back(frep::exec::PathKind::CpuIr);

    Backend want = desired_backend_for(active);

    // If the lan path is active, push the LAN tab's endpoint into whichever
    // ExecutorViewport will hold it, so its RemoteExecutor binds the right port.
    bool has_lan = false;
    for (auto p : active) if (p == frep::exec::PathKind::Lan) has_lan = true;

    // For executor backends, lan composites like any other path — no filtering.
    if (want != active_backend_) {
        install_backend(want, active);
    } else if (executor_vp_) {
        executor_vp_->set_active_paths(active);
    }
    if (has_lan && executor_vp_ && lan_) {
        // Push the endpoint only when it actually changed, so we don't tear down
        // and re-accept the master (a new bind + accept) on every unrelated
        // re-apply. set_remote_config rebuilds the RemoteExecutor; doing it every
        // time would also drop a connected worker.
        int port = lan_->port->value(), nw = lan_->workers->value();
        if (port != last_remote_port_ || nw != last_remote_workers_) {
            executor_vp_->set_remote_config(port, nw);
            last_remote_port_ = port; last_remote_workers_ = nw;
        }
        // The lan path owns its own master (auto-started on the LAN tab's port,
        // waiting for workers). Disable the standalone "Start master" button so
        // the two can't fight over the port, and point the user at the worker
        // controls. The worker buttons stay enabled — that's how you attach
        // workers to the lan path's master (locally or from another machine).
        lan_->start_master->setEnabled(false);
        lan_->stop_master->setEnabled(false);
        lan_->master_status->setText(
            "Master auto-started by the 'lan' path on port " +
            QString::number(port) +
            " — start " + QString::number(nw) +
            " worker(s) to render.");
    } else if (lan_ && active_backend_ != Backend::None) {
        // Leaving the lan path → re-enable the standalone master controls.
        lan_->start_master->setEnabled(true);
        lan_->stop_master->setEnabled(false);
        if (lan_->master_status->text().startsWith("Master auto-started"))
            lan_->master_status->setText("idle");
    }

    // Layout applies only to the compositing executor view.
    MultiLayout layout = MultiLayout::Strips;
    if (multi_layout_combo_) {
        switch (multi_layout_combo_->currentData().toInt()) {
            case 1: layout = MultiLayout::WeightedStrips; break;
            case 2: layout = MultiLayout::Tiles; break;
            default: layout = MultiLayout::Strips; break;
        }
    }
    if (executor_vp_) executor_vp_->set_layout(layout);
    sync_path_menu_to_combo();
}

void MainWindow::sync_path_menu_to_combo() {
    const auto paths = picker_paths();
    // Keep the Render-menu path actions in lock-step with the combo.
    for (std::size_t i = 0; i < path_menu_actions_.size() &&
                            i < paths.size(); ++i) {
        QSignalBlocker b(path_menu_actions_[i]);
        path_menu_actions_[i]->setChecked(path_combo_->is_checked((int)i));
    }
    // And the layout submenu with the layout combo.
    if (multi_layout_combo_) {
        int li = multi_layout_combo_->currentIndex();
        for (std::size_t i = 0; i < layout_menu_actions_.size(); ++i) {
            QSignalBlocker b(layout_menu_actions_[i]);
            layout_menu_actions_[i]->setChecked((int)i == li);
        }
    }
}

void MainWindow::build_toolbar() {
    auto* toolbar = addToolBar("Primitives");

    // Built-in primitives — buttons for quick adding
    auto add_btn = [&](const QString& label, auto factory) {
        auto* a = toolbar->addAction(label);
        connect(a, &QAction::triggered, this, [this, factory]() {
            static int counter = 0;
            std::string id = "obj_" + std::to_string(counter++);
            auto node = factory(id);
            // Random color
            std::mt19937 rng(counter);
            std::uniform_real_distribution<float> dist(0.2f, 0.95f);
            undo_stack_->push_apply(
                std::make_unique<undo::AddObjectCommand>(
                    *scene_, id, node,
                    Material{{dist(rng), dist(rng), dist(rng)}}));
            inspector_->refresh();
            on_scene_changed();
        });
    };

    add_btn("Sphere", [](std::string id) {
        return std::make_shared<SphereNode>(0.8f, id);
    });
    add_btn("Box", [](std::string id) {
        return std::make_shared<BoxNode>(0.7f, 0.7f, 0.7f, id);
    });
    add_btn("Plane", [](std::string id) {
        return std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 1.5f, id);
    });

    toolbar->addSeparator();

    // Plugin-based primitives — combobox + button
    auto* cb_plugin = new QComboBox;
    for (const auto& p : registry_->primitives()) {
        cb_plugin->addItem(QString::fromStdString(std::string(p.info.name)));
    }
    toolbar->addWidget(new QLabel(" Plugin:"));
    toolbar->addWidget(cb_plugin);

    auto* add_plugin = toolbar->addAction("+ Add plugin");
    connect(add_plugin, &QAction::triggered, this, [this, cb_plugin]() {
        QString name = cb_plugin->currentText();
        on_add_primitive(name);
    });

    toolbar->addSeparator();

    // ── Instance the selected object (task 4, true instancing) ───────────────
    // Creates an InstanceNode referencing the selected object's geometry by id
    // and *sharing* its pointer (not a copy): editing the target later shows
    // live in every instance, and the emitted code will (Level 2) reuse one
    // subprogram. The instance gets its own Translate placement so it is visible;
    // its transform is independent of the original's.
    auto* act_instance = toolbar->addAction("Instance");
    act_instance->setToolTip("Reference the selected object as a live instance "
                             "(shares geometry; edits to the original propagate)");
    connect(act_instance, &QAction::triggered, this, [this]() {
        if (!inspector_) return;
        const QStringList sel = inspector_->selected_ids();
        if (sel.isEmpty()) {
            QMessageBox::information(this, "Instance",
                "Select an object in the Scene list to instance.");
            return;
        }
        static int inst_counter = 0;
        for (const QString& sid : sel) {
            const std::string src = sid.toStdString();
            const SceneObject* obj = scene_->find_object(src);
            if (!obj || !obj->geometry) continue;
            std::string nid = src + "_inst" + std::to_string(inst_counter++);
            // Reference the target's *bare* geometry (semantics ii): the instance
            // node shares the target root pointer; its own Translate places it.
            auto inst = std::make_shared<InstanceNode>(obj->geometry, src, nid);
            auto placed = std::make_shared<TranslateNode>(inst, 0.6f, 0.0f, 0.0f, nid + "_t");
            undo_stack_->push_apply(std::make_unique<undo::AddObjectCommand>(
                *scene_, nid, placed, obj->material));
        }
        io::resolve_instances(*scene_, registry_);
        inspector_->refresh();
        on_scene_changed();
    });

    toolbar->addSeparator();
    toolbar->addWidget(new QLabel(" Deform:"));

    // ── Apply a deformation to the selected object(s) (task 7) ───────────────
    // Deformations wrap existing geometry (unlike primitives, which add new
    // objects). Each button replaces the primary selection's geometry with the
    // deformation node wrapping the current root, via SetGeometryCommand so it
    // is undoable. Twist/Bend around Y, Taper along Y — the node kinds already
    // exist; this is the only place they can be applied from the GUI.
    auto apply_deform = [this](auto make_node, const char* label) {
        if (!inspector_) return;
        const QStringList sel = inspector_->selected_ids();
        if (sel.isEmpty()) {
            QMessageBox::information(this, "Deform",
                QString("Select an object first, then apply %1.").arg(label));
            return;
        }
        for (const QString& sid : sel) {
            const std::string id = sid.toStdString();
            const SceneObject* obj = scene_->find_object(id);
            if (!obj || !obj->geometry) continue;
            FRepNode::Ptr wrapped = make_node(obj->geometry, id);
            undo_stack_->push_apply(
                std::make_unique<undo::SetGeometryCommand>(*scene_, id, wrapped));
        }
        inspector_->refresh();
        on_scene_changed();
    };

    auto* act_twist = toolbar->addAction("Twist");
    act_twist->setToolTip("Wrap the selected object in a TwistY deformation");
    connect(act_twist, &QAction::triggered, this, [this, apply_deform]() {
        apply_deform([](FRepNode::Ptr g, std::string id) -> FRepNode::Ptr {
            return std::make_shared<TwistYNode>(g, 1.5f, id + "_tw");
        }, "Twist");
    });
    auto* act_bend = toolbar->addAction("Bend");
    act_bend->setToolTip("Wrap the selected object in a BendXY deformation");
    connect(act_bend, &QAction::triggered, this, [this, apply_deform]() {
        apply_deform([](FRepNode::Ptr g, std::string id) -> FRepNode::Ptr {
            return std::make_shared<BendXYNode>(g, 0.9f, id + "_bend");
        }, "Bend");
    });
    auto* act_taper = toolbar->addAction("Taper");
    act_taper->setToolTip("Wrap the selected object in a TaperY deformation");
    connect(act_taper, &QAction::triggered, this, [this, apply_deform]() {
        apply_deform([](FRepNode::Ptr g, std::string id) -> FRepNode::Ptr {
            return std::make_shared<TaperYNode>(g, 0.4f, 2.0f, id + "_tp");
        }, "Taper");
    });
}

QWidget* MainWindow::build_side_panel() {
    auto* panel = new QWidget;
    auto* v = new QVBoxLayout(panel);

    // ── Tracer settings ──────────────────────────────────────────────────────
    // Both the offscreen viewport (CPU JIT + offscreen GPU) and the
    // real-time viewport honour these settings identically. Edits go
    // through MainWindow::render_config_ → apply_render_config(),
    // which forwards to whatever backend is active via the
    // IViewport::set_tracer_config() abstraction.
    auto* tracer_box = new QGroupBox("Ray tracer");
    auto* tf = new QFormLayout(tracer_box);

    // ── Render mode selector (Stage 1) ──────────────────────────────────────
    // Unified runtime choice between the three backends. Mirrors the
    // Render-menu actions. Real-time is disabled when hardware Vulkan is
    // unavailable. Switching tears down the previous backend completely
    // (switch_render_mode) so nothing keeps rendering under the surface.
    // Path picker: a checkable-dropdown combo (one box, tick 1+ paths). One
    // ticked path renders full-frame; several split the frame by the layout
    // chosen in the adjacent combo. All four paths are driven by the single
    // ExecutorViewport, so there are no separate "modes" — just a path set.
    path_combo_ = new CheckableComboBox;
    const auto paths = picker_paths();
    const bool rt_ok = vulkan_viewport_available();
    for (std::size_t i = 0; i < paths.size(); ++i) {
        // Default selection: CPU_IR ticked (always available, no GPU needed).
        bool on = (paths[i] == frep::exec::PathKind::CpuIr);
        path_combo_->add_checkable_item(frep::exec::path_kind_name(paths[i]),
                                        (int)i, on);
        // GPU paths need a device; leave them tickable but note the dependency
        // in the tooltip — they degrade gracefully (per-path error) if absent.
    }
    (void)rt_ok;
    path_combo_->on_changed = [this] { apply_multipath_selection(); };
    tf->addRow("Paths:", path_combo_);

    // Multi-view layout: how the frame is divided when 2+ paths are ticked.
    multi_layout_combo_ = new QComboBox;
    multi_layout_combo_->addItem("split",    0);   // equal vertical strips
    multi_layout_combo_->addItem("weighted", 1);   // strips sized by throughput
    multi_layout_combo_->addItem("tiles",    2);   // interleaved tiles
    connect(multi_layout_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this] { apply_multipath_selection(); });
    tf->addRow("Multi view:", multi_layout_combo_);

    auto* cb_shadows = new QCheckBox;
    cb_shadows->setChecked(render_config_.enable_shadows);
    connect(cb_shadows, &QCheckBox::toggled, this, [this](bool on) {
        render_config_.enable_shadows = on;
        apply_render_config();
    });
    tf->addRow("Shadows:", cb_shadows);

    auto* cb_ao = new QCheckBox;
    cb_ao->setChecked(render_config_.enable_ao);
    connect(cb_ao, &QCheckBox::toggled, this, [this](bool on) {
        render_config_.enable_ao = on;
        apply_render_config();
    });
    tf->addRow("AO:", cb_ao);

    auto* sl_soft = new QSlider(Qt::Horizontal);
    sl_soft->setRange(1, 64);
    sl_soft->setValue(static_cast<int>(render_config_.shadow_softness));
    connect(sl_soft, &QSlider::valueChanged, this, [this](int val) {
        render_config_.shadow_softness = float(val);
        apply_render_config();
    });
    tf->addRow("Soft shadows:", sl_soft);

    auto* sl_ao = new QSlider(Qt::Horizontal);
    sl_ao->setRange(0, 100);
    sl_ao->setValue(static_cast<int>(render_config_.ao_strength * 100.0f));
    connect(sl_ao, &QSlider::valueChanged, this, [this](int val) {
        render_config_.ao_strength = val / 100.0f;
        apply_render_config();
    });
    tf->addRow("AO strength:", sl_ao);

    // SSAA — super-sampling anti-aliasing. Now works on both the
    // offscreen and the real-time paths. For real-time, the
    // ComputeBlitRenderer dispatches the compute shader at the
    // SSAA-scaled resolution into a larger storage image, then
    // vkCmdBlitImage downsamples to the swapchain with linear
    // filtering — a single-pass, hardware-implemented box filter.
    auto* cb_ssaa = new QComboBox;
    cb_ssaa->addItem("Off (1x)",      1);
    cb_ssaa->addItem("2x2 (4 rays)",  2);
    cb_ssaa->addItem("4x4 (16 rays)", 4);
    cb_ssaa->setCurrentIndex(0);
    connect(cb_ssaa, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_ssaa](int) {
        current_ssaa_ = cb_ssaa->currentData().toInt();
        if (viewport_iv_)
            viewport_iv_->set_ssaa(current_ssaa_);
    });
    tf->addRow("SSAA:", cb_ssaa);

    // Mouse sensitivity — orbit radians per pixel of drag. The one
    // camera-control constant that genuinely varies between input
    // devices (a trackpad usually wants a higher value than a mouse),
    // so it's surfaced here; the rest live in CameraControlConfig with
    // sensible defaults. Applied live to the active viewport and
    // preserved across render-mode switches.
    auto* sp_sens = new QDoubleSpinBox;
    sp_sens->setRange(0.001, 0.100);
    sp_sens->setDecimals(3);
    sp_sens->setSingleStep(0.001);
    sp_sens->setValue(current_cam_cfg_.mouse_sensitivity);
    connect(sp_sens, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v) {
        current_cam_cfg_.mouse_sensitivity = static_cast<float>(v);
        if (viewport_iv_)
            viewport_iv_->set_camera_control_config(current_cam_cfg_);
    });
    tf->addRow("Mouse sensitivity:", sp_sens);

    // Shading model — Blinn-Phong (fast) or Cook-Torrance (PBR).
    // Both backends emit code for both models (codegen.cpp for CPU
    // JIT / GPU offscreen, glsl_emitter.cpp for real-time).
    auto* cb_model = new QComboBox;
    cb_model->addItem("Cook-Torrance (PBR)",
        static_cast<int>(TracerConfig::ShadingModel::CookTorrance));
    cb_model->addItem("Blinn-Phong",
        static_cast<int>(TracerConfig::ShadingModel::BlinnPhong));
    cb_model->setCurrentIndex(
        render_config_.shading_model == TracerConfig::ShadingModel::BlinnPhong
            ? 1 : 0);
    connect(cb_model, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_model](int) {
        render_config_.shading_model =
            static_cast<TracerConfig::ShadingModel>(cb_model->currentData().toInt());
        apply_render_config();
    });
    tf->addRow("Shading:", cb_model);

    // ── Sky colour pickers ─────────────────────────────────────────────────
    // Two-stop gradient — horizon (looking flat) and zenith (looking
    // straight up). Both CPU and GPU paths bake the chosen colours
    // into the emitted code, so the scene hash drifts on edit and
    // triggers a rebuild. Live but ~0.5s settle on real-time.
    auto colour_button = [this](const char* label, float* rgb) {
        auto* btn = new QPushButton;
        btn->setFixedSize(48, 22);
        auto set_swatch = [btn, rgb]() {
            QColor c = QColor::fromRgbF(rgb[0], rgb[1], rgb[2]);
            btn->setStyleSheet(QString(
                "background-color: %1; border: 1px solid #888;").arg(c.name()));
        };
        set_swatch();
        connect(btn, &QPushButton::clicked, this,
                [this, rgb, set_swatch, label]() {
            QColor initial = QColor::fromRgbF(rgb[0], rgb[1], rgb[2]);
            QColor c = QColorDialog::getColor(initial, this, label);
            if (!c.isValid()) return;
            rgb[0] = static_cast<float>(c.redF());
            rgb[1] = static_cast<float>(c.greenF());
            rgb[2] = static_cast<float>(c.blueF());
            set_swatch();
            apply_render_config();
        });
        return btn;
    };
    tf->addRow("Sky (horizon):",
               colour_button("Sky horizon", render_config_.sky_horizon));
    tf->addRow("Sky (top):",
               colour_button("Sky top",     render_config_.sky_top));

    // ── Reflections ─────────────────────────────────────────────────────────
    // Max bounce depth. 0 disables reflections entirely (no secondary
    // rays, per-material reflectivity ignored). Each bounce ~doubles
    // trace cost, so cap at 4. Both CPU JIT and GPU paths honour it;
    // changing it forces a shader/JIT rebuild (it's in the scene hash).
    auto* sp_bounce = new QSpinBox;
    sp_bounce->setRange(0, 4);
    sp_bounce->setValue(render_config_.max_bounces);
    connect(sp_bounce, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int val) {
        render_config_.max_bounces = val;
        apply_render_config();
    });
    tf->addRow("Reflections (bounces):", sp_bounce);

    // ── Soft shadow quality ───────────────────────────────────────────────────
    // Samples per light. 1 = hard single-ray shadow (cheapest). Higher
    // casts N jittered rays toward a virtual area light of half-extent
    // `shadow light radius`, averaging for softer penumbrae. Cap 16.
    auto* sp_ssamp = new QSpinBox;
    sp_ssamp->setRange(1, 16);
    sp_ssamp->setValue(render_config_.shadow_samples);
    connect(sp_ssamp, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int val) {
        render_config_.shadow_samples = val;
        apply_render_config();
    });
    tf->addRow("Shadow samples:", sp_ssamp);

    auto* sl_lrad = new QSlider(Qt::Horizontal);
    sl_lrad->setRange(0, 100);   // maps to 0.0 .. 1.0 world units
    sl_lrad->setValue(static_cast<int>(render_config_.shadow_light_radius * 100.0f));
    connect(sl_lrad, &QSlider::valueChanged, this, [this](int val) {
        render_config_.shadow_light_radius = val / 100.0f;
        apply_render_config();
    });
    tf->addRow("Light radius:", sl_lrad);

    // ── Denoising (real-time temporal accumulation) ────────────────────────────
    // Frames to accumulate while the camera is static. 1 = off. Higher
    // values converge the jittered soft-shadow / AO noise over several
    // still frames. Real-time path only — the offscreen paths already
    // render a single clean frame, so the control has no effect there.
    auto* sp_accum = new QSpinBox;
    sp_accum->setRange(1, 64);
    sp_accum->setValue(render_config_.accum_frames);
    connect(sp_accum, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int val) {
        render_config_.accum_frames = val;
        apply_render_config();
    });
    tf->addRow("Denoise (accum frames):", sp_accum);

    // ── Compile mode (Constant / Incremental / Auto) ────────────────────────
    // Constant: full O3 constant-folding; every parameter change triggers
    //   a JIT recompile.
    // Incremental: parameters loaded from a runtime buffer; slider edits
    //   re-use the cached JIT'd function. Per-render is ~1.5x slower but
    //   total cycle is much faster when editing.
    // Auto: starts Constant, switches to Incremental after 3 recompiles
    //   within 5 seconds.
    auto* cb_compile = new QComboBox;
    cb_compile->addItem("Constant (full O3)",
        static_cast<int>(TracerConfig::CompileMode::Constant));
    cb_compile->addItem("Incremental (runtime params)",
        static_cast<int>(TracerConfig::CompileMode::Incremental));
    cb_compile->addItem("Auto (switch on demand)",
        static_cast<int>(TracerConfig::CompileMode::Auto));
    cb_compile->setCurrentIndex(2);  // Auto by default
    if (viewport_) {
        viewport_->compiler().policy().set_mode(TracerConfig::CompileMode::Auto);
    }
    connect(cb_compile, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_compile](int) {
        if (!viewport_) return;  // tracer_box is disabled anyway
        auto mode = static_cast<TracerConfig::CompileMode>(
                        cb_compile->currentData().toInt());
        viewport_->compiler().policy().set_mode(mode);
        force_recompile_if_offscreen();
    });
    tf->addRow("Compile:", cb_compile);

    v->addWidget(tracer_box);

    // ── Camera ───────────────────────────────────────────────────────────────
    auto* cam_box = new QGroupBox("Camera");
    auto* cf = new QFormLayout(cam_box);

    auto* cb_proj = new QComboBox;
    cb_proj->addItem("Perspective",  static_cast<int>(Camera::Projection::Perspective));
    cb_proj->addItem("Orthographic", static_cast<int>(Camera::Projection::Orthographic));
    cb_proj->setCurrentIndex(0);
    connect(cb_proj, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_proj](int) {
        scene_->camera().projection =
            static_cast<Camera::Projection>(cb_proj->currentData().toInt());
        viewport_iv_->invalidate();
    });
    cf->addRow("Projection:", cb_proj);

    auto* sb_fov = new QDoubleSpinBox;
    sb_fov->setRange(10.0, 120.0);
    sb_fov->setValue(scene_->camera().fov_deg);
    sb_fov->setSuffix("°");
    connect(sb_fov, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v) {
        scene_->camera().fov_deg = static_cast<float>(v);
        viewport_iv_->invalidate();
    });
    cf->addRow("FoV (perspective):", sb_fov);

    auto* sb_ortho = new QDoubleSpinBox;
    sb_ortho->setRange(0.5, 50.0);
    sb_ortho->setValue(scene_->camera().ortho_size);
    sb_ortho->setSingleStep(0.5);
    connect(sb_ortho, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v) {
        scene_->camera().ortho_size = static_cast<float>(v);
        viewport_iv_->invalidate();
    });
    cf->addRow("Size (ortho):", sb_ortho);

    // Camera presets — common viewpoints (front / top / right / iso).
    auto* presets_row = new QHBoxLayout;
    auto add_preset = [&](const QString& label,
                          std::array<float,3> pos,
                          std::array<float,3> up = {0, 1, 0}) {
        auto* btn = new QPushButton(label);
        connect(btn, &QPushButton::clicked, this, [this, pos, up]() {
            scene_->camera().position = pos;
            scene_->camera().target   = {0, 0, 0};
            scene_->camera().up       = up;
            viewport_iv_->invalidate();
        });
        presets_row->addWidget(btn);
    };
    add_preset("Front", { 0.0f,  0.0f,  6.0f});
    add_preset("Top",   { 0.0f,  6.0f,  0.01f}, {0, 0, -1});
    add_preset("Right", { 6.0f,  0.0f,  0.0f});
    add_preset("Iso",   { 4.0f,  3.0f,  4.0f});
    cf->addRow("Presets:", presets_row);

    v->addWidget(cam_box);

    // ── Registered plugins ───────────────────────────────────────────────────
    // ── GPU tile cull (task 1) ───────────────────────────────────────────────
    // Live control of the depth-slab tile cull used by the real-time GPU path:
    // method (Auto/Lipschitz/Interval/Off), slab count and the Lipschitz L. All
    // feed render_config_ and apply_render_config() like the other settings, so
    // the effect is visible immediately while orbiting the scene.
    auto* cull_box = new QGroupBox("GPU tile cull");
    auto* cull_form = new QFormLayout(cull_box);

    auto* cb_cull_method = new QComboBox;
    cb_cull_method->addItem("Auto",      static_cast<int>(TracerConfig::CullMethod::Auto));
    cb_cull_method->addItem("Lipschitz", static_cast<int>(TracerConfig::CullMethod::Lipschitz));
    cb_cull_method->addItem("Interval",  static_cast<int>(TracerConfig::CullMethod::Interval));
    cb_cull_method->addItem("Off",       static_cast<int>(TracerConfig::CullMethod::Off));
    cb_cull_method->setCurrentIndex(0);
    connect(cb_cull_method, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_cull_method](int) {
        render_config_.cull_method =
            static_cast<TracerConfig::CullMethod>(cb_cull_method->currentData().toInt());
        apply_render_config();
    });
    cull_form->addRow("Method:", cb_cull_method);

    auto* sp_slabs = new QSpinBox;
    sp_slabs->setRange(0, 128);
    sp_slabs->setSingleStep(8);
    sp_slabs->setValue(render_config_.cull_slabs);
    sp_slabs->setToolTip("Depth slabs per tile; 0 disables the cull");
    connect(sp_slabs, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int v) { render_config_.cull_slabs = v; apply_render_config(); });
    cull_form->addRow("Slabs:", sp_slabs);

    auto* sp_lip = new QDoubleSpinBox;
    sp_lip->setRange(0.1, 32.0);
    sp_lip->setDecimals(2);
    sp_lip->setSingleStep(0.5);
    sp_lip->setValue(render_config_.cull_lipschitz);
    sp_lip->setToolTip("Lipschitz constant L for the Lipschitz method "
                       "(1.0 for a true SDF; a raw implicit needs its max |grad f|)");
    connect(sp_lip, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v) {
        render_config_.cull_lipschitz = static_cast<float>(v); apply_render_config();
    });
    cull_form->addRow("Lipschitz L:", sp_lip);

    // Metrics overlay toggle (task 2). Drives an on-viewport HUD (ms/frame,
    // cull method in use, cull rate); wired to the viewport below.
    auto* cb_metrics = new QCheckBox;
    cb_metrics->setChecked(false);
    connect(cb_metrics, &QCheckBox::toggled, this, [this](bool on) {
        show_metrics_overlay_ = on;
        if (viewport_iv_) viewport_iv_->set_metrics_overlay(on);
        // On-viewport HUD: a small translucent label pinned to the top-left of
        // the viewport container, refreshed a few times a second from the
        // active viewport's metrics_text(). Created lazily on first enable.
        if (on) {
            if (!metrics_label_ && viewport_container_) {
                metrics_label_ = new QLabel(viewport_container_);
                metrics_label_->setStyleSheet(
                    "QLabel { color: #e0e0e0; background: rgba(0,0,0,140);"
                    " padding: 4px 8px; border-radius: 4px;"
                    " font-family: monospace; font-size: 11px; }");
                metrics_label_->setAttribute(Qt::WA_TransparentForMouseEvents);
                metrics_label_->move(10, 10);
            }
            if (!metrics_timer_) {
                metrics_timer_ = new QTimer(this);
                metrics_timer_->setInterval(250);
                connect(metrics_timer_, &QTimer::timeout, this, [this] {
                    if (!metrics_label_ || !viewport_iv_) return;
                    QString t = viewport_iv_->metrics_text();
                    if (t.isEmpty()) { metrics_label_->hide(); return; }
                    metrics_label_->setText(t);
                    metrics_label_->adjustSize();
                    metrics_label_->show();
                    metrics_label_->raise();
                });
            }
            metrics_timer_->start();
        } else {
            if (metrics_timer_) metrics_timer_->stop();
            if (metrics_label_) metrics_label_->hide();
        }
    });
    cull_form->addRow("Show metrics:", cb_metrics);

    // Debug view (task 3): step-count heatmap instead of shaded output, to see
    // where marching is expensive and where the cull skips work.
    auto* cb_debug = new QComboBox;
    cb_debug->addItem("Shaded",          static_cast<int>(TracerConfig::DebugView::Off));
    cb_debug->addItem("Step heatmap",    static_cast<int>(TracerConfig::DebugView::StepHeatmap));
    cb_debug->addItem("Cull span",       static_cast<int>(TracerConfig::DebugView::CullSpan));
    cb_debug->setCurrentIndex(0);
    connect(cb_debug, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this, cb_debug](int) {
        render_config_.debug_view =
            static_cast<TracerConfig::DebugView>(cb_debug->currentData().toInt());
        apply_render_config();
    });
    cull_form->addRow("Debug view:", cb_debug);

    // Instancing: shared subprograms (Level 2). On = emit each instanced shape
    // once as a GLSL function; Off = inline a copy at each instance.
    auto* cb_inst = new QCheckBox;
    cb_inst->setChecked(render_config_.instance_shared_subprograms);
    cb_inst->setToolTip("Emit instanced geometry once as a shared function "
                        "(smaller shader for repeated shapes) vs inlining copies");
    connect(cb_inst, &QCheckBox::toggled, this, [this](bool on) {
        render_config_.instance_shared_subprograms = on;
        apply_render_config();
    });
    cull_form->addRow("Share instances:", cb_inst);

    // Ray-box clip for the IR paths (CpuIr/GpuIr) — the analog of the GLSL tile
    // cull. Clips camera rays to the scene bounding box so empty space isn't
    // marched. On by default; auto-disabled for unbounded scenes (planes).
    auto* cb_bbox = new QCheckBox;
    cb_bbox->setChecked(render_config_.bbox_clip);
    cb_bbox->setToolTip("Clip IR-path rays to the scene bounding box "
                        "(skips empty space; CpuIr/GpuIr only)");
    connect(cb_bbox, &QCheckBox::toggled, this, [this](bool on) {
        render_config_.bbox_clip = on;
        apply_render_config();
    });
    cull_form->addRow("BBox clip (IR):", cb_bbox);

    v->addWidget(cull_box);

    v->addStretch(1);

    return panel;
}

// Plugins tab — lists the registered plugin primitives (moved out of the
// Render tab into its own tab).
QWidget* MainWindow::build_plugins_panel() {
    auto* panel = new QWidget;
    auto* v = new QVBoxLayout(panel);

    auto* plugins_box = new QGroupBox("Registered plugins");
    auto* pv = new QVBoxLayout(plugins_box);
    auto* lst = new QListWidget;
    for (const auto& p : registry_->primitives()) {
        QString text = QString("%1 v%2\n  %3")
            .arg(QString::fromStdString(std::string(p.info.name)))
            .arg(QString::fromStdString(std::string(p.info.version)))
            .arg(QString::fromStdString(std::string(p.info.description)));
        lst->addItem(text);
    }
    pv->addWidget(lst);
    v->addWidget(plugins_box, 1);

    return panel;
}

void MainWindow::on_render_completed(double render_ms, double total_ms,
                                     bool was_cached, bool structure_unchanged) {
    QString msg = QString("Render %1ms | Total %2ms")
        .arg(render_ms, 0, 'f', 1)
        .arg(total_ms,  0, 'f', 1);
    if (was_cached) {
        msg += " | cached";
    } else if (structure_unchanged) {
        // Tree shape stayed the same, only parameter values changed.
        // Helpful breadcrumb: future versions could exploit this case.
        msg += " | params only";
    }
    // Show which backend/path produced the frame, driven by the active backend
    // (the path selection picks the backend, so this reflects what really ran).
    QString path;
    switch (active_backend_) {
        case Backend::OffscreenCpu:
            path = "CPU: IR→JIT (offscreen)"; break;
        case Backend::OffscreenCuda:
            if (viewport_ && viewport_->gpu_mode())
                path = "GPU: IR→PTX→CUDA (offscreen)";
            else
                path = "CPU: IR→JIT (CUDA unavailable)";
            break;
        case Backend::RealtimeGlsl:
            path = "GPU: GLSL→Vulkan (realtime swapchain)"; break;
        case Backend::Executor:
            // Single RTX path, or a multi-path composite. Show per-path timings
            // and the heterogeneous aggregate when more than one is active.
            if (executor_vp_) {
                path = executor_vp_->status_text();
                double agg = executor_vp_->aggregate_mpix_s();
                if (agg >= 0 && executor_vp_->active_paths().size() > 1)
                    path += QString("  |  Σ %1 Mpix/s").arg(agg, 0, 'f', 1);
            } else {
                path = "executor";
            }
            break;
        case Backend::None:
            path = "—"; break;
    }
    msg += " | " + path;
    statusBar()->showMessage(msg);
}

void MainWindow::on_scene_changed() {
    // Rebind instances first: an edit may have replaced a target's geometry root
    // (SetGeometryCommand swaps the pointer rather than mutating it), added a new
    // instance, or removed a target. Re-resolving here keeps every instance
    // pointing at the current geometry so the live-edit link holds. Cyclic or
    // dangling references are left empty by resolve_instances (no recursion).
    io::resolve_instances(*scene_, registry_);
    force_recompile_if_offscreen();
    // Propagate scene changes back into the node graph view. The
    // syncing_ guard prevents this from triggering a graph_changed →
    // sync_graph_to_scene → on_scene_changed loop.
    if (!syncing_) sync_scene_to_graph();
}

void MainWindow::on_graph_changed() {
    if (syncing_) return;
    sync_graph_to_scene();
}

void MainWindow::on_inspector_selection_changed(const QString& object_id) {
    auto id = object_id.toStdString();
    if (id == active_object_id_) return;  // no-op
    // Respect the "Follow selection" toggle in the graph toolbar — when
    // unchecked, the graph stays locked to its current active object
    // even when the inspector or viewport selects something else.
    if (graph_follow_selection_ && !graph_follow_selection_->isChecked()) return;
    active_object_id_ = id;
    if (!syncing_) sync_scene_to_graph();
}

void MainWindow::sync_scene_to_graph() {
    if (syncing_) return;
    syncing_ = true;

    auto& objs = scene_->objects();

    // If the active object disappeared from the scene (e.g. removed via
    // inspector), pick a fallback so the graph view never points at a
    // dangling id.
    if (!active_object_id_.empty() && objs.find(active_object_id_) == objs.end()) {
        active_object_id_ = objs.empty() ? std::string{} : objs.begin()->first;
    }
    if (active_object_id_.empty() && !objs.empty()) {
        active_object_id_ = objs.begin()->first;
    }

    // Repopulate the object picker dropdown — same set of ids as the
    // inspector list, ordered the same way. We block its signals so
    // the model-change doesn't fire currentIndexChanged → recursive
    // selection update.
    if (graph_object_picker_) {
        QSignalBlocker block(graph_object_picker_);
        graph_object_picker_->clear();
        for (const auto& [id, obj] : objs)
            graph_object_picker_->addItem(QString::fromStdString(id));
        if (!active_object_id_.empty()) {
            int idx = graph_object_picker_->findText(
                QString::fromStdString(active_object_id_));
            if (idx >= 0) graph_object_picker_->setCurrentIndex(idx);
        }
    }

    auto* graph_scene = graph_editor_->graph_scene();
    if (active_object_id_.empty()) {
        graph_scene->clear_graph();
    } else {
        graph_scene->load_from_tree(objs.at(active_object_id_).geometry);
    }
    statusBar()->showMessage(
        active_object_id_.empty()
            ? QString("Node graph: no object")
            : QString("Node graph showing '%1'").arg(
                  QString::fromStdString(active_object_id_)));
    syncing_ = false;
}

void MainWindow::sync_graph_to_scene() {
    if (syncing_) return;
    syncing_ = true;

    auto tree = graph_editor_->graph_scene()->build_tree();
    if (!tree) {
        // The output node has no connected input or the graph is invalid —
        // leave the scene untouched, just report in the status bar.
        statusBar()->showMessage("Node graph: output has no connected model");
        syncing_ = false;
        return;
    }

    auto& objs = scene_->objects();
    if (!active_object_id_.empty() && objs.count(active_object_id_)) {
        // Edit-in-place via the undo stack — graph edits become
        // undoable as a single atomic SetGeometryCommand. Without this
        // routing, Ctrl+Z would skip past node-graph changes entirely.
        if (undo_stack_) {
            undo_stack_->push_apply(
                std::make_unique<undo::SetGeometryCommand>(
                    *scene_, active_object_id_, tree));
        } else {
            scene_->set_geometry(active_object_id_, tree);
        }
    } else {
        // No active object yet — the graph created a tree from scratch.
        // Prompt the user for an id so they aren't surprised by an
        // auto-generated "graph_obj_3" appearing in the inspector.
        QString suggested = "graph_obj";
        int n = 0;
        while (scene_->objects().count(suggested.toStdString())) {
            ++n;
            suggested = QString("graph_obj_%1").arg(n);
        }
        bool ok = false;
        QString id_qs = QInputDialog::getText(
            this, "Name new object",
            "The node graph built a new object. Object id:",
            QLineEdit::Normal, suggested, &ok);
        if (!ok || id_qs.isEmpty()) {
            // User cancelled — discard the would-be add. The graph
            // state stays; nothing else changes.
            statusBar()->showMessage("Node graph → cancelled");
            syncing_ = false;
            return;
        }
        std::string id = id_qs.toStdString();
        // Ensure uniqueness even if user typed an existing id.
        int extra = 0;
        std::string final_id = id;
        while (scene_->objects().count(final_id)) {
            ++extra;
            final_id = id + "_" + std::to_string(extra);
        }
        tree->id = final_id;
        Material mat{{0.85f, 0.55f, 0.35f}};
        if (undo_stack_) {
            undo_stack_->push_apply(
                std::make_unique<undo::AddObjectCommand>(
                    *scene_, final_id, tree, mat));
        } else {
            scene_->add_object(tree, mat);
        }
        active_object_id_ = final_id;
    }
    // Inspector list shows ids + visibility checkboxes; refresh it so
    // any geometry kind change (e.g. now-a-Box was a Sphere) shows up.
    inspector_->refresh();
    force_recompile_if_offscreen();
    statusBar()->showMessage(
        QString("Node graph → object '%1' updated")
            .arg(QString::fromStdString(active_object_id_)));
    syncing_ = false;
}

void MainWindow::on_add_primitive(const QString& plugin_name) {
    auto name_str = plugin_name.toStdString();
    for (const auto& p : registry_->primitives()) {
        if (p.info.name == name_str) {
            static int counter = 0;
            std::string id = "plugin_" + name_str + "_" + std::to_string(counter++);
            // Default parameters from the plugin
            std::vector<float> params(p.param_defaults.begin(), p.param_defaults.end());
            auto node = p.create(params, id);

            std::mt19937 rng(counter * 7919u);
            std::uniform_real_distribution<float> dist(0.2f, 0.95f);
            undo_stack_->push_apply(
                std::make_unique<undo::AddObjectCommand>(
                    *scene_, id, node,
                    Material{{dist(rng), dist(rng), dist(rng)}}));
            inspector_->refresh();
            on_scene_changed();
            return;
        }
    }
    QMessageBox::warning(this, "Plugin error",
        QString("Plugin '%1' was not found in the registry.").arg(plugin_name));
}

QWidget* MainWindow::build_lights_panel() {
    auto* panel = new QWidget;
    auto* v = new QVBoxLayout(panel);

    // ── Lights list ──────────────────────────────────────────────────────────
    auto* lst_box = new QGroupBox("Lights");
    auto* lst_v = new QVBoxLayout(lst_box);

    lights_list_ = new QListWidget;
    lst_v->addWidget(lights_list_);

    auto* row = new QHBoxLayout;
    auto* add_btn = new QPushButton("Add");
    auto* rm_btn  = new QPushButton("Remove");
    row->addWidget(add_btn);
    row->addWidget(rm_btn);
    lst_v->addLayout(row);

    v->addWidget(lst_box);

    // ── Selected light editor ────────────────────────────────────────────────
    auto* edit_box = new QGroupBox("Selected light");
    auto* form = new QFormLayout(edit_box);

    auto mk_pos = [&]() {
        auto* sp = new QDoubleSpinBox;
        sp->setRange(-100.0, 100.0);
        sp->setDecimals(2);
        sp->setSingleStep(0.5);
        return sp;
    };
    light_pos_x_ = mk_pos();
    light_pos_y_ = mk_pos();
    light_pos_z_ = mk_pos();
    light_intensity_ = new QDoubleSpinBox;
    light_intensity_->setRange(0.0, 10.0);
    light_intensity_->setDecimals(2);
    light_intensity_->setSingleStep(0.1);
    light_color_btn_ = new QPushButton("(no light selected)");

    form->addRow("Position X:", light_pos_x_);
    form->addRow("Position Y:", light_pos_y_);
    form->addRow("Position Z:", light_pos_z_);
    form->addRow("Intensity:",  light_intensity_);
    form->addRow("Color:",      light_color_btn_);

    v->addWidget(edit_box);
    v->addStretch();

    // ── Wiring ───────────────────────────────────────────────────────────────
    // Helper that pushes the current editor state into scene_->lights()[i].
    auto push_changes_to_scene = [this]() {
        if (lights_selected_ < 0) return;
        auto& Ls = scene_->lights();
        if (lights_selected_ >= static_cast<int>(Ls.size())) return;
        auto& L = Ls[lights_selected_];
        L.pos[0] = static_cast<float>(light_pos_x_->value());
        L.pos[1] = static_cast<float>(light_pos_y_->value());
        L.pos[2] = static_cast<float>(light_pos_z_->value());
        L.intensity = static_cast<float>(light_intensity_->value());
        // Light geometry doesn't change the FRepNode tree, so we don't dirty
        // the scene hash (the compiler would re-emit anyway because the light
        // is baked into render_tile as a constant). Force the recompile.
        force_recompile_if_offscreen();
    };

    connect(light_pos_x_, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [push_changes_to_scene](double){ push_changes_to_scene(); });
    connect(light_pos_y_, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [push_changes_to_scene](double){ push_changes_to_scene(); });
    connect(light_pos_z_, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [push_changes_to_scene](double){ push_changes_to_scene(); });
    connect(light_intensity_, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [push_changes_to_scene](double){ push_changes_to_scene(); });

    connect(light_color_btn_, &QPushButton::clicked, this, [this]() {
        if (lights_selected_ < 0) return;
        auto& Ls = scene_->lights();
        if (lights_selected_ >= static_cast<int>(Ls.size())) return;
        auto& L = Ls[lights_selected_];
        QColor cur = QColor::fromRgbF(L.color[0], L.color[1], L.color[2]);
        QColor c = QColorDialog::getColor(cur, this, "Pick light colour");
        if (!c.isValid()) return;
        L.color[0] = static_cast<float>(c.redF());
        L.color[1] = static_cast<float>(c.greenF());
        L.color[2] = static_cast<float>(c.blueF());
        // Update the button preview swatch.
        light_color_btn_->setStyleSheet(
            QString("background-color: %1").arg(c.name()));
        force_recompile_if_offscreen();
    });

    connect(lights_list_, &QListWidget::currentRowChanged,
            this, [this](int row) {
        lights_selected_ = row;
        if (row < 0) {
            light_pos_x_->setValue(0);
            light_pos_y_->setValue(0);
            light_pos_z_->setValue(0);
            light_intensity_->setValue(0);
            light_color_btn_->setText("(no light selected)");
            light_color_btn_->setStyleSheet("");
            return;
        }
        auto& Ls = scene_->lights();
        if (row >= static_cast<int>(Ls.size())) return;
        const auto& L = Ls[row];

        // Block signals while we set values so we don't recompile per field.
        QSignalBlocker b1(light_pos_x_),     b2(light_pos_y_);
        QSignalBlocker b3(light_pos_z_),     b4(light_intensity_);

        light_pos_x_->setValue(L.pos[0]);
        light_pos_y_->setValue(L.pos[1]);
        light_pos_z_->setValue(L.pos[2]);
        light_intensity_->setValue(L.intensity);

        QColor c = QColor::fromRgbF(L.color[0], L.color[1], L.color[2]);
        light_color_btn_->setText(c.name());
        light_color_btn_->setStyleSheet(
            QString("background-color: %1").arg(c.name()));
    });

    connect(add_btn, &QPushButton::clicked, this, [this]() {
        auto& Ls = scene_->lights();
        Ls.push_back(PointLight{
            {0.0f, 6.0f, 4.0f},
            {1.0f, 1.0f, 1.0f},
            0.6f
        });
        refresh_lights_panel();
        lights_list_->setCurrentRow(static_cast<int>(Ls.size()) - 1);
        force_recompile_if_offscreen();
    });

    connect(rm_btn, &QPushButton::clicked, this, [this]() {
        if (lights_selected_ < 0) return;
        auto& Ls = scene_->lights();
        if (lights_selected_ >= static_cast<int>(Ls.size())) return;
        Ls.erase(Ls.begin() + lights_selected_);
        lights_selected_ = -1;
        refresh_lights_panel();
        force_recompile_if_offscreen();
    });

    refresh_lights_panel();
    return panel;
}

// ── LAN distributed render tab ──────────────────────────────────────────────
QWidget* MainWindow::build_lan_panel() {
    lan_ = std::make_unique<LanWidgets>();
    auto* panel = new QWidget;
    auto* col = new QVBoxLayout(panel);

    // ── Master group ────────────────────────────────────────────────────────
    auto* mg = new QGroupBox("Master (this machine renders the frame)");
    auto* mf = new QFormLayout(mg);
    lan_->port = new QSpinBox; lan_->port->setRange(1024, 65535);
    lan_->port->setValue(53900);
    mf->addRow("Port:", lan_->port);
    lan_->workers = new QSpinBox; lan_->workers->setRange(1, 64);
    lan_->workers->setValue(1);
    mf->addRow("Workers to await:", lan_->workers);
    lan_->tile = new QSpinBox; lan_->tile->setRange(8, 512);
    lan_->tile->setValue(64); lan_->tile->setSingleStep(8);
    mf->addRow("Tile size (px):", lan_->tile);
    lan_->scheduler = new QComboBox;
    lan_->scheduler->addItem("pull (work-stealing)", 0);
    lan_->scheduler->addItem("push (static)", 1);
    mf->addRow("Scheduler:", lan_->scheduler);
    auto* mrow = new QHBoxLayout;
    lan_->start_master = new QPushButton("Start render");
    lan_->stop_master = new QPushButton("Stop"); lan_->stop_master->setEnabled(false);
    mrow->addWidget(lan_->start_master); mrow->addWidget(lan_->stop_master);
    mf->addRow(mrow);
    lan_->master_status = new QLabel("idle");
    lan_->master_status->setWordWrap(true);
    mf->addRow("Status:", lan_->master_status);
    col->addWidget(mg);

    // Frame size note + the two ways to use LAN.
    auto* note = new QLabel(
        "Two ways to render over LAN:\n"
        "• Pick the \"lan\" path in the Render tab — the master starts "
        "automatically here on the port above (alone, or composited with local "
        "paths like cpu_ir). Then start worker(s) below or on other machines.\n"
        "• Or run a standalone one-shot render: Start render below to wait for "
        "workers and render a single frame into the viewport.\n"
        "Frame size + scene come from the Render tab and the current scene.");
    note->setWordWrap(true);
    note->setStyleSheet("color: gray; font-size: 11px;");
    col->addWidget(note);

    // ── Worker group ────────────────────────────────────────────────────────
    auto* wg = new QGroupBox("Worker (this machine helps another master)");
    auto* wf = new QFormLayout(wg);
    lan_->host = new QLineEdit("127.0.0.1");
    wf->addRow("Master host:", lan_->host);
    lan_->wport = new QSpinBox; lan_->wport->setRange(1024, 65535);
    lan_->wport->setValue(53900);
    wf->addRow("Master port:", lan_->wport);
    lan_->wpath = new QComboBox;
    for (auto pk : frep::exec::local_paths())
        lan_->wpath->addItem(frep::exec::path_kind_name(pk));
    wf->addRow("Render with:", lan_->wpath);
    auto* wrow = new QHBoxLayout;
    lan_->start_worker = new QPushButton("Start worker");
    lan_->stop_worker = new QPushButton("Stop"); lan_->stop_worker->setEnabled(false);
    wrow->addWidget(lan_->start_worker); wrow->addWidget(lan_->stop_worker);
    wf->addRow(wrow);
    lan_->worker_status = new QLabel("idle");
    lan_->worker_status->setWordWrap(true);
    wf->addRow("Status:", lan_->worker_status);
    // Debug-only: optionally show the last tile this worker rendered, beneath
    // the status. Off by default — it's purely diagnostic, not for throughput.
    lan_->worker_preview = new QCheckBox("Show rendered tiles (debug)");
    lan_->worker_preview->setChecked(false);
    wf->addRow(lan_->worker_preview);
    // A chronological list of every tile this worker has rendered, newest at
    // the bottom. Cleared on a new worker run and on toggling the checkbox.
    lan_->worker_preview_list = new QListWidget;
    lan_->worker_preview_list->setViewMode(QListView::IconMode);
    lan_->worker_preview_list->setIconSize(QSize(64, 64));
    lan_->worker_preview_list->setResizeMode(QListView::Adjust);
    lan_->worker_preview_list->setMovement(QListView::Static);
    lan_->worker_preview_list->setMinimumHeight(120);
    lan_->worker_preview_list->setStyleSheet(
        "background:#202020; border:1px solid #444;");
    lan_->worker_preview_list->setVisible(false);
    wf->addRow(lan_->worker_preview_list);
    connect(lan_->worker_preview, &QCheckBox::toggled, this, [this](bool on) {
        if (lan_ && lan_->worker_preview_list) {
            lan_->worker_preview_list->setVisible(on);
            lan_->worker_preview_list->clear();   // clear on toggle
        }
    });
    col->addWidget(wg);
    col->addStretch();

    connect(lan_->start_master, &QPushButton::clicked, this,
            [this] { start_lan_master(); });
    connect(lan_->stop_master, &QPushButton::clicked, this,
            [this] { stop_lan_master(); });
    connect(lan_->start_worker, &QPushButton::clicked, this,
            [this] { start_lan_worker(); });
    connect(lan_->stop_worker, &QPushButton::clicked, this,
            [this] { stop_lan_worker(); });

    return panel;
}

void MainWindow::start_lan_master() {
    if (!lan_) return;
    // Build a DistViewport with the configured settings and install it as the
    // active viewport so tiles paint into the main render area progressively.
    DistRenderConfig cfg;
    cfg.port = lan_->port->value();
    cfg.n_workers = lan_->workers->value();
    cfg.tile = lan_->tile->value();
    cfg.push = lan_->scheduler->currentData().toInt() == 1;
    QSize vp = viewport_container_ ? viewport_container_->size() : QSize(400, 300);
    cfg.width  = std::max(64, vp.width());
    cfg.height = std::max(64, vp.height());

    // Tear down the current viewport. If it's a DistViewport, stop() it
    // (synchronous join of the master thread) and delete it synchronously: a
    // synchronous delete runs ~QObject, which removes any posted invokeMethod
    // events targeting it atomically. deleteLater would leave queued repaint /
    // render_completed events that fire on a half-destroyed object → crash.
    if (viewport_iv_) {
        if (lan_ && lan_->dist_vp) {
            if (lan_->master_poll) lan_->master_poll->stop();
            DistViewport* old_dv = lan_->dist_vp;
            lan_->dist_vp = nullptr;
            disconnect(old_dv, nullptr, this, nullptr);
            old_dv->stop();
            QWidget* old_w = old_dv->widget();
            if (old_w && viewport_container_ && viewport_container_->layout())
                viewport_container_->layout()->removeWidget(old_w);
            if (viewport_iv_ == old_dv) {
                viewport_iv_ = nullptr; viewport_ = nullptr; executor_vp_ = nullptr;
            }
            delete old_dv;               // ~DistViewport joins + deletes canvas
        } else {
            // A normal path viewport — the usual async teardown is fine.
            QWidget* old_w = viewport_iv_->widget();
            if (old_w) {
                if (viewport_container_ && viewport_container_->layout())
                    viewport_container_->layout()->removeWidget(old_w);
                old_w->hide(); old_w->deleteLater();
            }
            viewport_iv_->deleteLater();
            viewport_iv_ = nullptr; viewport_ = nullptr; executor_vp_ = nullptr;
        }
    }
    auto* dv = new DistViewport(scene_, cfg, nullptr);  // owned via lan_->dist_vp
    lan_->dist_vp = dv;
    viewport_iv_ = dv;
    active_backend_ = Backend::None;   // DistViewport isn't one of the path backends
    if (viewport_container_ && viewport_container_->layout())
        viewport_container_->layout()->addWidget(dv->widget());
    connect(dv, &IViewport::render_completed, this, [this](int) {
        if (lan_ && lan_->dist_vp)
            lan_->master_status->setText(lan_->dist_vp->status_text());
        if (lan_) { lan_->start_master->setEnabled(true);
                    lan_->stop_master->setEnabled(false); }
    });

    dv->invalidate();   // start the master thread
    lan_->start_master->setEnabled(false);
    lan_->stop_master->setEnabled(true);
    lan_->master_status->setText("listening…");
    // Poll status into the label while running. Stored on lan_ so stop can kill
    // it; guards on dist_vp so it never touches a deleted viewport.
    if (!lan_->master_poll) {
        lan_->master_poll = new QTimer(this);
        connect(lan_->master_poll, &QTimer::timeout, this, [this] {
            if (lan_ && lan_->dist_vp && lan_->stop_master->isEnabled())
                lan_->master_status->setText(lan_->dist_vp->status_text());
            else if (lan_ && lan_->master_poll)
                lan_->master_poll->stop();
        });
    }
    lan_->master_poll->start(200);
}

void MainWindow::stop_lan_master() {
    if (!lan_ || !lan_->dist_vp) return;
    if (lan_->master_poll) lan_->master_poll->stop();
    // Stop + delete the DistViewport synchronously (see start_lan_master), then
    // rebuild the normal path viewport via install_backend.
    DistViewport* dv = lan_->dist_vp;
    lan_->dist_vp = nullptr;             // clear first: timer/lambda see null
    disconnect(dv, nullptr, this, nullptr);  // drop render_completed connection
    dv->stop();                          // interrupt + join master thread
    QWidget* old_w = dv->widget();
    if (old_w && viewport_container_ && viewport_container_->layout())
        viewport_container_->layout()->removeWidget(old_w);
    if (viewport_iv_ == dv) { viewport_iv_ = nullptr; viewport_ = nullptr;
                              executor_vp_ = nullptr; }
    delete dv;                           // ~QObject clears pending posted events
    install_backend(Backend::OffscreenCpu, { frep::exec::PathKind::CpuIr });
    lan_->start_master->setEnabled(true);
    lan_->stop_master->setEnabled(false);
    lan_->master_status->setText("stopped");
}

void MainWindow::start_lan_worker() {
    if (!lan_ || lan_worker_running_) return;
    std::string host = lan_->host->text().toStdString();
    int port = lan_->wport->value();
    std::string path = lan_->wpath->currentText().toStdString();
    QSize vp = viewport_container_ ? viewport_container_->size() : QSize(400, 300);
    int W = std::max(64, vp.width());
    int H = std::max(64, vp.height());
    TracerConfig cfg = render_config_;

    lan_worker_running_ = true;
    lan_->start_worker->setEnabled(false);
    lan_->stop_worker->setEnabled(true);
    lan_->worker_status->setText("connecting to " + QString::fromStdString(host) +
                                 ":" + QString::number(port) + "…");

    // A previous worker thread may still be joinable (it finished, but the
    // std::thread object was never joined/detached). Assigning a new thread to a
    // joinable std::thread calls std::terminate — so reap the old one first.
    // The previous run has already completed (lan_worker_running_ was false), so
    // join() returns immediately.
    if (lan_worker_thread_.joinable()) lan_worker_thread_.join();

    lan_worker_thread_ = std::thread([this, host, port, path, W, H, cfg] {
        dist::WorkerConfig wc;
        wc.host = host; wc.port = port; wc.width = W; wc.height = H;
        wc.retry_secs = 0; wc.verbose = false;
        // Debug preview: only attach the hook if the checkbox is on (captured
        // once at start). Each rendered tile is turned into a small QImage and
        // marshalled onto the GUI thread to update the preview label.
        bool preview = lan_ && lan_->worker_preview &&
                       lan_->worker_preview->isChecked();
        if (preview && lan_->worker_preview_list)
            lan_->worker_preview_list->clear();   // fresh run → empty the list
        if (preview) {
            wc.on_tile = [this](const exec::Tile& t, const exec::TileResult& rr) {
                int tw = t.x1 - t.x0, th = t.y1 - t.y0;
                if (tw <= 0 || th <= 0) return;
                QImage tile(tw, th, QImage::Format_ARGB32);
                for (int y = 0; y < th; ++y) {
                    auto* row = reinterpret_cast<QRgb*>(tile.scanLine(y));
                    for (int x = 0; x < tw; ++x) {
                        std::size_t p = ((std::size_t)y * tw + x) * 4;
                        if (p + 3 >= rr.rgba.size()) { row[x] = qRgba(0,0,0,255); continue; }
                        auto f = [&](float v){ return (int)std::clamp(v*255.f,0.f,255.f); };
                        row[x] = qRgba(f(rr.rgba[p]), f(rr.rgba[p+1]),
                                       f(rr.rgba[p+2]), 255);
                    }
                }
                // Marshal onto the GUI thread: append a thumbnail labelled with
                // the tile's frame-space position, newest scrolled into view.
                QString label = QString("(%1,%2)").arg(t.x0).arg(t.y0);
                QMetaObject::invokeMethod(this, [this, tile, label] {
                    if (lan_ && lan_->worker_preview_list &&
                        lan_->worker_preview->isChecked()) {
                        auto* item = new QListWidgetItem(
                            QIcon(QPixmap::fromImage(tile)), label);
                        lan_->worker_preview_list->addItem(item);
                        lan_->worker_preview_list->scrollToBottom();
                    }
                }, Qt::QueuedConnection);
            };
        }
        dist::Worker worker(wc, [path, cfg] {
            return exec::make_executor(path, cfg);
        });
        auto res = worker.run();
        QString msg = res ? QString("done — rendered %1 tiles").arg(*res)
                          : QString("error — %1").arg(QString::fromStdString(res.error()));
        QMetaObject::invokeMethod(this, [this, msg] {
            if (lan_) {
                lan_->worker_status->setText(msg);
                lan_->start_worker->setEnabled(true);
                lan_->stop_worker->setEnabled(false);
            }
            lan_worker_running_ = false;
        }, Qt::QueuedConnection);
    });
}

void MainWindow::stop_lan_worker() {
    // The worker blocks on its socket; detach so the GUI stays responsive and
    // it exits when the master closes the connection or the render completes.
    if (lan_worker_thread_.joinable()) lan_worker_thread_.detach();
    lan_worker_running_ = false;
    if (lan_) {
        lan_->worker_status->setText("stopping (finishes current tile)…");
        lan_->start_worker->setEnabled(true);
        lan_->stop_worker->setEnabled(false);
    }
}


// Defined here (not inline at line ~84) so unique_ptr<LanWidgets> sees the
// complete type when destroying the member.
MainWindow::~MainWindow() {
    if (lan_worker_thread_.joinable()) lan_worker_thread_.detach();
}

// ── Duplicate / Paste helpers ───────────────────────────────────────────────
namespace {

// Builds a clone id that doesn't collide with anything already in the
// scene. Starts from `<base>_copy` and appends an incrementing index
// until free.
std::string unique_clone_id(const SceneGraph& s, const std::string& base) {
    std::string cand = base + "_copy";
    int n = 1;
    while (s.objects().count(cand))
        cand = base + "_copy" + std::to_string(++n);
    return cand;
}
} // anonymous namespace

void MainWindow::duplicate_selection() {
    QStringList ids = inspector_->selected_ids();
    if (ids.isEmpty()) return;

    QStringList new_ids;
    for (const QString& qid : ids) {
        std::string src = qid.toStdString();
        auto it = scene_->objects().find(src);
        if (it == scene_->objects().end()) continue;

        std::string new_id = unique_clone_id(*scene_, src);
        // Deep-clone the geometry tree with fresh ids (registry-aware so
        // plugin primitives clone too). clone_node round-trips through
        // JSON, so every node type is handled without a bespoke clone().
        FRepNode::Ptr geom =
            io::clone_node(*it->second.geometry, new_id, registry_);
        if (!geom) continue;

        Material mat = it->second.material;
        undo_stack_->push_apply(std::make_unique<undo::AddObjectCommand>(
            *scene_, new_id, geom, mat));
        new_ids.push_back(QString::fromStdString(new_id));
    }

    if (!new_ids.isEmpty()) {
        inspector_->refresh();
        inspector_->select_objects(new_ids);   // select the clones
        on_scene_changed();
    }
}

void MainWindow::paste_clipboard() {
    if (clipboard_ids_.isEmpty()) return;

    QStringList new_ids;
    for (const QString& qid : clipboard_ids_) {
        std::string src = qid.toStdString();
        auto it = scene_->objects().find(src);
        if (it == scene_->objects().end()) continue;  // source gone

        std::string new_id = unique_clone_id(*scene_, src);
        FRepNode::Ptr geom =
            io::clone_node(*it->second.geometry, new_id, registry_);
        if (!geom) continue;

        Material mat = it->second.material;
        undo_stack_->push_apply(std::make_unique<undo::AddObjectCommand>(
            *scene_, new_id, geom, mat));
        new_ids.push_back(QString::fromStdString(new_id));
    }

    if (!new_ids.isEmpty()) {
        inspector_->refresh();
        inspector_->select_objects(new_ids);
        on_scene_changed();
    }
}

void MainWindow::refresh_undo_redo_actions() {
    if (!undo_action_ || !redo_action_) return;
    undo_action_->setEnabled(undo_stack_->can_undo());
    redo_action_->setEnabled(undo_stack_->can_redo());
    QString u = undo_stack_->can_undo()
        ? QString("Undo (%1)").arg(QString::fromStdString(
              undo_stack_->undo_description()))
        : QString("Undo");
    QString r = undo_stack_->can_redo()
        ? QString("Redo (%1)").arg(QString::fromStdString(
              undo_stack_->redo_description()))
        : QString("Redo");
    undo_action_->setText(u);
    redo_action_->setText(r);
}

void MainWindow::refresh_lights_panel() {
    if (!lights_list_) return;
    QSignalBlocker block(lights_list_);
    lights_list_->clear();
    const auto& Ls = scene_->lights();
    for (std::size_t i = 0; i < Ls.size(); ++i) {
        const auto& L = Ls[i];
        QString label = QString("Light %1  [%2, %3, %4]  I=%5")
            .arg(i + 1)
            .arg(L.pos[0], 0, 'f', 1)
            .arg(L.pos[1], 0, 'f', 1)
            .arg(L.pos[2], 0, 'f', 1)
            .arg(L.intensity, 0, 'f', 2);
        auto* item = new QListWidgetItem(label);
        QColor c = QColor::fromRgbF(L.color[0], L.color[1], L.color[2]);
        item->setForeground(c);
        lights_list_->addItem(item);
    }
    if (Ls.empty()) {
        lights_list_->addItem("(no lights — scene falls back to a default key light)");
        lights_list_->item(0)->setFlags(Qt::NoItemFlags);
    }
}

} // namespace frep::gui
