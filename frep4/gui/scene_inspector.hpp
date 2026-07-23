#pragma once
// gui/scene_inspector.hpp
//
// SceneInspector — a side panel showing the objects in the scene.
// On object selection it shows the property editor (albedo color, visibility).
// Supports multi-selection (Ctrl/Shift-click in the list, Ctrl-click on
// the viewport). Multi-edit applies to all selected objects.

#include <QStringList>
#include <QWidget>
#include <QTreeWidget>

#include <string>
#include <vector>

namespace frep {
class SceneGraph;
namespace undo { class UndoStack; }
}

class QListWidget;
class QListWidgetItem;
class QPushButton;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;

namespace frep::gui {

class SceneInspector : public QWidget {
    Q_OBJECT

public:
    explicit SceneInspector(SceneGraph* scene, QWidget* parent = nullptr);

    // Optional: routes mutating actions (remove, color, visibility) through
    // the undo stack so they become undoable. When not set, the inspector
    // calls SceneGraph mutators directly (good for tests).
    void set_undo_stack(undo::UndoStack* u) { undo_ = u; }

    // Called from outside when the scene has been changed externally.
    void refresh();

    // Current selection — empty if nothing selected. Order matches list
    // ordering (top to bottom in the inspector). The "primary" id is
    // selected_ids().first() when non-empty; it's the object whose
    // values populate the property panel (other selected objects keep
    // their independent state and only get touched when the user
    // explicitly edits a property).
    QStringList selected_ids() const;

Q_SIGNALS:
    void scene_changed();
    // Emitted whenever the active object selection changes. `ids` may be
    // empty (nothing selected). When non-empty, ids[0] is the primary
    // (most recently clicked) selection; the node graph and property
    // panel key off it. Consumers that care about every selected
    // object (e.g. multi-edit material properties) iterate the full
    // list.
    void selection_changed(const QStringList& ids);

public Q_SLOTS:
    // Programmatic selection by id — typically called on a ray-cast pick
    // from the viewport. `additive=true` means add to the current
    // selection (Ctrl-click from the viewport); `additive=false`
    // replaces. An empty id clears the selection (when not additive).
    void select_object(const QString& object_id, bool additive = false);

    // Replace the current selection with the given list of ids.
    void select_objects(const QStringList& ids);

private Q_SLOTS:
    void on_selection_changed();
    void on_remove_selected();
    void on_visibility_toggled(bool checked);
    void on_transform_changed();
    void on_rotation_changed();
    void on_scale_changed();
    void on_change_color();

private:
    SceneGraph*       scene_;
    undo::UndoStack*  undo_       = nullptr;  // optional — see set_undo_stack
    QListWidget* list_       = nullptr;
    QPushButton* btn_remove_ = nullptr;
    QPushButton* btn_color_  = nullptr;
    QCheckBox*   cb_visible_ = nullptr;
    QLabel*      lbl_info_   = nullptr;

    // Per-object translation gizmo (numeric). Editing any of these
    // applies a SetTransformCommand to the primary selection. They are
    // disabled when nothing (or a multi-selection) is active, since a
    // single translation field can't represent divergent offsets.
    QDoubleSpinBox* sp_tx_ = nullptr;
    QDoubleSpinBox* sp_ty_ = nullptr;
    QDoubleSpinBox* sp_tz_ = nullptr;

    // Grid snapping for the translation gizmo. When enabled, committed
    // translations are rounded to the nearest multiple of snap_step_.
    QCheckBox*      cb_snap_   = nullptr;
    QDoubleSpinBox* sp_snap_   = nullptr;

    // Y-axis rotation (shown in degrees, stored in radians) and uniform
    // scale gizmos. Like translation, single-selection only; editing
    // commits a SetRotationCommand / SetScaleCommand.
    QDoubleSpinBox* sp_rot_y_  = nullptr;
    QDoubleSpinBox* sp_rot_x_  = nullptr;
    QDoubleSpinBox* sp_rot_z_  = nullptr;
    QDoubleSpinBox* sp_scale_  = nullptr;
    QDoubleSpinBox* sp_scale_y_ = nullptr;
    QDoubleSpinBox* sp_scale_z_ = nullptr;

    // Property grid: a tree of the selected object's nodes and their numeric
    // parameters, editable. populate_properties() rebuilds it; on_property_edited
    // commits a value edit. building_props_ guards the itemChanged signal while
    // we populate (so programmatic fills don't look like user edits).
    QTreeWidget* prop_tree_ = nullptr;
    bool         building_props_ = false;
    void populate_properties(const QString& object_id);
    void on_property_edited(QTreeWidgetItem* item, int column);
};

} // namespace frep::gui
