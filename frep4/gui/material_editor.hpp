#pragma once
// gui/material_editor.hpp
//
// Side-panel widget for editing the material (PBR + pattern + texture)
// of the currently selected scene object. Mirrors the field set in
// frep::Material: pattern type, albedo/albedo2 colors, pattern scale,
// roughness, metallic, texture path.
//
// Updates push through the undo stack so each individual slider
// adjustment can be reverted. The widget syncs FROM the scene when
// selection changes; it only writes back on explicit user interaction.

#include <QWidget>
#include <QString>
#include <QStringList>

#include <string>
#include <vector>

class QComboBox;
class QPushButton;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;

namespace frep {
class SceneGraph;
namespace undo { class UndoStack; }
}

namespace frep::gui {

class MaterialEditor : public QWidget {
    Q_OBJECT
public:
    explicit MaterialEditor(SceneGraph* scene, QWidget* parent = nullptr);
    // Defined in the .cpp where Snapshot is complete; needed because
    // we hold a unique_ptr<Snapshot> and the default destructor would
    // be inlined in callers, where Snapshot is forward-declared only.
    ~MaterialEditor() override;

    void set_undo_stack(undo::UndoStack* s) { undo_stack_ = s; }

public Q_SLOTS:
    // Called by MainWindow when the inspector selection changes.
    // Empty id means "no selection" — controls become disabled.
    void on_selection_changed(const QString& object_id);

    // Multi-selection variant: the first id is the "primary" whose
    // values populate the widgets; edits broadcast to all of them.
    // MainWindow calls this with the full selection list.
    void on_selection_changed_multi(const QStringList& ids);

Q_SIGNALS:
    void material_changed();

private Q_SLOTS:
    void on_pattern_changed(int idx);
    void on_albedo_clicked();
    void on_albedo2_clicked();
    void on_scale_changed(double v);
    void on_roughness_changed(double v);
    void on_metallic_changed(double v);
    void on_texture_browse();
    void on_texture_clear();

private:
    void refresh_from_scene();
    void apply_to_scene();
    // Push the current edits as a SetMaterialCommand. Called by any of
    // the on_*_changed slots, debounced so dragging a slider becomes
    // one undo entry rather than dozens.
    void push_undo();

    SceneGraph*       scene_      = nullptr;
    undo::UndoStack*  undo_stack_ = nullptr;
    std::string       selected_id_;
    // All selected object ids for multi-edit broadcast. selected_id_ is
    // ids[0] (the primary, whose values populate the widgets). When the
    // user edits a control, the change applies to every id in this list.
    std::vector<std::string> selected_ids_;

    // Saved BEFORE we start mutating — used as the "old" value when we
    // push the SetMaterialCommand on commit.
    bool              dirty_ = false;
    // The full Material snapshot at the moment selection changed.
    // Stored as a pointer-to-by-value so the header doesn't need the
    // Material definition.
    struct Snapshot;
    std::unique_ptr<Snapshot> baseline_;

    QComboBox*        cb_pattern_   = nullptr;
    QComboBox*        cb_preset_    = nullptr;
    QPushButton*      btn_albedo_   = nullptr;
    QPushButton*      btn_albedo2_  = nullptr;
    QDoubleSpinBox*   sp_scale_     = nullptr;
    QDoubleSpinBox*   sp_rough_     = nullptr;
    QDoubleSpinBox*   sp_metal_     = nullptr;
    QDoubleSpinBox*   sp_refl_      = nullptr;
    QLineEdit*        ed_texture_   = nullptr;
    QPushButton*      btn_tex_browse_ = nullptr;
    QPushButton*      btn_tex_clear_  = nullptr;
    QLabel*           lbl_tex_thumb_  = nullptr;   ///< 64×64 preview of the loaded texture
    QLabel*           lbl_tex_info_   = nullptr;   ///< Dimensions + payload size
    QLabel*           lbl_status_   = nullptr;
};

} // namespace frep::gui
