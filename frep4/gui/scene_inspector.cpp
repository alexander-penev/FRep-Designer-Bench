// gui/scene_inspector.cpp

#include "scene_inspector.hpp"

#include "core/frep/scene.hpp"
#include "core/undo/undo_stack.hpp"
#include "core/io/scene_io.hpp"

#include <QAbstractItemView>
#include <QColorDialog>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QMessageBox>
#include <QSet>
#include <QCheckBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QDoubleSpinBox>

#include <cmath>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QVBoxLayout>

namespace frep::gui {

SceneInspector::SceneInspector(SceneGraph* scene, QWidget* parent)
    : QWidget(parent), scene_(scene)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);

    auto* lbl_title = new QLabel("<b>Objects in the scene</b>");
    layout->addWidget(lbl_title);

    list_ = new QListWidget;
    // Extended selection: Ctrl-click toggles individual items, Shift-
    // click range-selects, plain click resets and selects one. This
    // matches the standard file-manager / Houdini network conventions
    // the user already expects from other tools.
    list_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    layout->addWidget(list_, 1);
    connect(list_, &QListWidget::itemSelectionChanged,
            this, &SceneInspector::on_selection_changed);

    auto* hl_buttons = new QHBoxLayout;
    btn_remove_ = new QPushButton("Remove");
    btn_remove_->setEnabled(false);
    connect(btn_remove_, &QPushButton::clicked,
            this, &SceneInspector::on_remove_selected);
    hl_buttons->addWidget(btn_remove_);
    layout->addLayout(hl_buttons);

    auto* lbl_props = new QLabel("<b>Properties</b>");
    layout->addWidget(lbl_props);

    auto* form = new QFormLayout;
    cb_visible_ = new QCheckBox;
    cb_visible_->setEnabled(false);
    connect(cb_visible_, &QCheckBox::toggled,
            this, &SceneInspector::on_visibility_toggled);
    form->addRow("Visible:", cb_visible_);

    btn_color_ = new QPushButton("Change color");
    btn_color_->setEnabled(false);
    connect(btn_color_, &QPushButton::clicked,
            this, &SceneInspector::on_change_color);
    form->addRow("Albedo:", btn_color_);

    // ── Translation gizmo (numeric) ──────────────────────────────────────────
    // X/Y/Z world-space offset for the selected object. Editing commits
    // a SetTransformCommand (undoable). Disabled unless exactly one
    // object is selected — a single triple can't represent a
    // multi-selection with divergent offsets.
    auto make_t_spin = [this]() {
        auto* sp = new QDoubleSpinBox;
        sp->setRange(-1000.0, 1000.0);
        sp->setDecimals(3);
        sp->setSingleStep(0.1);
        sp->setEnabled(false);
        connect(sp, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &SceneInspector::on_transform_changed);
        return sp;
    };
    sp_tx_ = make_t_spin();
    sp_ty_ = make_t_spin();
    sp_tz_ = make_t_spin();
    auto* trow = new QHBoxLayout;
    trow->addWidget(sp_tx_);
    trow->addWidget(sp_ty_);
    trow->addWidget(sp_tz_);
    auto* twrap = new QWidget;
    twrap->setLayout(trow);
    form->addRow("Position (xyz):", twrap);

    // Grid snapping: checkbox + step size. When on, committed
    // translations round to the nearest multiple of the step. Useful
    // for aligning objects to a regular grid.
    auto* snap_row = new QHBoxLayout;
    cb_snap_ = new QCheckBox("Snap");
    cb_snap_->setEnabled(false);
    sp_snap_ = new QDoubleSpinBox;
    sp_snap_->setRange(0.01, 100.0);
    sp_snap_->setDecimals(2);
    sp_snap_->setSingleStep(0.25);
    sp_snap_->setValue(0.5);
    sp_snap_->setEnabled(false);
    // Re-snap the current position immediately when the user toggles
    // snapping on, so the object jumps to the grid right away.
    connect(cb_snap_, &QCheckBox::toggled, this, [this](bool on) {
        sp_snap_->setEnabled(on && sp_tx_->isEnabled());
        if (on) on_transform_changed();
    });
    snap_row->addWidget(cb_snap_);
    snap_row->addWidget(sp_snap_);
    auto* snap_wrap = new QWidget;
    snap_wrap->setLayout(snap_row);
    form->addRow("Grid:", snap_wrap);

    // ── Rotation gizmo (X / Y / Z) ────────────────────────────────────────────
    // Three-axis rotation in degrees, committed as SetRotationAxisCommand. Stored
    // internally in radians (canonical RotateX→Y→Z chain); shown in degrees.
    // Single-selection only, like translation.
    auto mk_rot_sp = [this]() {
        auto* sp = new QDoubleSpinBox;
        sp->setRange(-360.0, 360.0);
        sp->setDecimals(1);
        sp->setSingleStep(5.0);
        sp->setSuffix("\u00B0");
        sp->setWrapping(true);
        sp->setEnabled(false);
        connect(sp, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &SceneInspector::on_rotation_changed);
        return sp;
    };
    sp_rot_x_ = mk_rot_sp();
    sp_rot_y_ = mk_rot_sp();   // kept name; existing refresh/handlers use it
    sp_rot_z_ = mk_rot_sp();
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel("X")); row->addWidget(sp_rot_x_);
        row->addWidget(new QLabel("Y")); row->addWidget(sp_rot_y_);
        row->addWidget(new QLabel("Z")); row->addWidget(sp_rot_z_);
        form->addRow("Rotation:", row);
    }

    // ── Per-axis scale gizmo ──────────────────────────────────────────────────
    // Three factors (X/Y/Z). Equal values = uniform scale; unequal = a
    // non-uniform (ellipsoidal) scale. Each stays strictly positive. Committed
    // as a SetScaleCommand. Single-selection only.
    auto mk_scale_sp = [this]() {
        auto* sp = new QDoubleSpinBox;
        sp->setRange(0.05, 100.0);
        sp->setDecimals(3);
        sp->setSingleStep(0.1);
        sp->setValue(1.0);
        sp->setEnabled(false);
        connect(sp, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &SceneInspector::on_scale_changed);
        return sp;
    };
    sp_scale_   = mk_scale_sp();   // X (kept name for the existing handler)
    sp_scale_y_ = mk_scale_sp();
    sp_scale_z_ = mk_scale_sp();
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel("X")); row->addWidget(sp_scale_);
        row->addWidget(new QLabel("Y")); row->addWidget(sp_scale_y_);
        row->addWidget(new QLabel("Z")); row->addWidget(sp_scale_z_);
        form->addRow("Scale:", row);
    }

    layout->addLayout(form);

    // ── Property grid ─────────────────────────────────────────────────────────
    // A tree of the selected object's geometry nodes and their numeric
    // parameters. The value column is editable; committing an edit applies an
    // undoable SetParamCommand (parameter-only, so it can ride the incremental
    // recompile). Rebuilt on every selection change and scene edit.
    layout->addWidget(new QLabel("Properties:"));
    prop_tree_ = new QTreeWidget;
    prop_tree_->setColumnCount(2);
    prop_tree_->setHeaderLabels({"Node / parameter", "Value"});
    prop_tree_->setRootIsDecorated(true);
    prop_tree_->setAlternatingRowColors(true);
    prop_tree_->setEditTriggers(QAbstractItemView::DoubleClicked |
                                QAbstractItemView::SelectedClicked);
    prop_tree_->setMinimumHeight(160);
    connect(prop_tree_, &QTreeWidget::itemChanged,
            this, &SceneInspector::on_property_edited);
    layout->addWidget(prop_tree_, 1);

    lbl_info_ = new QLabel;
    lbl_info_->setWordWrap(true);
    lbl_info_->setStyleSheet("color: #888");
    layout->addWidget(lbl_info_);

    refresh();
}

void SceneInspector::refresh() {
    // Preserve the current selection across the rebuild. clear() destroys
    // every item (and with them the selection), which would otherwise
    // deselect the objects the user is editing — making the Material /
    // Properties panels disable themselves mid-edit. We snapshot the
    // selected ids and the primary, rebuild, then restore.
    QStringList prev_selected = selected_ids();           // primary is [0]
    QString     prev_primary  = prev_selected.isEmpty()
                                    ? QString() : prev_selected.first();

    {
        QSignalBlocker block(list_);   // don't emit selection churn mid-rebuild
        list_->clear();
        for (const auto& [id, obj] : scene_->objects()) {
            auto* item = new QListWidgetItem(QString::fromStdString(id));
            item->setData(Qt::UserRole, QString::fromStdString(id));
            if (!obj.visible) {
                item->setForeground(Qt::gray);
            }
            list_->addItem(item);
        }

        // Restore selection for ids that still exist.
        QListWidgetItem* primary_item = nullptr;
        for (int i = 0; i < list_->count(); ++i) {
            QListWidgetItem* item = list_->item(i);
            QString id = item->data(Qt::UserRole).toString();
            if (prev_selected.contains(id)) {
                item->setSelected(true);
                if (id == prev_primary) primary_item = item;
            }
        }
        if (primary_item) list_->setCurrentItem(primary_item);
    }

    // Emitting nothing here would leave the property/material panels
    // believing the selection changed (their enable-state was set up on
    // the previous selection). Only re-emit if the selection actually
    // survived — keeps editor widgets enabled so the user can keep
    // adjusting without re-clicking.
    QStringList now_selected = selected_ids();
    if (now_selected != prev_selected)
        on_selection_changed();   // selection genuinely changed (e.g. deletions)
}

QStringList SceneInspector::selected_ids() const {
    QStringList ids;
    auto items = list_->selectedItems();
    ids.reserve(items.size());
    // Reorder so the *current* item (the last-clicked) comes first;
    // downstream consumers treat ids[0] as the "primary" selection
    // and key the property panel / node graph off it.
    QListWidgetItem* primary = list_->currentItem();
    if (primary && items.contains(primary))
        ids.append(primary->data(Qt::UserRole).toString());
    for (auto* it : items) {
        if (it == primary) continue;
        ids.append(it->data(Qt::UserRole).toString());
    }
    return ids;
}

void SceneInspector::select_object(const QString& object_id, bool additive) {
    if (object_id.isEmpty()) {
        if (!additive) list_->clearSelection();
        return;
    }
    // Find the item with the matching id.
    for (int i = 0; i < list_->count(); ++i) {
        QListWidgetItem* item = list_->item(i);
        if (item->data(Qt::UserRole).toString() == object_id) {
            if (additive) {
                // Toggle: if already selected, deselect; if not, add.
                // Mirrors what Ctrl-click would do in the list itself,
                // matching the user's expectation that Ctrl is a
                // "toggle in-set membership" gesture regardless of
                // whether it originated from the list or the viewport.
                item->setSelected(!item->isSelected());
                if (item->isSelected())
                    list_->setCurrentItem(item);
            } else {
                list_->clearSelection();
                list_->setCurrentItem(item);
                item->setSelected(true);
            }
            list_->scrollToItem(item);
            return;
        }
    }
    // id not found (it may have been deleted) — clear when not additive.
    if (!additive) list_->clearSelection();
}

void SceneInspector::select_objects(const QStringList& ids) {
    list_->clearSelection();
    if (ids.isEmpty()) return;
    QListWidgetItem* primary_item = nullptr;
    for (int i = 0; i < list_->count(); ++i) {
        QListWidgetItem* item = list_->item(i);
        if (ids.contains(item->data(Qt::UserRole).toString())) {
            item->setSelected(true);
            if (!primary_item) primary_item = item;
        }
    }
    if (primary_item) {
        list_->setCurrentItem(primary_item);
        list_->scrollToItem(primary_item);
    }
}

void SceneInspector::on_selection_changed() {
    QStringList ids = selected_ids();
    bool has   = !ids.isEmpty();
    bool multi = ids.size() > 1;
    btn_remove_->setEnabled(has);
    cb_visible_->setEnabled(has);
    btn_color_->setEnabled(has);
    // Transform spinboxes only make sense for a single object — a lone
    // XYZ triple can't represent a multi-selection. Enable for exactly
    // one selected object.
    bool single = has && !multi;
    sp_tx_->setEnabled(single);
    sp_ty_->setEnabled(single);
    sp_tz_->setEnabled(single);
    cb_snap_->setEnabled(single);
    sp_snap_->setEnabled(single && cb_snap_->isChecked());
    sp_rot_y_->setEnabled(single);
    sp_rot_x_->setEnabled(single);
    sp_rot_z_->setEnabled(single);
    sp_scale_->setEnabled(single);
    sp_scale_y_->setEnabled(single);
    sp_scale_z_->setEnabled(single);

    // Notify listeners of the new selection — the node graph view uses
    // ids[0] (the primary) for its tree view; multi-edit consumers
    // iterate all of them.
    Q_EMIT selection_changed(ids);

    if (!has) {
        lbl_info_->clear();
        populate_properties(QString());   // clear the property grid
        return;
    }

    const auto& objs = scene_->objects();
    QString primary = ids.first();
    auto it = objs.find(primary.toStdString());
    if (it == objs.end()) return;

    // For multi-selection: compare each property across the set, and
    // show "—" (em dash) when the values diverge. This matches the
    // way Houdini's parameter editor handles mixed-value selections.
    bool visible_mixed = false;
    bool albedo_mixed  = false;
    for (int i = 1; i < ids.size(); ++i) {
        auto it2 = objs.find(ids[i].toStdString());
        if (it2 == objs.end()) continue;
        if (it2->second.visible != it->second.visible) visible_mixed = true;
        for (int c = 0; c < 3; ++c)
            if (std::abs(it2->second.material.albedo[c]
                       - it->second.material.albedo[c]) > 1e-5f)
                albedo_mixed = true;
    }

    cb_visible_->blockSignals(true);
    if (visible_mixed) {
        cb_visible_->setCheckState(Qt::PartiallyChecked);
        cb_visible_->setTristate(true);
    } else {
        cb_visible_->setTristate(false);
        cb_visible_->setChecked(it->second.visible);
    }
    cb_visible_->blockSignals(false);

    auto& m = it->second.material;
    QString albedo_text = albedo_mixed
        ? QString("(\u2014, \u2014, \u2014)")
        : QString("(%1, %2, %3)")
              .arg(m.albedo[0], 0, 'f', 2)
              .arg(m.albedo[1], 0, 'f', 2)
              .arg(m.albedo[2], 0, 'f', 2);
    QString header = multi
        ? QString("Selected (%1):\n").arg(ids.size())
        : QString("ID: %1\n").arg(primary);
    lbl_info_->setText(header
        + QString("Geometry kind: %1\nAlbedo: %2")
              .arg(static_cast<int>(it->second.geometry->kind))
              .arg(albedo_text));

    // Populate translation spinboxes from the primary object's current
    // offset (read back from the implicit TranslateNode root, if any).
    // Block signals so setting the values doesn't fire on_transform_changed
    // and create a spurious undo entry.
    {
        auto t = scene_->get_translation(primary.toStdString());
        QSignalBlocker bx(sp_tx_), by(sp_ty_), bz(sp_tz_);
        sp_tx_->setValue(t[0]);
        sp_ty_->setValue(t[1]);
        sp_tz_->setValue(t[2]);
    }
    // Rotation (radians → degrees for display) and per-axis scale.
    {
        const float r2d = 180.0f / 3.14159265358979f;
        float ax = scene_->get_rotation_axis(primary.toStdString(), 0) * r2d;
        float ay = scene_->get_rotation_axis(primary.toStdString(), 1) * r2d;
        float az = scene_->get_rotation_axis(primary.toStdString(), 2) * r2d;
        float sx, sy, sz;
        scene_->get_scale_xyz(primary.toStdString(), sx, sy, sz);
        QSignalBlocker brx(sp_rot_x_), bry(sp_rot_y_), brz(sp_rot_z_),
                       bs(sp_scale_), bsy(sp_scale_y_), bsz(sp_scale_z_);
        sp_rot_x_->setValue(ax);
        sp_rot_y_->setValue(ay);
        sp_rot_z_->setValue(az);
        sp_scale_->setValue(sx);
        sp_scale_y_->setValue(sy);
        sp_scale_z_->setValue(sz);
    }
    // Property grid for the primary selection (empty for multi/none selection).
    populate_properties(single ? primary : QString());
}

void SceneInspector::on_transform_changed() {
    // Only the primary (single) selection has editable transform fields.
    QStringList ids = selected_ids();
    if (ids.size() != 1) return;
    std::string id = ids.first().toStdString();
    std::array<float, 3> t = {
        static_cast<float>(sp_tx_->value()),
        static_cast<float>(sp_ty_->value()),
        static_cast<float>(sp_tz_->value()),
    };
    // Grid snapping: round each axis to the nearest multiple of the
    // step. We reflect the snapped values back into the spinboxes so
    // the UI shows where the object actually landed (blocking signals
    // to avoid re-entering this slot).
    if (cb_snap_->isChecked()) {
        float step = static_cast<float>(sp_snap_->value());
        if (step > 1e-4f) {
            for (float& v : t)
                v = std::round(v / step) * step;
            QSignalBlocker bx(sp_tx_), by(sp_ty_), bz(sp_tz_);
            sp_tx_->setValue(t[0]);
            sp_ty_->setValue(t[1]);
            sp_tz_->setValue(t[2]);
        }
    }
    if (undo_) {
        undo_->push_apply(
            std::make_unique<undo::SetTransformCommand>(*scene_, id, t));
    } else {
        scene_->set_translation(id, t);
    }
    Q_EMIT scene_changed();
}

void SceneInspector::on_rotation_changed() {
    QStringList ids = selected_ids();
    if (ids.size() != 1) return;
    std::string id = ids.first().toStdString();
    // UI is in degrees; the scene stores radians. Commit each axis whose value
    // differs from the scene's current angle, so a single spinbox edit produces
    // one undo entry rather than three.
    const float d2r = 3.14159265358979f / 180.0f;
    struct { QDoubleSpinBox* sp; int axis; } axes[] = {
        { sp_rot_x_, 0 }, { sp_rot_y_, 1 }, { sp_rot_z_, 2 } };
    for (auto& a : axes) {
        float want = static_cast<float>(a.sp->value()) * d2r;
        float have = scene_->get_rotation_axis(id, a.axis);
        if (std::abs(want - have) < 1e-6f) continue;
        if (undo_) {
            undo_->push_apply(std::make_unique<undo::SetRotationAxisCommand>(
                *scene_, id, a.axis, want));
        } else {
            scene_->set_rotation_axis(id, a.axis, want);
        }
    }
    Q_EMIT scene_changed();
}

// Roles used to stash the (node_id, param_name) an editable value row targets.
namespace { constexpr int RoleNodeId = Qt::UserRole + 1;
            constexpr int RoleParam  = Qt::UserRole + 2; }

// Recursively add a geometry node (and its children) to the property tree. Each
// node becomes a top-level/child item labelled "type (id)", with one editable
// child row per numeric parameter.
static void add_node_item(QTreeWidgetItem* parent, const frep::FRepNode& n) {
    auto* item = new QTreeWidgetItem(parent);
    item->setText(0, QString("%1 (%2)")
        .arg(QString::fromLatin1(n.type_name()))
        .arg(QString::fromStdString(n.id)));
    item->setFirstColumnSpanned(true);
    std::vector<std::string> keys;
    for (const auto& [k, v] : n.params) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) {
        auto* pr = new QTreeWidgetItem(item);
        pr->setText(0, QString::fromStdString(k));
        pr->setText(1, QString::number(n.params.at(k), 'g', 6));
        pr->setData(0, RoleNodeId, QString::fromStdString(n.id));
        pr->setData(0, RoleParam,  QString::fromStdString(k));
        pr->setFlags(pr->flags() | Qt::ItemIsEditable);
    }
    for (const auto& c : n.children)
        if (c) add_node_item(item, *c);
}

void SceneInspector::populate_properties(const QString& object_id) {
    if (!prop_tree_) return;
    building_props_ = true;
    prop_tree_->clear();
    const SceneObject* obj = object_id.isEmpty()
        ? nullptr : scene_->find_object(object_id.toStdString());
    if (obj && obj->geometry) {
        add_node_item(prop_tree_->invisibleRootItem(), *obj->geometry);
        prop_tree_->expandAll();
        prop_tree_->resizeColumnToContents(0);
    }
    building_props_ = false;
}

void SceneInspector::on_property_edited(QTreeWidgetItem* item, int column) {
    if (building_props_ || !item || column != 1) return;
    const QString node_id = item->data(0, RoleNodeId).toString();
    const QString param   = item->data(0, RoleParam).toString();
    if (node_id.isEmpty() || param.isEmpty()) return;   // not a value row
    bool ok = false;
    float value = item->text(1).toFloat(&ok);
    if (!ok) { populate_properties(selected_ids().value(0)); return; }  // revert bad input
    QStringList sel = selected_ids();
    if (sel.isEmpty()) return;
    const std::string obj = sel.first().toStdString();
    if (undo_) {
        undo_->push_apply(std::make_unique<undo::SetParamCommand>(
            *scene_, obj, node_id.toStdString(), param.toStdString(), value));
    } else {
        scene_->set_node_param(obj, node_id.toStdString(), param.toStdString(), value);
    }
    Q_EMIT scene_changed();
}

void SceneInspector::on_scale_changed() {
    QStringList ids = selected_ids();
    if (ids.size() != 1) return;
    std::string id = ids.first().toStdString();
    float sx = static_cast<float>(sp_scale_->value());
    float sy = static_cast<float>(sp_scale_y_->value());
    float sz = static_cast<float>(sp_scale_z_->value());
    if (undo_) {
        undo_->push_apply(
            std::make_unique<undo::SetScaleXYZCommand>(*scene_, id, sx, sy, sz));
    } else {
        scene_->set_scale_xyz(id, sx, sy, sz);
    }
    Q_EMIT scene_changed();
}

void SceneInspector::on_remove_selected() {
    QStringList ids = selected_ids();
    if (ids.isEmpty()) return;

    // Deleting a target must also remove every instance that references it,
    // otherwise those instances would dangle. Gather dependents across the whole
    // selection, warn once, and on confirmation delete instances + targets
    // together. (Deleting an instance itself has no dependents and skips this.)
    QStringList to_delete = ids;
    QSet<QString> dep_set;
    for (const QString& sid : ids) {
        for (const auto& dep : io::find_dependent_instances(*scene_, sid.toStdString())) {
            QString q = QString::fromStdString(dep);
            if (!ids.contains(q)) dep_set.insert(q);
        }
    }
    if (!dep_set.isEmpty()) {
        QStringList deps = dep_set.values();
        auto reply = QMessageBox::question(this, "Delete object and its instances",
            QString("%1 instance(s) reference the object(s) you're deleting:\n  %2\n\n"
                    "Deleting will also remove those instances. Continue?")
                .arg(deps.size()).arg(deps.join(", ")),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (reply != QMessageBox::Yes) return;
        to_delete += deps;
    }

    // Remove in reverse so indexing inside scene_->objects() doesn't shift.
    for (int i = to_delete.size() - 1; i >= 0; --i) {
        std::string id = to_delete[i].toStdString();
        if (undo_) {
            undo_->push_apply(
                std::make_unique<undo::RemoveObjectCommand>(*scene_, id));
        } else {
            scene_->remove_object(id);
        }
    }
    refresh();
    Q_EMIT scene_changed();
}

void SceneInspector::on_visibility_toggled(bool checked) {
    QStringList ids = selected_ids();
    if (ids.isEmpty()) return;
    // When the checkbox was PartiallyChecked because the selection
    // had mixed visibilities, this slot still fires with `checked`
    // = whatever Qt landed on (true or false). Either way we now
    // force every selected object to that state — the user clicked
    // to resolve the ambiguity.
    for (const QString& qid : ids) {
        std::string id = qid.toStdString();
        if (undo_) {
            undo_->push_apply(
                std::make_unique<undo::SetVisibilityCommand>(*scene_, id, checked));
        } else {
            scene_->set_visibility(id, checked);
        }
    }
    refresh();
    Q_EMIT scene_changed();
}

void SceneInspector::on_change_color() {
    QStringList ids = selected_ids();
    if (ids.isEmpty()) return;

    const auto& objs = scene_->objects();
    // Use the primary object's current colour to populate the dialog.
    auto it_primary = objs.find(ids.first().toStdString());
    if (it_primary == objs.end()) return;
    const auto& m_primary = it_primary->second.material;
    QColor cur(static_cast<int>(m_primary.albedo[0] * 255),
               static_cast<int>(m_primary.albedo[1] * 255),
               static_cast<int>(m_primary.albedo[2] * 255));

    QString title = ids.size() > 1
        ? QString("Pick albedo (applies to %1 objects)").arg(ids.size())
        : QString("Pick albedo");
    QColor c = QColorDialog::getColor(cur, this, title);
    if (!c.isValid()) return;

    // Apply the chosen colour to every selected object. Each object
    // keeps its own material struct (roughness, metallic, etc.) — we
    // only mutate the albedo channel so a mixed-material selection
    // doesn't lose its other settings.
    for (const QString& qid : ids) {
        std::string id = qid.toStdString();
        auto it = objs.find(id);
        if (it == objs.end()) continue;
        Material new_mat = it->second.material;
        new_mat.albedo = {static_cast<float>(c.redF()),
                          static_cast<float>(c.greenF()),
                          static_cast<float>(c.blueF())};
        if (undo_) {
            undo_->push_apply(
                std::make_unique<undo::SetMaterialCommand>(*scene_, id, new_mat));
        } else {
            scene_->set_material(id, new_mat);
        }
    }

    on_selection_changed();  // refresh info label
    Q_EMIT scene_changed();
}

} // namespace frep::gui
