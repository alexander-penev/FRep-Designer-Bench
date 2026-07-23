// gui/material_editor.cpp

#include "gui/material_editor.hpp"

#include "core/frep/scene.hpp"
#include "core/io/png_loader.hpp"
#include "core/undo/undo_stack.hpp"
#include <map>

#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>

#include <memory>

namespace frep::gui {

// Snapshot of the material as it was when selection changed. Kept as a
// nested struct so the header doesn't need to know the field layout.
struct MaterialEditor::Snapshot {
    // Baseline material of the primary object — drives the editor's
    // widget values on selection.
    Material mat;
    // Per-object baseline materials for every object in the current
    // (possibly multi-) selection, captured at selection time. push_undo
    // diffs each against the object's current material so one undo entry
    // reverts the whole broadcast, not just the primary object.
    std::map<std::string, Material> per_object;
};

namespace {
const char* kPatternNames[] = {
    "Solid", "Checker", "Stripes", "GradientY", "Noise", "Texture"
};

// Material presets — quick PBR starting points. Each sets the scalar
// reflectance fields but leaves albedo untouched (so an object keeps
// its colour when you apply a preset). Reflectivity only has a visible
// effect when the Render tab's bounce count > 0.
struct MaterialPreset {
    const char* name;
    float roughness;
    float metallic;
    float reflectivity;
};
const MaterialPreset kMaterialPresets[] = {
    // name           rough  metal  refl
    {"Metal",         0.20f, 1.00f, 0.60f},  // polished metal
    {"Brushed metal", 0.55f, 1.00f, 0.30f},  // satin/brushed
    {"Plastic",       0.40f, 0.00f, 0.05f},  // glossy dielectric
    {"Matte",         0.90f, 0.00f, 0.00f},  // chalky, no reflection
    {"Glass",         0.05f, 0.00f, 0.90f},  // smooth, highly reflective
    {"Mirror",        0.02f, 1.00f, 1.00f},  // perfect mirror
    {"Emissive",      0.50f, 0.00f, 0.00f},  // emission handled elsewhere
};

QString to_hex(const std::array<float, 3>& a) {
    auto clamp = [](float v) {
        return std::clamp(static_cast<int>(v * 255 + 0.5f), 0, 255);
    };
    return QColor(clamp(a[0]), clamp(a[1]), clamp(a[2])).name();
}

// Convert a row-major RGBA8 buffer (stored in `Material::texture_rgba`)
// into a QPixmap suitable for the 64×64 preview label. Scales with
// smooth interpolation to keep small details visible without
// introducing pixelated stairstepping at the preview size.
QPixmap make_thumbnail(const Material& mat, int side) {
    if (mat.texture_width <= 0 || mat.texture_height <= 0 ||
        mat.texture_rgba.empty())
    {
        return {};
    }
    // QImage wraps the existing buffer without copying — but we
    // then call .copy() before scaling because QPixmap::fromImage on
    // a wrapped-buffer QImage can outlive the buffer if the underlying
    // Material is modified mid-render. The copy makes the QPixmap
    // self-contained.
    QImage src(mat.texture_rgba.data(),
               mat.texture_width, mat.texture_height,
               mat.texture_width * 4,
               QImage::Format_RGBA8888);
    return QPixmap::fromImage(src.copy().scaled(
        side, side, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QString format_tex_info(const Material& mat) {
    if (mat.texture_width <= 0 || mat.texture_height <= 0 ||
        mat.texture_rgba.empty())
    {
        return QObject::tr("(no texture)");
    }
    qint64 bytes = static_cast<qint64>(mat.texture_rgba.size());
    QString size_str;
    if (bytes >= 1024 * 1024)
        size_str = QString("%1 MB").arg(double(bytes) / (1024.0 * 1024.0),
                                        0, 'f', 1);
    else if (bytes >= 1024)
        size_str = QString("%1 KB").arg(double(bytes) / 1024.0, 0, 'f', 1);
    else
        size_str = QString("%1 B").arg(bytes);
    return QString("%1×%2 RGBA8\n%3 in memory")
        .arg(mat.texture_width).arg(mat.texture_height).arg(size_str);
}
}  // anon

MaterialEditor::~MaterialEditor() = default;

MaterialEditor::MaterialEditor(SceneGraph* scene, QWidget* parent)
    : QWidget(parent), scene_(scene)
{
    auto* layout = new QVBoxLayout(this);
    auto* hint = new QLabel(
        "<b>Material</b><br>"
        "Edits the material of the currently selected object. "
        "Select an object in the Scene tab to enable these controls.");
    hint->setWordWrap(true);
    layout->addWidget(hint);

    auto* form = new QFormLayout;

    // ── Material presets ────────────────────────────────────────────────────
    // One-click starting points. Selecting a preset fills in roughness,
    // metallic, reflectivity (and leaves albedo alone so the object keeps
    // its colour). Applies to all selected objects via apply_to_scene.
    // "(custom)" is the inert first entry shown when no preset is active.
    cb_preset_ = new QComboBox(this);
    cb_preset_->addItem("(custom)");
    for (const auto& p : kMaterialPresets) cb_preset_->addItem(p.name);
    form->addRow("Preset:", cb_preset_);

    cb_pattern_ = new QComboBox(this);
    for (auto* n : kPatternNames) cb_pattern_->addItem(n);
    form->addRow("Pattern:", cb_pattern_);

    btn_albedo_  = new QPushButton(this);
    btn_albedo_->setText("Click to pick");
    form->addRow("Albedo (primary):", btn_albedo_);

    btn_albedo2_ = new QPushButton(this);
    btn_albedo2_->setText("Click to pick");
    form->addRow("Albedo (secondary):", btn_albedo2_);

    sp_scale_ = new QDoubleSpinBox(this);
    sp_scale_->setRange(0.0, 100.0);
    sp_scale_->setDecimals(3);
    sp_scale_->setSingleStep(0.1);
    form->addRow("Pattern scale:", sp_scale_);

    sp_rough_ = new QDoubleSpinBox(this);
    sp_rough_->setRange(0.0, 1.0);
    sp_rough_->setDecimals(3);
    sp_rough_->setSingleStep(0.05);
    form->addRow("Roughness:", sp_rough_);

    sp_metal_ = new QDoubleSpinBox(this);
    sp_metal_->setRange(0.0, 1.0);
    sp_metal_->setDecimals(3);
    sp_metal_->setSingleStep(0.05);
    form->addRow("Metallic:", sp_metal_);

    // Mirror reflectivity. Only has a visible effect when the Render
    // tab's "Reflections (bounces)" is > 0 — otherwise the renderer
    // casts no secondary rays and this value is ignored. 0 = matte,
    // 1 = perfect mirror.
    sp_refl_ = new QDoubleSpinBox(this);
    sp_refl_->setRange(0.0, 1.0);
    sp_refl_->setDecimals(3);
    sp_refl_->setSingleStep(0.05);
    form->addRow("Reflectivity:", sp_refl_);

    auto* tex_row = new QHBoxLayout;
    ed_texture_      = new QLineEdit(this);
    ed_texture_->setReadOnly(true);
    btn_tex_browse_  = new QPushButton("Browse…", this);
    btn_tex_clear_   = new QPushButton("Clear", this);
    tex_row->addWidget(ed_texture_, 1);
    tex_row->addWidget(btn_tex_browse_);
    tex_row->addWidget(btn_tex_clear_);
    form->addRow("Texture:", tex_row);

    // Preview row — thumbnail on the left, dimensions/size on the right.
    // The thumbnail is rebuilt directly from the in-memory RGBA buffer
    // (Material::texture_rgba) on every refresh, so there's no risk of
    // it falling out of sync with what the renderer actually samples.
    auto* preview_row = new QHBoxLayout;
    lbl_tex_thumb_ = new QLabel(this);
    lbl_tex_thumb_->setFixedSize(64, 64);
    lbl_tex_thumb_->setFrameShape(QFrame::Box);
    lbl_tex_thumb_->setStyleSheet("background-color: #222;");
    lbl_tex_thumb_->setAlignment(Qt::AlignCenter);
    lbl_tex_thumb_->setText("—");
    lbl_tex_info_ = new QLabel("(no texture)", this);
    lbl_tex_info_->setStyleSheet("color: #888;");
    lbl_tex_info_->setWordWrap(true);
    preview_row->addWidget(lbl_tex_thumb_);
    preview_row->addWidget(lbl_tex_info_, 1);
    form->addRow("Preview:", preview_row);

    layout->addLayout(form);

    lbl_status_ = new QLabel("(no selection)", this);
    lbl_status_->setStyleSheet("color: #888;");
    layout->addWidget(lbl_status_);
    layout->addStretch(1);

    connect(cb_pattern_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MaterialEditor::on_pattern_changed);
    connect(btn_albedo_,    &QPushButton::clicked, this, &MaterialEditor::on_albedo_clicked);
    connect(btn_albedo2_,   &QPushButton::clicked, this, &MaterialEditor::on_albedo2_clicked);
    connect(sp_scale_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MaterialEditor::on_scale_changed);
    connect(sp_rough_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MaterialEditor::on_roughness_changed);
    connect(sp_metal_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MaterialEditor::on_metallic_changed);
    connect(sp_refl_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double){ apply_to_scene(); });

    // Preset selection: apply the chosen preset's scalar fields to the
    // widgets, then broadcast. Index 0 is "(custom)" — selecting it is
    // a no-op (the user is just looking).
    connect(cb_preset_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int idx) {
        if (idx <= 0) return;  // "(custom)"
        const auto& p = kMaterialPresets[idx - 1];
        QSignalBlocker br(sp_rough_), bm(sp_metal_), bf(sp_refl_);
        sp_rough_->setValue(p.roughness);
        sp_metal_->setValue(p.metallic);
        sp_refl_->setValue(p.reflectivity);
        apply_to_scene();
    });
    connect(btn_tex_browse_, &QPushButton::clicked, this, &MaterialEditor::on_texture_browse);
    connect(btn_tex_clear_,  &QPushButton::clicked, this, &MaterialEditor::on_texture_clear);

    on_selection_changed({});  // start disabled
}

void MaterialEditor::on_selection_changed(const QString& object_id) {
    // Commit any pending edit before switching.
    if (dirty_) push_undo();
    selected_id_ = object_id.toStdString();
    selected_ids_.clear();
    if (!selected_id_.empty()) selected_ids_.push_back(selected_id_);
    refresh_from_scene();
}

void MaterialEditor::on_selection_changed_multi(const QStringList& ids) {
    if (dirty_) push_undo();
    selected_ids_.clear();
    for (const QString& id : ids)
        selected_ids_.push_back(id.toStdString());
    selected_id_ = selected_ids_.empty() ? std::string{} : selected_ids_.front();
    refresh_from_scene();
}

void MaterialEditor::refresh_from_scene() {
    bool has = !selected_id_.empty() &&
               scene_->objects().count(selected_id_) > 0;

    // Enable / disable all controls based on whether there's a target.
    for (QWidget* w : {static_cast<QWidget*>(cb_pattern_),
                       static_cast<QWidget*>(cb_preset_),
                       static_cast<QWidget*>(btn_albedo_),
                       static_cast<QWidget*>(btn_albedo2_),
                       static_cast<QWidget*>(sp_scale_),
                       static_cast<QWidget*>(sp_rough_),
                       static_cast<QWidget*>(sp_metal_),
                       static_cast<QWidget*>(sp_refl_),
                       static_cast<QWidget*>(ed_texture_),
                       static_cast<QWidget*>(btn_tex_browse_),
                       static_cast<QWidget*>(btn_tex_clear_),
                       static_cast<QWidget*>(lbl_tex_thumb_),
                       static_cast<QWidget*>(lbl_tex_info_)})
        w->setEnabled(has);

    if (!has) {
        lbl_status_->setText("(no selection)");
        lbl_tex_thumb_->setPixmap({});
        lbl_tex_thumb_->setText("—");
        lbl_tex_info_->setText("(no texture)");
        baseline_.reset();
        return;
    }

    const auto& mat = scene_->objects().at(selected_id_).material;
    baseline_ = std::make_unique<Snapshot>();
    baseline_->mat = mat;
    // Snapshot every selected object's current material so a coalesced
    // multi-object undo can revert them all (see push_undo).
    baseline_->per_object.clear();
    for (const std::string& id : selected_ids_)
        if (scene_->objects().count(id))
            baseline_->per_object[id] = scene_->objects().at(id).material;

    // Push values to widgets while suppressing the change signals — we
    // don't want this refresh to look like a user edit.
    QSignalBlocker b0(cb_pattern_), b1(sp_scale_), b2(sp_rough_),
                   b3(sp_metal_),   b4(ed_texture_), b5(sp_refl_),
                   b6(cb_preset_);
    cb_pattern_->setCurrentIndex(static_cast<int>(mat.pattern));
    cb_preset_->setCurrentIndex(0);  // "(custom)" — presets are write-only triggers
    sp_scale_->setValue(mat.pattern_scale);
    sp_rough_->setValue(mat.roughness);
    sp_metal_->setValue(mat.metallic);
    sp_refl_->setValue(mat.reflectivity);
    ed_texture_->setText(QString::fromStdString(mat.texture_path));
    btn_albedo_->setStyleSheet(
        QString("background-color: %1").arg(to_hex(mat.albedo)));
    btn_albedo_->setText(to_hex(mat.albedo));
    btn_albedo2_->setStyleSheet(
        QString("background-color: %1").arg(to_hex(mat.albedo2)));
    btn_albedo2_->setText(to_hex(mat.albedo2));

    // Preview: regenerate thumbnail + info from in-memory RGBA.
    QPixmap pm = make_thumbnail(mat, lbl_tex_thumb_->width());
    if (pm.isNull()) {
        lbl_tex_thumb_->setPixmap({});
        lbl_tex_thumb_->setText("—");
    } else {
        lbl_tex_thumb_->setPixmap(pm);
        lbl_tex_thumb_->setText({});
    }
    lbl_tex_info_->setText(format_tex_info(mat));

    lbl_status_->setText(
        QString("Editing material of '%1'").arg(
            QString::fromStdString(selected_id_)));
    dirty_ = false;
}

void MaterialEditor::apply_to_scene() {
    if (selected_id_.empty() ||
        !scene_->objects().count(selected_id_)) return;

    // Broadcast the edited PBR/pattern fields to every selected object.
    // Each object keeps its own albedo/albedo2/texture (those are edited
    // through the dedicated colour buttons / texture browser, which act
    // on the primary only) — here we only push the scalar/enum controls
    // that the user just touched: pattern, scale, roughness, metallic,
    // reflectivity. This matches the Inspector's "apply to all" colour
    // behaviour while preserving each object's distinct base colour.
    auto pat   = static_cast<Material::Pattern>(cb_pattern_->currentIndex());
    auto scale = static_cast<float>(sp_scale_->value());
    auto rough = static_cast<float>(sp_rough_->value());
    auto metal = static_cast<float>(sp_metal_->value());
    auto refl  = static_cast<float>(sp_refl_->value());

    for (const std::string& id : selected_ids_) {
        if (!scene_->objects().count(id)) continue;
        Material m = scene_->objects().at(id).material;
        m.pattern       = pat;
        m.pattern_scale = scale;
        m.roughness     = rough;
        m.metallic      = metal;
        m.reflectivity  = refl;
        scene_->set_material(id, m);
    }
    dirty_ = true;
    Q_EMIT material_changed();
}

void MaterialEditor::push_undo() {
    // Coalesce the editing session into one undo entry covering EVERY
    // object the edit was broadcast to — not just the primary. For each
    // object we diff its captured baseline against its current material;
    // objects that actually changed go into a single SetMaterialsCommand
    // so one Ctrl+Z reverts the whole multi-object edit at once.
    if (!dirty_ || !undo_stack_ || !baseline_) {
        dirty_ = false;
        return;
    }

    std::vector<undo::SetMaterialsCommand::Entry> entries;
    for (const auto& [id, old_mat] : baseline_->per_object) {
        if (!scene_->objects().count(id)) continue;     // deleted mid-edit
        Material cur = scene_->objects().at(id).material;
        if (!(cur == old_mat))                           // only real changes
            entries.push_back({id, old_mat, cur});
    }

    if (!entries.empty()) {
        // Roll back to baseline, then push the apply so the command's
        // apply() re-establishes the edited state and undo() restores
        // the baseline — one entry for the whole session.
        for (const auto& e : entries)
            scene_->set_material(e.id, e.old_mat);
        undo_stack_->push_apply(
            std::make_unique<undo::SetMaterialsCommand>(*scene_, entries));
        // Refresh the captured baselines to the new current state so a
        // subsequent edit in the same selection diffs from here.
        for (const auto& e : entries)
            baseline_->per_object[e.id] = e.new_mat;
        baseline_->mat = scene_->objects().count(selected_id_)
            ? scene_->objects().at(selected_id_).material
            : baseline_->mat;
    }
    dirty_ = false;
}

void MaterialEditor::on_pattern_changed(int /*idx*/)  { apply_to_scene(); }
void MaterialEditor::on_scale_changed(double /*v*/)   { apply_to_scene(); }
void MaterialEditor::on_roughness_changed(double /*v*/) { apply_to_scene(); }
void MaterialEditor::on_metallic_changed(double /*v*/)  { apply_to_scene(); }

void MaterialEditor::on_albedo_clicked() {
    if (selected_id_.empty()) return;
    auto& m = scene_->objects().at(selected_id_).material;
    QColor c = QColorDialog::getColor(
        QColor::fromRgbF(m.albedo[0], m.albedo[1], m.albedo[2]),
        this, "Albedo (primary)");
    if (!c.isValid()) return;
    Material nm = m;
    nm.albedo = {float(c.redF()), float(c.greenF()), float(c.blueF())};
    scene_->set_material(selected_id_, nm);
    btn_albedo_->setStyleSheet(QString("background-color: %1").arg(c.name()));
    btn_albedo_->setText(c.name());
    dirty_ = true;
    Q_EMIT material_changed();
}

void MaterialEditor::on_albedo2_clicked() {
    if (selected_id_.empty()) return;
    auto& m = scene_->objects().at(selected_id_).material;
    QColor c = QColorDialog::getColor(
        QColor::fromRgbF(m.albedo2[0], m.albedo2[1], m.albedo2[2]),
        this, "Albedo (secondary)");
    if (!c.isValid()) return;
    Material nm = m;
    nm.albedo2 = {float(c.redF()), float(c.greenF()), float(c.blueF())};
    scene_->set_material(selected_id_, nm);
    btn_albedo2_->setStyleSheet(QString("background-color: %1").arg(c.name()));
    btn_albedo2_->setText(c.name());
    dirty_ = true;
    Q_EMIT material_changed();
}

void MaterialEditor::on_texture_browse() {
    if (selected_id_.empty()) return;
    QString path = QFileDialog::getOpenFileName(
        this, "Select texture image", "",
        "Image files (*.png *.bmp);;All files (*)");
    if (path.isEmpty()) return;

    auto img = io::load_image(path.toStdString());
    if (img.width <= 0 || img.height <= 0) {
        lbl_status_->setText(QString("Failed to load %1").arg(path));
        lbl_status_->setStyleSheet("color: #c44;");
        return;
    }

    auto& m = scene_->objects().at(selected_id_).material;
    Material nm = m;
    nm.texture_path   = path.toStdString();
    nm.texture_rgba   = img.rgba;
    nm.texture_width  = img.width;
    nm.texture_height = img.height;
    if (nm.pattern != Material::Pattern::Texture)
        nm.pattern = Material::Pattern::Texture;
    scene_->set_material(selected_id_, nm);
    ed_texture_->setText(path);
    cb_pattern_->setCurrentIndex(static_cast<int>(Material::Pattern::Texture));

    // Refresh thumbnail/info now that the material owns new pixels.
    QPixmap pm = make_thumbnail(nm, lbl_tex_thumb_->width());
    if (pm.isNull()) {
        lbl_tex_thumb_->setPixmap({});
        lbl_tex_thumb_->setText("—");
    } else {
        lbl_tex_thumb_->setPixmap(pm);
        lbl_tex_thumb_->setText({});
    }
    lbl_tex_info_->setText(format_tex_info(nm));

    dirty_ = true;
    Q_EMIT material_changed();
}

void MaterialEditor::on_texture_clear() {
    if (selected_id_.empty()) return;
    Material nm = scene_->objects().at(selected_id_).material;
    nm.texture_path.clear();
    nm.texture_rgba.clear();
    nm.texture_width  = 0;
    nm.texture_height = 0;
    if (nm.pattern == Material::Pattern::Texture)
        nm.pattern = Material::Pattern::Solid;
    scene_->set_material(selected_id_, nm);
    ed_texture_->clear();
    cb_pattern_->setCurrentIndex(static_cast<int>(nm.pattern));

    // Reset preview to the no-texture state.
    lbl_tex_thumb_->setPixmap({});
    lbl_tex_thumb_->setText("—");
    lbl_tex_info_->setText("(no texture)");

    dirty_ = true;
    Q_EMIT material_changed();
}

} // namespace frep::gui
