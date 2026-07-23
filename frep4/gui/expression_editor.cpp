// gui/expression_editor.cpp

#include "gui/expression_editor.hpp"

#include "core/frep/custom_expr.hpp"
#include "core/frep/expr_ast.hpp"
#include "core/frep/scene.hpp"
#include "core/undo/undo_stack.hpp"

#include <QComboBox>
#include <QFont>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include <memory>

namespace frep::gui {

// A few hand-picked sample expressions, each demonstrating a different
// surface family. The first is selected by default in the combo box and
// pre-fills the editor on widget construction.
namespace {
struct Sample { const char* name; const char* expr; };
const std::array<Sample, 6> kSamples = {
    Sample{"Unit sphere",        "sqrt(x*x + y*y + z*z) - 1.0"},
    Sample{"Sphere (squared)",   "x*x + y*y + z*z - 1.0"},
    Sample{"Octahedron",         "abs(x) + abs(y) + abs(z) - 1.0"},
    Sample{"Gyroid",
        "sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)"},
    Sample{"Torus (approx)",
        "sqrt(pow(sqrt(x*x + z*z) - 0.7, 2) + y*y) - 0.25"},
    Sample{"Schwarz P",
        "cos(x) + cos(y) + cos(z)"},
};
} // anon

ExpressionEditor::ExpressionEditor(SceneGraph* scene, QWidget* parent)
    : QWidget(parent), scene_(scene)
{
    auto* layout = new QVBoxLayout(this);

    auto* hint = new QLabel(
        "<b>Custom Expression</b><br>"
        "Define an SDF as an analytic expression in <code>x, y, z</code>. "
        "Supported: <code>+ - * /</code>, "
        "<code>sin cos tan sqrt abs exp log floor ceil pow min max</code>, "
        "constants <code>pi e</code>.");
    hint->setWordWrap(true);
    layout->addWidget(hint);

    // Sample picker — inserts a pre-canned expression into the editor.
    sample_box_ = new QComboBox(this);
    for (const auto& s : kSamples) sample_box_->addItem(s.name);
    layout->addWidget(sample_box_);
    connect(sample_box_,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ExpressionEditor::on_insert_sample);

    // Object id field — required so the scene can track + select it.
    auto* form = new QFormLayout;
    id_edit_ = new QLineEdit(this);
    id_edit_->setText("expr_obj");
    form->addRow("Object id:", id_edit_);
    layout->addLayout(form);

    // Main text editor for the expression. Monospace font helps
    // legibility for math-heavy text.
    editor_ = new QPlainTextEdit(this);
    QFont mono("Monospace");
    mono.setStyleHint(QFont::TypeWriter);
    editor_->setFont(mono);
    editor_->setMinimumHeight(80);
    editor_->setPlainText(kSamples[0].expr);
    layout->addWidget(editor_);

    // Status / parse-error label.
    status_lbl_ = new QLabel("✓ valid", this);
    status_lbl_->setStyleSheet("color: #2a8;");
    status_lbl_->setWordWrap(true);
    layout->addWidget(status_lbl_);

    add_btn_ = new QPushButton("Add to scene", this);
    layout->addWidget(add_btn_);
    connect(add_btn_, &QPushButton::clicked, this, &ExpressionEditor::on_add);

    // Debounce the parse on keystrokes — re-parsing on every character
    // is wasteful and makes typing jittery for long expressions.
    debounce_ = new QTimer(this);
    debounce_->setSingleShot(true);
    debounce_->setInterval(250);
    connect(debounce_, &QTimer::timeout, this, &ExpressionEditor::on_validate);
    connect(editor_,   &QPlainTextEdit::textChanged,
            this, &ExpressionEditor::on_text_changed);

    layout->addStretch(1);

    on_validate();  // initial pass
}

void ExpressionEditor::on_text_changed() {
    debounce_->start();
}

void ExpressionEditor::on_validate() {
    auto src = editor_->toPlainText().trimmed().toStdString();
    if (src.empty()) {
        status_lbl_->setText("(empty)");
        status_lbl_->setStyleSheet("color: #888;");
        add_btn_->setEnabled(false);
        return;
    }
    try {
        auto ast = frep::expr::parse(src);
        (void)ast;
        status_lbl_->setText("✓ valid");
        status_lbl_->setStyleSheet("color: #2a8;");
        add_btn_->setEnabled(true);
    } catch (const frep::expr::ParseError& e) {
        // Show "✗ msg @ col N" so the user can locate the error. The
        // parser stores 0-based column; we display 1-based to match
        // editor conventions.
        QString msg;
        if (e.column >= 0)
            msg = QString("✗ %1 (col %2)").arg(e.what()).arg(e.column + 1);
        else
            msg = QString("✗ %1").arg(e.what());
        status_lbl_->setText(msg);
        status_lbl_->setStyleSheet("color: #c44;");
        add_btn_->setEnabled(false);
    } catch (const std::exception& e) {
        status_lbl_->setText(QString("✗ %1").arg(e.what()));
        status_lbl_->setStyleSheet("color: #c44;");
        add_btn_->setEnabled(false);
    }
}

void ExpressionEditor::on_insert_sample(int index) {
    if (index < 0 || index >= static_cast<int>(kSamples.size())) return;
    editor_->setPlainText(kSamples[index].expr);
    // Trigger immediate validation so the user sees the new sample is
    // accepted before they can finish reading the label.
    on_validate();
}

void ExpressionEditor::on_add() {
    auto src = editor_->toPlainText().trimmed().toStdString();
    if (src.empty()) return;
    auto id = id_edit_->text().toStdString();
    if (id.empty()) id = "expr_obj";

    // Ensure the id is unique — if there's already an object with this
    // id, append a counter suffix until we find a free name. This keeps
    // the user from accidentally overwriting an existing object.
    auto& objs = scene_->objects();
    if (objs.count(id)) {
        int n = 1;
        while (objs.count(id + "_" + std::to_string(n))) ++n;
        id = id + "_" + std::to_string(n);
    }

    // Parse once here so we can fail gracefully without adding a broken
    // node to the scene; on_validate() should already have caught this
    // earlier, but a race could occur if the user clicks while the
    // debounce timer is pending.
    try {
        (void)frep::expr::parse(src);
    } catch (const std::exception& e) {
        status_lbl_->setText(QString("Can't add — parse error: %1").arg(e.what()));
        status_lbl_->setStyleSheet("color: #c44;");
        return;
    }

    auto node = std::make_shared<CustomExprNode>(src, id);
    Material mat{{0.8f, 0.5f, 0.9f}};
    mat.roughness = 0.4f;

    if (undo_stack_) {
        undo_stack_->push_apply(
            std::make_unique<undo::AddObjectCommand>(
                *scene_, id, node, mat));
    } else {
        scene_->add_object(node, mat);
    }

    status_lbl_->setText(QString("✓ added as '%1'").arg(QString::fromStdString(id)));
    status_lbl_->setStyleSheet("color: #2a8;");
    Q_EMIT scene_changed();
}

} // namespace frep::gui
