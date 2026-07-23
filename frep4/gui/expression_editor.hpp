#pragma once
// gui/expression_editor.hpp
//
// Side-panel widget for adding CustomExprNode objects to the scene.
// Provides:
//   - Multi-line text editor for the expression
//   - "Insert sample" dropdown with named pre-canned expressions
//   - Live parse-error feedback below the editor
//   - "Add to scene" button that creates the node and emits
//     scene_changed so the viewport re-renders
//
// The editor uses frep::expr::parse() to validate syntax in real time
// (debounced ~250 ms after the last keystroke), so users see parse
// errors with their position before they commit to adding to the scene.

#include <QWidget>
#include <QString>

class QPlainTextEdit;
class QPushButton;
class QComboBox;
class QLabel;
class QTimer;
class QLineEdit;

namespace frep {
class SceneGraph;
namespace undo { class UndoStack; }
}

namespace frep::gui {

class ExpressionEditor : public QWidget {
    Q_OBJECT
public:
    explicit ExpressionEditor(SceneGraph* scene, QWidget* parent = nullptr);

    void set_undo_stack(undo::UndoStack* s) { undo_stack_ = s; }

Q_SIGNALS:
    // Emitted when an expression is committed and an object is added
    // to the scene. MainWindow connects this to refresh inspectors +
    // trigger a re-render.
    void scene_changed();

private Q_SLOTS:
    // Debounced syntax check — wired to QPlainTextEdit::textChanged
    // via a 250 ms QTimer to avoid re-parsing on every keystroke.
    void on_text_changed();
    void on_validate();
    void on_insert_sample(int index);
    void on_add();

private:
    SceneGraph*       scene_      = nullptr;
    undo::UndoStack*  undo_stack_ = nullptr;

    QPlainTextEdit*   editor_     = nullptr;
    QLineEdit*        id_edit_    = nullptr;
    QComboBox*        sample_box_ = nullptr;
    QLabel*           status_lbl_ = nullptr;
    QPushButton*      add_btn_    = nullptr;
    QTimer*           debounce_   = nullptr;
};

} // namespace frep::gui
