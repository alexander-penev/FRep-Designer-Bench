// gui/checkable_combo_box.cpp
#include "gui/checkable_combo_box.hpp"

#include <QEvent>
#include <QLineEdit>
#include <QAbstractItemView>
#include <QMouseEvent>
#include <QStandardItem>

namespace frep::gui {

CheckableComboBox::CheckableComboBox(QWidget* parent) : QComboBox(parent) {
    model_ = new QStandardItemModel(this);
    setModel(model_);
    // Editable purely so we can show a custom summary in the closed box; the
    // line edit is read-only so the user can't type into it.
    setEditable(true);
    lineEdit()->setReadOnly(true);
    lineEdit()->setFocusPolicy(Qt::NoFocus);
    // Intercept clicks on the popup list so a toggle doesn't close it.
    view()->viewport()->installEventFilter(this);

    connect(model_, &QStandardItemModel::itemChanged, this,
            [this](QStandardItem*) {
                if (updating_) return;
                refresh_summary_text();
                if (on_changed) on_changed();
            });
}

void CheckableComboBox::add_checkable_item(const QString& text,
                                           const QVariant& data, bool checked) {
    auto* item = new QStandardItem(text);
    item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    item->setData(checked ? Qt::Checked : Qt::Unchecked, Qt::CheckStateRole);
    item->setData(data, Qt::UserRole);
    model_->appendRow(item);
    refresh_summary_text();
}

std::vector<int> CheckableComboBox::checked_indices() const {
    std::vector<int> out;
    for (int i = 0; i < model_->rowCount(); ++i)
        if (model_->item(i)->checkState() == Qt::Checked) out.push_back(i);
    return out;
}

std::vector<QVariant> CheckableComboBox::checked_data() const {
    std::vector<QVariant> out;
    for (int i = 0; i < model_->rowCount(); ++i)
        if (model_->item(i)->checkState() == Qt::Checked)
            out.push_back(model_->item(i)->data(Qt::UserRole));
    return out;
}

void CheckableComboBox::set_checked(int index, bool checked) {
    if (index < 0 || index >= model_->rowCount()) return;
    updating_ = true;
    model_->item(index)->setData(checked ? Qt::Checked : Qt::Unchecked,
                                 Qt::CheckStateRole);
    updating_ = false;
    refresh_summary_text();
}

bool CheckableComboBox::is_checked(int index) const {
    if (index < 0 || index >= model_->rowCount()) return false;
    return model_->item(index)->checkState() == Qt::Checked;
}

void CheckableComboBox::refresh_summary_text() {
    // Compose "a, b, c" of the checked labels, or "(none)".
    QStringList on;
    for (int i = 0; i < model_->rowCount(); ++i)
        if (model_->item(i)->checkState() == Qt::Checked)
            on << model_->item(i)->text();
    QString summary = on.isEmpty() ? QStringLiteral("(none)") : on.join(", ");
    if (lineEdit()) lineEdit()->setText(summary);
    setToolTip(summary);
}

bool CheckableComboBox::eventFilter(QObject* obj, QEvent* ev) {
    // Toggle the item under a left click and keep the popup open.
    if (ev->type() == QEvent::MouseButtonRelease) {
        auto* me = static_cast<QMouseEvent*>(ev);
        QModelIndex idx = view()->indexAt(me->pos());
        if (idx.isValid()) {
            auto* item = model_->item(idx.row());
            item->setData(item->checkState() == Qt::Checked ? Qt::Unchecked
                                                             : Qt::Checked,
                          Qt::CheckStateRole);
            return true;  // swallow — don't let the combo close
        }
    }
    return QComboBox::eventFilter(obj, ev);
}

}  // namespace frep::gui
