#pragma once

// gui/checkable_combo_box.hpp
//
// A QComboBox whose dropdown items carry checkboxes, so several can be selected
// at once (Qt's stock combobox is single-select). Used for the Render-tab path
// picker: tick one path to render it full-frame, tick several to split the
// frame across them. The closed combo shows a summary of what's ticked.

#include <QComboBox>
#include <QStandardItemModel>
#include <QString>

#include <functional>
#include <vector>

namespace frep::gui {

class CheckableComboBox : public QComboBox {
    Q_OBJECT
public:
    explicit CheckableComboBox(QWidget* parent = nullptr);

    // Add a checkable entry with arbitrary user data (e.g. a path index).
    void add_checkable_item(const QString& text, const QVariant& data,
                            bool checked = false);

    // Indices (insertion order) of the currently checked items.
    std::vector<int> checked_indices() const;
    // user-data of checked items.
    std::vector<QVariant> checked_data() const;

    void set_checked(int index, bool checked);
    bool is_checked(int index) const;

    // Called whenever the set of checked items changes (user toggled one).
    std::function<void()> on_changed;

protected:
    // Keep the popup open while the user toggles multiple boxes.
    bool eventFilter(QObject* obj, QEvent* ev) override;

private:
    void refresh_summary_text();
    QStandardItemModel* model_ = nullptr;
    bool updating_ = false;
};

}  // namespace frep::gui
