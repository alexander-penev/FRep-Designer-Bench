#pragma once
// gui/node_graph.hpp
//
// Node graph editor — visual editing of the FRepNode tree.
//
// Architecture:
//   PortItem     — a circle on the edge of a node (input or output), a link target
//   NodeItem     — a rectangle: header + parameters + ports
//   EdgeItem     — a Bezier curve connecting an output port → an input port
//   NodeGraphScene — QGraphicsScene; holds nodes/edges, build_tree()
//   NodeGraphEditor — QGraphicsView; context menu for adding nodes
//
// The Output node is special — it is the "root". build_tree() walks backward
// from its input and assembles the FRepNode::Ptr tree.

#include "gui/node_types.hpp"

#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QGraphicsView>

#include <map>
#include <memory>
#include <vector>

namespace frep {
class FRepNode;
namespace plugin { class PluginRegistry; }
}

namespace frep::gui {

class NodeItem;
class EdgeItem;

// ─────────────────────────────────────────────────────────────────────────────
// PortItem — a connector on the edge of a node.
// ─────────────────────────────────────────────────────────────────────────────
class PortItem : public QGraphicsItem {
public:
    enum { Type = UserType + 1 };
    int type() const override { return Type; }

    enum class Kind { Input, Output };

    PortItem(NodeItem* node, Kind kind, int index);

    QRectF boundingRect() const override;
    void   paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) override;

    NodeItem* node()  const { return node_; }
    Kind      kind()  const { return kind_; }
    int       index() const { return index_; }

    // Global position of the center — for drawing the edges.
    QPointF scene_center() const;

    // For input ports: the edge that connects here (or nullptr).
    EdgeItem* edge = nullptr;

private:
    NodeItem* node_;
    Kind      kind_;
    int       index_;
};

// ─────────────────────────────────────────────────────────────────────────────
// NodeItem — a visual node.
// ─────────────────────────────────────────────────────────────────────────────
class NodeItem : public QGraphicsItem {
public:
    enum { Type = UserType + 2 };
    int type() const override { return Type; }

    NodeItem(const NodeTypeInfo& info, int unique_id);

    QRectF boundingRect() const override;
    void   paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) override;

    const NodeTypeInfo& info() const { return info_; }
    int  node_id() const { return node_id_; }

    // The parameter values (param name → value).
    std::map<QString, float>& param_values() { return param_values_; }
    const std::map<QString, float>& param_values() const { return param_values_; }

    PortItem* input_port(int i)  const { return i < (int)inputs_.size() ? inputs_[i] : nullptr; }
    PortItem* output_port()      const { return output_; }
    int       input_count()      const { return (int)inputs_.size(); }

    // Called when the node is moved — updates the connected edges.
    void moved();

    // Optional second line under the title (e.g. an Instance's target id).
    void set_subtitle(const QString& s) { subtitle_ = s; update(); }
    const QString& subtitle() const { return subtitle_; }

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
    // Double-click on a parameter row pops up an inline editor.
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* e) override;

private:
    NodeTypeInfo               info_;
    int                        node_id_;
    QString                    subtitle_;
    std::map<QString, float>   param_values_;
    std::vector<PortItem*>     inputs_;
    PortItem*                  output_ = nullptr;

    static constexpr qreal kWidth      = 160.0;
    static constexpr qreal kHeaderH    = 26.0;
    static constexpr qreal kRowH       = 20.0;

    qreal body_height() const {
        // header + one row per parameter (at least 1 row for the empty ones)
        int rows = std::max<int>(1, (int)info_.params.size());
        return kHeaderH + rows * kRowH + 8.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// EdgeItem — a connection between two ports.
// ─────────────────────────────────────────────────────────────────────────────
class EdgeItem : public QGraphicsPathItem {
public:
    enum { Type = UserType + 3 };
    int type() const override { return Type; }

    EdgeItem(PortItem* from_output, PortItem* to_input);
    ~EdgeItem() override;

    PortItem* from() const { return from_; }
    PortItem* to()   const { return to_; }

    void update_path();

private:
    PortItem* from_;  // Output port
    PortItem* to_;    // Input port
};

// ─────────────────────────────────────────────────────────────────────────────
// NodeGraphScene — manages nodes and edges, assembles the FRepNode tree.
// ─────────────────────────────────────────────────────────────────────────────
class NodeGraphScene : public QGraphicsScene {
    Q_OBJECT
public:
    explicit NodeGraphScene(QObject* parent = nullptr);
    ~NodeGraphScene() override;

    // Adds a new node of type type_name at position pos.
    NodeItem* add_node(const QString& type_name, QPointF pos);

    // Optional PluginRegistry — when set, plugin-based primitive types
    // (Torus, Octahedron, Capsule, ...) become available in the context menu
    // and in add_node(). add_node() uses make_plugin_node_info() to derive
    // an NodeTypeInfo on the fly for plugin types not in node_catalog().
    void set_registry(const plugin::PluginRegistry* reg) { registry_ = reg; }
    const plugin::PluginRegistry* registry() const { return registry_; }

    // Tries to connect two ports. Returns true on success.
    // Rules: output → input, no cycles, an input may have only 1 edge.
    bool connect_ports(PortItem* a, PortItem* b);

    // Removes an edge.
    void remove_edge(EdgeItem* edge);

    // The Output node (the root of the tree).
    NodeItem* output_node() const { return output_node_; }

    // Assembles the FRepNode tree from the graph.
    // Returns nullptr if the output has no connected input or the graph is invalid.
    std::shared_ptr<FRepNode> build_tree() const;

    // Loads a scene from an FRepNode tree — auto-layout of the nodes.
    void load_from_tree(const std::shared_ptr<FRepNode>& root);

    // Removes all nodes/edges (except the output node, which is recreated).
    void clear_graph();

Q_SIGNALS:
    // Emitted on every change that affects the tree
    // (new node, new connection, changed parameter, move).
    void graph_changed();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* e) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* e) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* e) override;

private:
    std::vector<NodeItem*> nodes_;
    std::vector<EdgeItem*> edges_;
    NodeItem*              output_node_ = nullptr;
    int                    next_id_     = 1;

    const plugin::PluginRegistry* registry_ = nullptr;
    // Cached plugin NodeTypeInfo by type_name — NodeItem stores a const&
    // to its info, so the cache must outlive every NodeItem. std::map
    // gives reference stability across insertion, which a flat vector
    // would not.
    mutable std::map<QString, NodeTypeInfo> plugin_info_cache_;

    // Drag-to-connect state.
    PortItem*              drag_from_   = nullptr;
    QGraphicsPathItem*     drag_edge_   = nullptr;

    PortItem* port_at(QPointF scene_pos) const;

    // Recursive assembly — from a node down through its inputs.
    std::shared_ptr<FRepNode> build_subtree(NodeItem* node) const;

    friend class NodeItem;
};

// ─────────────────────────────────────────────────────────────────────────────
// NodeGraphEditor — a view widget with a context menu.
// ─────────────────────────────────────────────────────────────────────────────
class NodeGraphEditor : public QGraphicsView {
    Q_OBJECT
public:
    explicit NodeGraphEditor(QWidget* parent = nullptr);

    NodeGraphScene* graph_scene() const { return scene_; }

    // Fit the entire scene contents into the view, with a small margin.
    // Used by the "Fit" button in the surrounding panel. Doesn't touch
    // scene contents — just the view transform.
    void fit_all();

protected:
    void contextMenuEvent(QContextMenuEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    NodeGraphScene* scene_;
};

} // namespace frep::gui
