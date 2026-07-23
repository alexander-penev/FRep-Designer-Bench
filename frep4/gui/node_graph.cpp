// gui/node_graph.cpp

#include "node_graph.hpp"

#include "core/frep/custom_expr.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/instance.hpp"
#include "core/plugin/plugin_api.hpp"

#include <QContextMenuEvent>
#include <QGraphicsSceneMouseEvent>
#include <QInputDialog>
#include <QMenu>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <functional>

namespace frep::gui {

// ═════════════════════════════════════════════════════════════════════════════
// PortItem
// ═════════════════════════════════════════════════════════════════════════════
static constexpr qreal kPortR = 6.0;

PortItem::PortItem(NodeItem* node, Kind kind, int index)
    : QGraphicsItem(node), node_(node), kind_(kind), index_(index)
{
    setZValue(2);
}

QRectF PortItem::boundingRect() const {
    return QRectF(-kPortR, -kPortR, 2 * kPortR, 2 * kPortR);
}

void PortItem::paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) {
    p->setRenderHint(QPainter::Antialiasing);
    QColor fill = (kind_ == Kind::Input) ? QColor(0xff, 0xc1, 0x07)
                                         : QColor(0x8b, 0xc3, 0x4a);
    // if an input port has a connection — more saturated
    if (kind_ == Kind::Input && edge) fill = fill.darker(130);
    p->setBrush(fill);
    p->setPen(QPen(Qt::black, 1.0));
    p->drawEllipse(QPointF(0, 0), kPortR, kPortR);
}

QPointF PortItem::scene_center() const {
    return mapToScene(QPointF(0, 0));
}

// ═════════════════════════════════════════════════════════════════════════════
// NodeItem
// ═════════════════════════════════════════════════════════════════════════════
NodeItem::NodeItem(const NodeTypeInfo& info, int unique_id)
    : info_(info), node_id_(unique_id)
{
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    setZValue(1);

    // Initialize the parameters with default values.
    for (const auto& pr : info_.params)
        param_values_[pr.name] = pr.default_value;

    // Input ports — distributed along the left edge.
    for (int i = 0; i < info_.input_count; ++i) {
        auto* port = new PortItem(this, PortItem::Kind::Input, i);
        qreal y = kHeaderH + (i + 0.5) * (body_height() - kHeaderH)
                  / std::max(1, info_.input_count);
        port->setPos(0, y);
        inputs_.push_back(port);
    }
    // Output port — right edge, at the middle of the header.
    // The Output node (category Output) has no output.
    if (info_.category != NodeCategory::Output) {
        output_ = new PortItem(this, PortItem::Kind::Output, 0);
        output_->setPos(kWidth, kHeaderH / 2);
    }
}

QRectF NodeItem::boundingRect() const {
    return QRectF(-2, -2, kWidth + 4, body_height() + 4);
}

void NodeItem::paint(QPainter* p, const QStyleOptionGraphicsItem*, QWidget*) {
    p->setRenderHint(QPainter::Antialiasing);

    QRectF body(0, 0, kWidth, body_height());

    // Body
    p->setBrush(QColor(0x3a, 0x3a, 0x3a));
    p->setPen(isSelected() ? QPen(QColor(0xff, 0xff, 0xff), 2.0)
                           : QPen(QColor(0x20, 0x20, 0x20), 1.0));
    p->drawRoundedRect(body, 6, 6);

    // Header
    QRectF header(0, 0, kWidth, kHeaderH);
    QPainterPath hp;
    hp.addRoundedRect(header, 6, 6);
    // clip the bottom corners of the header so they are square
    hp.addRect(0, kHeaderH - 6, kWidth, 6);
    p->setBrush(info_.header_color());
    p->setPen(Qt::NoPen);
    p->drawPath(hp.simplified());

    // Title
    p->setPen(Qt::white);
    QFont f = p->font();
    f.setBold(true);
    p->setFont(f);
    p->drawText(header.adjusted(8, 0, -8, 0),
                Qt::AlignVCenter | Qt::AlignLeft, info_.display);

    // Optional subtitle (e.g. an Instance's "→ target_id"), just below the header.
    qreal params_top = kHeaderH + 4;
    if (!subtitle_.isEmpty()) {
        f.setBold(false);
        p->setFont(f);
        p->setPen(QColor(0xff, 0xd0, 0xe8));
        p->drawText(QRectF(8, kHeaderH + 2, kWidth - 16, kRowH),
                    Qt::AlignVCenter | Qt::AlignLeft, subtitle_);
        params_top += kRowH;
    }

    // Parameters
    f.setBold(false);
    p->setFont(f);
    p->setPen(QColor(0xcc, 0xcc, 0xcc));
    qreal y = params_top;
    for (const auto& pr : info_.params) {
        QRectF row(8, y, kWidth - 16, kRowH);
        float val = param_values_.at(pr.name);
        p->drawText(row, Qt::AlignVCenter | Qt::AlignLeft, pr.name);
        // Value drawn with a faint dotted underline as a visual hint
        // that it is editable on double-click.
        QString val_str = QString::number(val, 'g', 3);
        p->drawText(row, Qt::AlignVCenter | Qt::AlignRight, val_str);
        QFontMetrics fm(p->font());
        int text_w = fm.horizontalAdvance(val_str);
        qreal underline_y = y + kRowH - 4;
        QPen old = p->pen();
        QPen dotted(QColor(0x66, 0x66, 0x66), 1.0, Qt::DotLine);
        p->setPen(dotted);
        p->drawLine(QPointF(kWidth - 8 - text_w, underline_y),
                    QPointF(kWidth - 8,         underline_y));
        p->setPen(old);
        y += kRowH;
    }
    // If there are no parameters — show a category note
    if (info_.params.empty() && info_.category != NodeCategory::Output) {
        p->setPen(QColor(0x77, 0x77, 0x77));
        p->drawText(QRectF(8, kHeaderH + 4, kWidth - 16, kRowH),
                    Qt::AlignVCenter | Qt::AlignLeft, "(no parameters)");
    }
    if (info_.category == NodeCategory::Output) {
        p->setPen(QColor(0xdd, 0xdd, 0xdd));
        p->drawText(QRectF(8, kHeaderH + 4, kWidth - 16, kRowH),
                    Qt::AlignVCenter | Qt::AlignLeft, "← final model");
    }
}

void NodeItem::moved() {
    // Update all connected edges.
    for (auto* port : inputs_)
        if (port->edge) port->edge->update_path();
    // scene() may be nullptr if setPos is called before addItem.
    if (output_ && scene()) {
        // An output port may have multiple outgoing edges — walk the scene.
        for (auto* it : scene()->items()) {
            if (auto* e = qgraphicsitem_cast<EdgeItem*>(it))
                if (e->from() == output_) e->update_path();
        }
    }
}

QVariant NodeItem::itemChange(GraphicsItemChange change, const QVariant& value) {
    if (change == ItemPositionHasChanged) {
        moved();
        // scene() is nullptr while the item is not yet added — skip the signal.
        if (auto* gs = static_cast<NodeGraphScene*>(scene()))
            Q_EMIT gs->graph_changed();
    }
    return QGraphicsItem::itemChange(change, value);
}

void NodeItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* e) {
    // Identify which parameter row was clicked. Rows start at kHeaderH+4 and
    // are kRowH tall; the node body is kWidth wide.
    if (info_.params.empty()) {
        QGraphicsItem::mouseDoubleClickEvent(e);
        return;
    }
    qreal local_y = e->pos().y();
    qreal first_row_y = kHeaderH + 4;
    if (local_y < first_row_y || local_y >= first_row_y + kRowH * info_.params.size()) {
        QGraphicsItem::mouseDoubleClickEvent(e);
        return;
    }
    int row = static_cast<int>((local_y - first_row_y) / kRowH);
    if (row < 0 || row >= static_cast<int>(info_.params.size())) {
        QGraphicsItem::mouseDoubleClickEvent(e);
        return;
    }

    const NodeParam& param = info_.params[row];
    float current = param_values_.at(param.name);

    // Inline editor: a QInputDialog with the parameter's min/max range.
    // Cheap and effective for a PoC; richer inline editors (slider widget
    // inside the QGraphicsItem) could come later.
    bool ok = false;
    // QInputDialog::getDouble works with int decimals; 4 is plenty for our
    // typical slider ranges (0.01 precision over [-3.14, 3.14] etc.).
    double new_val = QInputDialog::getDouble(
        nullptr,
        QString("Edit %1").arg(info_.display),
        QString("%1 (range: %2 to %3)")
            .arg(param.name)
            .arg(param.min_value)
            .arg(param.max_value),
        static_cast<double>(current),
        static_cast<double>(param.min_value),
        static_cast<double>(param.max_value),
        4,
        &ok);
    if (!ok) {
        e->accept();
        return;
    }

    param_values_[param.name] = static_cast<float>(new_val);
    update();   // repaint with new value
    if (auto* gs = static_cast<NodeGraphScene*>(scene()))
        Q_EMIT gs->graph_changed();
    e->accept();
}

// ═════════════════════════════════════════════════════════════════════════════
// EdgeItem
// ═════════════════════════════════════════════════════════════════════════════
EdgeItem::EdgeItem(PortItem* from_output, PortItem* to_input)
    : from_(from_output), to_(to_input)
{
    setZValue(0);
    setPen(QPen(QColor(0xdd, 0xdd, 0xdd), 2.0));
    to_input->edge = this;
    update_path();
}

EdgeItem::~EdgeItem() {
    // Detach from the input port.
    if (to_ && to_->edge == this)
        to_->edge = nullptr;
}

void EdgeItem::update_path() {
    QPointF a = from_->scene_center();
    QPointF b = to_->scene_center();
    // Bezier — control points pulled out horizontally.
    qreal dx = std::max<qreal>(40.0, std::abs(b.x() - a.x()) * 0.5);
    QPainterPath path(a);
    path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b);
    setPath(path);
}

// ═════════════════════════════════════════════════════════════════════════════
// NodeGraphScene
// ═════════════════════════════════════════════════════════════════════════════
NodeGraphScene::NodeGraphScene(QObject* parent)
    : QGraphicsScene(parent)
{
    setSceneRect(-2000, -2000, 4000, 4000);
    setBackgroundBrush(QColor(0x2b, 0x2b, 0x2b));

    // Create the special Output node.
    static const NodeTypeInfo output_info{
        "Output", "Output", NodeCategory::Output, 1, {}
    };
    output_node_ = new NodeItem(output_info, 0);
    addItem(output_node_);          // addItem first — for a valid scene()
    output_node_->setPos(300, 0);
}

NodeGraphScene::~NodeGraphScene() {
    // Delete edges first, while their endpoint PortItems are still alive.
    // ~EdgeItem dereferences to_ (an input PortItem owned by a NodeItem); if Qt's
    // automatic ~QGraphicsScene cleared items in an order that freed a NodeItem
    // before its EdgeItem, that deref would be a use-after-free. Clearing edges
    // explicitly here removes that ordering hazard.
    for (auto* e : edges_) { removeItem(e); delete e; }
    edges_.clear();
}

NodeItem* NodeGraphScene::add_node(const QString& type_name, QPointF pos) {
    const NodeTypeInfo* info = find_node_type(type_name);

    // Plugin fallback: if the type isn't in the built-in catalog, ask the
    // registry. The derived NodeTypeInfo is cached in plugin_info_cache_
    // because NodeItem stores a const& to it.
    if (!info && registry_) {
        auto it = plugin_info_cache_.find(type_name);
        if (it == plugin_info_cache_.end()) {
            auto pi = make_plugin_node_info_by_type_name(*registry_, type_name);
            if (pi.display.isEmpty()) return nullptr;  // not found in registry
            it = plugin_info_cache_.emplace(type_name, std::move(pi)).first;
        }
        info = &it->second;
    }
    if (!info) return nullptr;

    auto* node = new NodeItem(*info, next_id_++);
    addItem(node);          // add first — scene() becomes valid
    node->setPos(pos);      // then position — itemChange sees scene()
    nodes_.push_back(node);
    Q_EMIT graph_changed();
    return node;
}

bool NodeGraphScene::connect_ports(PortItem* a, PortItem* b) {
    if (!a || !b) return false;

    // Order them: output → input.
    PortItem* out = nullptr;
    PortItem* in  = nullptr;
    if (a->kind() == PortItem::Kind::Output && b->kind() == PortItem::Kind::Input) {
        out = a; in = b;
    } else if (a->kind() == PortItem::Kind::Input && b->kind() == PortItem::Kind::Output) {
        out = b; in = a;
    } else {
        return false;  // cannot do input→input or output→output
    }

    // A node cannot connect to itself.
    if (out->node() == in->node()) return false;

    // Cycle detection: if out->node() is reachable from in->node() through the
    // inputs, creating this connection would form a cycle.
    std::function<bool(NodeItem*, NodeItem*)> reaches =
        [&](NodeItem* from, NodeItem* target) -> bool {
        if (from == target) return true;
        for (int i = 0; i < from->input_count(); ++i) {
            PortItem* ip = from->input_port(i);
            if (ip->edge) {
                NodeItem* upstream = ip->edge->from()->node();
                if (reaches(upstream, target)) return true;
            }
        }
        return false;
    };
    if (reaches(out->node(), in->node())) return false;

    // An input port may have only 1 connection — remove the old one.
    if (in->edge)
        remove_edge(in->edge);

    auto* edge = new EdgeItem(out, in);
    addItem(edge);
    edges_.push_back(edge);
    Q_EMIT graph_changed();
    return true;
}

void NodeGraphScene::remove_edge(EdgeItem* edge) {
    if (!edge) return;
    edges_.erase(std::remove(edges_.begin(), edges_.end(), edge), edges_.end());
    removeItem(edge);
    delete edge;
    Q_EMIT graph_changed();
}

PortItem* NodeGraphScene::port_at(QPointF scene_pos) const {
    // Look for a port under the cursor (with a small tolerance).
    for (auto* it : items(QRectF(scene_pos - QPointF(8, 8), QSizeF(16, 16)))) {
        if (auto* port = qgraphicsitem_cast<PortItem*>(it))
            return port;
    }
    return nullptr;
}

void NodeGraphScene::mousePressEvent(QGraphicsSceneMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        PortItem* port = port_at(e->scenePos());
        if (port) {
            // Start drag-to-connect.
            drag_from_ = port;
            drag_edge_ = new QGraphicsPathItem();
            drag_edge_->setPen(QPen(QColor(0xff, 0xff, 0x00, 160), 2.0,
                                    Qt::DashLine));
            drag_edge_->setZValue(3);
            addItem(drag_edge_);
            e->accept();
            return;
        }
    }
    QGraphicsScene::mousePressEvent(e);
}

void NodeGraphScene::mouseMoveEvent(QGraphicsSceneMouseEvent* e) {
    if (drag_from_ && drag_edge_) {
        QPointF a = drag_from_->scene_center();
        QPointF b = e->scenePos();
        qreal dx = std::max<qreal>(40.0, std::abs(b.x() - a.x()) * 0.5);
        QPainterPath path(a);
        // the control point direction depends on whether we start from output or input
        qreal sign = (drag_from_->kind() == PortItem::Kind::Output) ? 1.0 : -1.0;
        path.cubicTo(a + QPointF(sign * dx, 0), b - QPointF(sign * dx, 0), b);
        drag_edge_->setPath(path);
        e->accept();
        return;
    }
    QGraphicsScene::mouseMoveEvent(e);
}

void NodeGraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* e) {
    if (drag_from_ && drag_edge_) {
        PortItem* target = port_at(e->scenePos());
        if (target && target != drag_from_)
            connect_ports(drag_from_, target);

        removeItem(drag_edge_);
        delete drag_edge_;
        drag_edge_ = nullptr;
        drag_from_ = nullptr;
        e->accept();
        return;
    }
    QGraphicsScene::mouseReleaseEvent(e);
}

// ─── build_tree ─────────────────────────────────────────────────────────────
std::shared_ptr<FRepNode>
NodeGraphScene::build_subtree(NodeItem* node) const {
    if (!node) return nullptr;
    const QString& type = node->info().type_name;
    const auto& pv = node->param_values();
    auto P = [&](const char* k) -> float {
        auto it = pv.find(k);
        return it != pv.end() ? it->second : 0.0f;
    };
    std::string id = type.toStdString() + "_" + std::to_string(node->node_id());

    // Helper: gets the child from input port i (or nullptr).
    auto child = [&](int i) -> std::shared_ptr<FRepNode> {
        PortItem* ip = node->input_port(i);
        if (!ip || !ip->edge) return nullptr;
        return build_subtree(ip->edge->from()->node());
    };

    // ── Primitives ────────────────────────────────────────────────────────────
    if (type == "Sphere")
        return std::make_shared<SphereNode>(P("r"), id);
    if (type == "Box")
        return std::make_shared<BoxNode>(P("hx"), P("hy"), P("hz"), id);
    if (type == "Plane")
        return std::make_shared<PlaneNode>(P("nx"), P("ny"), P("nz"), P("d"), id);

    // ── Operations ────────────────────────────────────────────────────────────
    if (type == "Union") {
        auto a = child(0), b = child(1);
        if (!a || !b) return nullptr;
        return std::make_shared<UnionNode>(a, b, id);
    }
    if (type == "Intersection") {
        auto a = child(0), b = child(1);
        if (!a || !b) return nullptr;
        return std::make_shared<IntersectionNode>(a, b, id);
    }
    if (type == "Difference") {
        auto a = child(0), b = child(1);
        if (!a || !b) return nullptr;
        return std::make_shared<DifferenceNode>(a, b, id);
    }
    if (type == "SmoothUnion") {
        auto a = child(0), b = child(1);
        if (!a || !b) return nullptr;
        return std::make_shared<SmoothUnionNode>(a, b, P("k"), id);
    }
    if (type == "Negate") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<NegateNode>(a, id);
    }

    // ── Transforms ────────────────────────────────────────────────────────────
    if (type == "Translate") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<TranslateNode>(a, P("tx"), P("ty"), P("tz"), id);
    }
    if (type == "Scale") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<ScaleNode>(a, P("sx"), P("sy"), P("sz"), id);
    }
    if (type == "RotateX") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<RotateXNode>(a, P("a"), id);
    }
    if (type == "RotateY") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<RotateYNode>(a, P("a"), id);
    }
    if (type == "RotateZ") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<RotateZNode>(a, P("a"), id);
    }

    // ── Deformations ──────────────────────────────────────────────────────────
    if (type == "TwistY") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<TwistYNode>(a, P("k"), id);
    }
    if (type == "BendXY") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<BendXYNode>(a, P("k"), id);
    }
    if (type == "TaperY") {
        auto a = child(0);
        if (!a) return nullptr;
        return std::make_shared<TaperYNode>(a, P("t"), P("h"), id);
    }

    // ── Plugin-based primitives ───────────────────────────────────────────────
    // For types not in the built-in switch above, ask the registry. The
    // plugin's param_names list is the canonical parameter order; we read
    // them from the node's stored params (which the user may have edited)
    // and hand them to the plugin's create().
    if (registry_) {
        if (const auto* slot = registry_->find_primitive_by_type_name(
                                  type.toStdString())) {
            std::vector<float> ordered;
            ordered.reserve(slot->param_names.size());
            for (std::size_t i = 0; i < slot->param_names.size(); ++i) {
                QString key = QString::fromStdString(
                                std::string(slot->param_names[i]));
                float def = i < slot->param_defaults.size()
                            ? slot->param_defaults[i] : 0.0f;
                auto it = node->param_values().find(key);
                ordered.push_back(it != node->param_values().end()
                                  ? it->second : def);
            }
            return slot->create(ordered, id);
        }
    }

    return nullptr;
}

std::shared_ptr<FRepNode> NodeGraphScene::build_tree() const {
    // The Output node has 1 input — that is the root of the tree.
    if (!output_node_) return nullptr;
    PortItem* in = output_node_->input_port(0);
    if (!in || !in->edge) return nullptr;
    return build_subtree(in->edge->from()->node());
}

// ─── load_from_tree — auto-layout ───────────────────────────────────────────
void NodeGraphScene::load_from_tree(const std::shared_ptr<FRepNode>& root) {
    clear_graph();
    if (!root) return;

    // Recursive layout: depth → X column, row within a depth → Y.
    // Build the nodes bottom-up, then connect them.
    std::map<int, int> depth_count;  // how many nodes are already at a given depth

    std::function<NodeItem*(const std::shared_ptr<FRepNode>&, int)> rec =
        [&](const std::shared_ptr<FRepNode>& fnode, int depth) -> NodeItem* {
        if (!fnode) return nullptr;

        QString type = QString::fromStdString(fnode->type_name());
        // CustomExpr and plugin types have no catalog entry — skip them.
        const NodeTypeInfo* info = find_node_type(type);
        if (!info) return nullptr;

        // Position: X decreases with depth (root on the right).
        int row = depth_count[depth]++;
        qreal x = -depth * 220.0;
        qreal y = row * 140.0;

        auto* node = add_node(type, QPointF(x, y));
        if (!node) return nullptr;

        // Instance references another object's geometry by id. We must NOT
        // recurse into its shared child — that would duplicate the whole target
        // subtree in the graph (and loop forever on a cyclic reference). Show it
        // as a terminal node whose label carries the target id, so the graph
        // reflects that an instance exists and what it points at.
        if (fnode->kind == NodeKind::Instance) {
            const auto* in = static_cast<const InstanceNode*>(fnode.get());
            node->set_subtitle(QString("→ %1")
                .arg(QString::fromStdString(in->target_id())));
            return node;
        }

        // Parameters.
        for (auto& [k, v] : fnode->params) {
            QString qk = QString::fromStdString(k);
            if (node->param_values().count(qk))
                node->param_values()[qk] = v;
        }

        // Children — connect the input ports.
        for (std::size_t i = 0; i < fnode->children.size() &&
                                (int)i < node->input_count(); ++i) {
            NodeItem* child = rec(fnode->children[i], depth + 1);
            if (child && child->output_port())
                connect_ports(child->output_port(), node->input_port((int)i));
        }
        return node;
    };

    NodeItem* root_node = rec(root, 1);
    if (root_node && root_node->output_port())
        connect_ports(root_node->output_port(), output_node_->input_port(0));

    Q_EMIT graph_changed();
}

void NodeGraphScene::clear_graph() {
    // Remove all edges.
    for (auto* e : edges_) {
        removeItem(e);
        delete e;
    }
    edges_.clear();

    // Remove all nodes except the output.
    for (auto* n : nodes_) {
        removeItem(n);
        delete n;
    }
    nodes_.clear();
    next_id_ = 1;

    // The Output node stays; its input connection is cleared (already removed with edges_).
    Q_EMIT graph_changed();
}

// ═════════════════════════════════════════════════════════════════════════════
// NodeGraphEditor
// ═════════════════════════════════════════════════════════════════════════════
NodeGraphEditor::NodeGraphEditor(QWidget* parent)
    : QGraphicsView(parent)
{
    scene_ = new NodeGraphScene(this);
    setScene(scene_);
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::RubberBandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    // Center the view on the Output node.
    centerOn(300, 0);
}

void NodeGraphEditor::contextMenuEvent(QContextMenuEvent* e) {
    QMenu menu(this);

    // Group by category.
    QMenu* prim = menu.addMenu("Primitives");
    QMenu* op   = menu.addMenu("Operations");
    QMenu* tr   = menu.addMenu("Transforms");
    QMenu* def  = menu.addMenu("Deformations");

    QPointF scene_pos = mapToScene(e->pos());

    for (const auto& info : node_catalog()) {
        QMenu* target = nullptr;
        switch (info.category) {
            case NodeCategory::Primitive:   target = prim; break;
            case NodeCategory::Operation:   target = op;   break;
            case NodeCategory::Transform:   target = tr;   break;
            case NodeCategory::Deformation: target = def;  break;
            // Instance isn't creatable from the palette (it needs a target
            // object); it's created from the Scene toolbar and shown here.
            default: continue;
        }
        QString type = info.type_name;
        QAction* a = target->addAction(info.display);
        connect(a, &QAction::triggered, this, [this, type, scene_pos]() {
            scene_->add_node(type, scene_pos);
        });
    }

    // Plugin-based primitives — listed under a separate submenu when a
    // registry is configured. Uses node_type_name as the canonical type
    // identifier (matches what FRepNode::type_name() reports for the
    // plugin's nodes).
    if (const auto* reg = scene_->registry()) {
        QMenu* plug = nullptr;
        for (const auto& slot : reg->primitives()) {
            if (slot.node_type_name.empty()) continue;
            if (!plug) plug = menu.addMenu("Plugins");
            QString type = QString::fromStdString(slot.node_type_name);
            QString display = QString("%1 (plugin)")
                .arg(QString::fromStdString(std::string(slot.info.name)));
            QAction* a = plug->addAction(display);
            connect(a, &QAction::triggered, this, [this, type, scene_pos]() {
                scene_->add_node(type, scene_pos);
            });
        }
    }

    menu.addSeparator();
    QAction* del = menu.addAction("Delete selected");
    connect(del, &QAction::triggered, this, [this]() {
        for (auto* it : scene_->selectedItems()) {
            if (auto* node = qgraphicsitem_cast<NodeItem*>(it)) {
                if (node == scene_->output_node()) continue; // the output is not deleted
                // remove the connected edges
                std::vector<EdgeItem*> to_remove;
                for (auto* sit : scene_->items())
                    if (auto* edge = qgraphicsitem_cast<EdgeItem*>(sit))
                        if (edge->from()->node() == node ||
                            edge->to()->node()   == node)
                            to_remove.push_back(edge);
                for (auto* edge : to_remove)
                    scene_->remove_edge(edge);
                scene_->removeItem(node);
                delete node;
            }
        }
        Q_EMIT scene_->graph_changed();
    });

    menu.exec(e->globalPos());
}

void NodeGraphEditor::wheelEvent(QWheelEvent* e) {
    // Zoom with the wheel.
    qreal factor = (e->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
    scale(factor, factor);
}

void NodeGraphEditor::fit_all() {
    // Compute the actual bounding rect of all items (not the scene's
    // sceneRect which Qt grows automatically and never shrinks). Add a
    // small margin so node borders don't touch the viewport edges.
    auto rect = scene_->itemsBoundingRect();
    if (rect.isEmpty()) return;
    constexpr qreal margin = 40.0;
    rect.adjust(-margin, -margin, margin, margin);
    fitInView(rect, Qt::KeepAspectRatio);
}

} // namespace frep::gui
