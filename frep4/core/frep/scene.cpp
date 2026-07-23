// core/frep/scene.cpp
//
// Out-of-line SceneGraph members that need the full definition of node
// types (which would create an include cycle if pulled into scene.hpp).

#include "core/frep/scene.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/custom_expr.hpp"

#include <string>

namespace frep {

namespace {
// Recursively point every CustomExpr node at the registry.
void wire_node(const FRepNode::Ptr& n, const TemplateRegistry* reg) {
    if (!n) return;
    if (std::string(n->type_name()) == "CustomExpr")
        static_cast<CustomExprNode*>(n.get())->set_templates(reg);
    for (const auto& c : n->children)
        wire_node(c, reg);
}
}  // namespace

void SceneGraph::wire_templates() {
    for (auto& [id, obj] : objects_)
        wire_node(obj.geometry, templates_.get());
}

void SceneGraph::set_translation(const std::string& id,
                                 std::array<float, 3> t) {
    auto it = objects_.find(id);
    if (it == objects_.end()) return;

    FRepNode::Ptr& geom = it->second.geometry;
    if (!geom) return;

    const bool is_zero = (t[0] == 0.0f && t[1] == 0.0f && t[2] == 0.0f);

    if (std::string(geom->type_name()) == "Translate") {
        // Root is already a Translate — adjust it in place. If the new
        // offset is zero, unwrap back to the child so we don't leave an
        // identity Translate cluttering the tree (keeps the node graph
        // clean and avoids unbounded nesting under repeated edits).
        if (is_zero && !geom->children.empty()) {
            geom = geom->children[0];
        } else {
            geom->params["tx"] = t[0];
            geom->params["ty"] = t[1];
            geom->params["tz"] = t[2];
        }
    } else if (!is_zero) {
        // Wrap the existing geometry in a fresh Translate. The wrapper
        // takes the object's id so the scene key (which is geom->id)
        // stays stable; the wrapped child gets a derived id.
        std::string root_id   = geom->id;
        std::string child_id  = root_id + "/geom";
        geom->id = child_id;
        geom = std::make_shared<TranslateNode>(geom, t[0], t[1], t[2], root_id);
    }
    // else: not a Translate and offset is zero → nothing to do.

    dirty_ = true;
    ++revision_;
}

// ── Rotation / scale gizmo ───────────────────────────────────────────────
//
// Translation, rotation, and scale are stored as nested transform-node
// wrappers around the object's geometry, in a canonical order so the
// three gizmo fields compose predictably regardless of edit sequence:
//
//     Translate( RotateY( Scale( geometry ) ) )
//
// i.e. scale is applied first (innermost), then rotation, then
// translation (outermost) — the usual T·R·S convention. Each setter
// locates its own node type in the chain (if present) and updates it in
// place, inserts it at the correct depth if absent, or unwraps it when
// the value returns to identity. The helpers below keep the chain in
// canonical order; because each transform type appears at most once,
// repeated edits never nest duplicates.
//
// Identity values: translation (0,0,0), rotation 0 rad, scale 1.0.

namespace {

// Returns the node's transform type tag, or empty if it's not one of the
// three gizmo wrappers. Used to walk/splice the canonical chain.
const char* gizmo_tag(const FRepNode::Ptr& n) {
    if (!n) return "";
    std::string t = n->type_name();
    if (t == "Translate") return "Translate";
    if (t == "RotateX")   return "RotateX";
    if (t == "RotateY")   return "RotateY";
    if (t == "RotateZ")   return "RotateZ";
    if (t == "Scale")     return "Scale";
    return "";
}

// Is this node any of the three rotation gizmo nodes?
static bool is_rotation_tag(const FRepNode::Ptr& n) {
    std::string t = gizmo_tag(n);
    return t == "RotateX" || t == "RotateY" || t == "RotateZ";
}

} // namespace

void SceneGraph::set_rotation_y(const std::string& id, float angle_rad) {
    auto it = objects_.find(id);
    if (it == objects_.end()) return;
    FRepNode::Ptr& geom = it->second.geometry;
    if (!geom) return;

    const bool is_identity = (angle_rad == 0.0f);

    // RotateY sits below Translate but above Scale. Walk past a leading
    // Translate (if any) to reach the rotation slot.
    FRepNode::Ptr* slot = &geom;
    if (gizmo_tag(*slot) == std::string("Translate") && !(*slot)->children.empty())
        slot = &(*slot)->children[0];

    if (gizmo_tag(*slot) == std::string("RotateY")) {
        if (is_identity && !(*slot)->children.empty()) {
            *slot = (*slot)->children[0];          // unwrap to child
        } else {
            (*slot)->params["a"] = angle_rad;  // update in place
        }
    } else if (!is_identity) {
        std::string root_id  = (*slot)->id;
        std::string child_id = root_id + "/rot";
        (*slot)->id = child_id;
        *slot = std::make_shared<RotateYNode>(*slot, angle_rad, root_id);
    }

    dirty_ = true;
    ++revision_;
}

// Per-axis rotation gizmo. The rotation chain between Translate and Scale is
// kept in a fixed canonical order — RotateX (outermost) → RotateY → RotateZ —
// so any subset can coexist. Setting an axis inserts/updates/removes its node in
// place without disturbing the others. axis: 0=X, 1=Y, 2=Z.
static const char* rot_tag_for_axis(int axis) {
    return axis == 0 ? "RotateX" : axis == 1 ? "RotateY" : "RotateZ";
}
static FRepNode::Ptr make_rot_for_axis(int axis, FRepNode::Ptr child, float a, std::string id) {
    if (axis == 0) return std::make_shared<RotateXNode>(std::move(child), a, std::move(id));
    if (axis == 1) return std::make_shared<RotateYNode>(std::move(child), a, std::move(id));
    return std::make_shared<RotateZNode>(std::move(child), a, std::move(id));
}

void SceneGraph::set_rotation_axis(const std::string& id, int axis, float angle_rad) {
    auto it = objects_.find(id);
    if (it == objects_.end() || !it->second.geometry) return;
    FRepNode::Ptr& geom = it->second.geometry;
    const bool is_identity = (angle_rad == 0.0f);

    // Walk past a leading Translate to the rotation region.
    FRepNode::Ptr* slot = &geom;
    if (gizmo_tag(*slot) == std::string("Translate") && !(*slot)->children.empty())
        slot = &(*slot)->children[0];

    // Peel the contiguous run of rotation nodes, recording each axis's angle.
    float ang[3] = {0, 0, 0};
    bool present[3] = {false, false, false};
    FRepNode::Ptr inner = *slot;
    while (is_rotation_tag(inner) && !inner->children.empty()) {
        std::string t = gizmo_tag(inner);
        int a = (t == "RotateX") ? 0 : (t == "RotateY") ? 1 : 2;
        auto p = inner->params.find("a");
        ang[a] = (p != inner->params.end()) ? p->second : 0.0f;
        present[a] = true;
        inner = inner->children[0];
    }
    // Apply the requested change to the target axis.
    ang[axis] = angle_rad;
    present[axis] = !is_identity;

    // Reassemble in canonical order: Z innermost → Y → X outermost, on top of the
    // non-rotation base (Scale or bare geometry). The outermost node inherits the
    // original slot id so the Translate above (or the object map) still refers to
    // it; the base keeps its own id.
    const std::string root_id = (*slot)->id;
    FRepNode::Ptr built = inner;
    for (int a = 2; a >= 0; --a) {
        if (!present[a]) continue;
        char axc = rot_tag_for_axis(a)[6];               // 'X' / 'Y' / 'Z'
        built = make_rot_for_axis(a, built, ang[a],
                                  root_id + "/rot" + std::string(1, axc));
    }
    built->id = root_id;
    *slot = built;

    dirty_ = true;
    ++revision_;
}

float SceneGraph::get_rotation_axis(const std::string& id, int axis) const {
    auto it = objects_.find(id);
    if (it == objects_.end() || !it->second.geometry) return 0.0f;
    const std::string want = rot_tag_for_axis(axis);
    FRepNode::Ptr n = it->second.geometry;
    if (gizmo_tag(n) == std::string("Translate") && !n->children.empty())
        n = n->children[0];
    while (is_rotation_tag(n) && !n->children.empty()) {
        if (gizmo_tag(n) == want) {
            auto p = n->params.find("a");
            return (p != n->params.end()) ? p->second : 0.0f;
        }
        n = n->children[0];
    }
    return 0.0f;
}

void SceneGraph::set_scale(const std::string& id, float s) {
    auto it = objects_.find(id);
    if (it == objects_.end()) return;
    FRepNode::Ptr& geom = it->second.geometry;
    if (!geom) return;

    const bool is_identity = (s == 1.0f);

    // Scale is the innermost wrapper: walk past Translate and RotateY.
    FRepNode::Ptr* slot = &geom;
    if (gizmo_tag(*slot) == std::string("Translate") && !(*slot)->children.empty())
        slot = &(*slot)->children[0];
    while (is_rotation_tag(*slot) && !(*slot)->children.empty())
        slot = &(*slot)->children[0];

    if (gizmo_tag(*slot) == std::string("Scale")) {
        if (is_identity && !(*slot)->children.empty()) {
            *slot = (*slot)->children[0];
        } else {
            (*slot)->params["sx"] = s;   // uniform: all three axes equal
            (*slot)->params["sy"] = s;
            (*slot)->params["sz"] = s;
        }
    } else if (!is_identity) {
        std::string root_id  = (*slot)->id;
        std::string child_id = root_id + "/scl";
        (*slot)->id = child_id;
        *slot = std::make_shared<ScaleNode>(*slot, s, root_id);
    }

    dirty_ = true;
    ++revision_;
}

float SceneGraph::get_rotation_y(const std::string& id) const {
    auto it = objects_.find(id);
    if (it == objects_.end() || !it->second.geometry) return 0.0f;
    FRepNode::Ptr n = it->second.geometry;
    if (gizmo_tag(n) == std::string("Translate") && !n->children.empty())
        n = n->children[0];
    if (gizmo_tag(n) == std::string("RotateY")) {
        auto p = n->params.find("a");
        if (p != n->params.end()) return p->second;
    }
    return 0.0f;
}

float SceneGraph::get_scale(const std::string& id) const {
    auto it = objects_.find(id);
    if (it == objects_.end() || !it->second.geometry) return 1.0f;
    FRepNode::Ptr n = it->second.geometry;
    if (gizmo_tag(n) == std::string("Translate") && !n->children.empty())
        n = n->children[0];
    while (is_rotation_tag(n) && !n->children.empty())
        n = n->children[0];
    if (gizmo_tag(n) == std::string("Scale")) {
        auto p = n->params.find("sx");
        if (p != n->params.end()) return p->second;
    }
    return 1.0f;
}

// Per-axis scale: sets all three factors. A uniform scale is sx==sy==sz.
void SceneGraph::set_scale_xyz(const std::string& id, float sx, float sy, float sz) {
    auto it = objects_.find(id);
    if (it == objects_.end()) return;
    FRepNode::Ptr& geom = it->second.geometry;
    if (!geom) return;
    const bool is_identity = (sx == 1.0f && sy == 1.0f && sz == 1.0f);
    FRepNode::Ptr* slot = &geom;
    if (gizmo_tag(*slot) == std::string("Translate") && !(*slot)->children.empty())
        slot = &(*slot)->children[0];
    while (is_rotation_tag(*slot) && !(*slot)->children.empty())
        slot = &(*slot)->children[0];
    if (gizmo_tag(*slot) == std::string("Scale")) {
        if (is_identity && !(*slot)->children.empty()) {
            *slot = (*slot)->children[0];
        } else {
            (*slot)->params["sx"] = sx;
            (*slot)->params["sy"] = sy;
            (*slot)->params["sz"] = sz;
        }
    } else if (!is_identity) {
        std::string root_id  = (*slot)->id;
        std::string child_id = root_id + "/scl";
        (*slot)->id = child_id;
        *slot = std::make_shared<ScaleNode>(*slot, sx, sy, sz, root_id);
    }
    dirty_ = true;
    ++revision_;
}

void SceneGraph::get_scale_xyz(const std::string& id, float& sx, float& sy, float& sz) const {
    sx = sy = sz = 1.0f;
    auto it = objects_.find(id);
    if (it == objects_.end() || !it->second.geometry) return;
    FRepNode::Ptr n = it->second.geometry;
    if (gizmo_tag(n) == std::string("Translate") && !n->children.empty())
        n = n->children[0];
    while (is_rotation_tag(n) && !n->children.empty())
        n = n->children[0];
    if (gizmo_tag(n) == std::string("Scale")) {
        auto px=n->params.find("sx"), py=n->params.find("sy"), pz=n->params.find("sz");
        if (px!=n->params.end()) sx=px->second;
        if (py!=n->params.end()) sy=py->second;
        if (pz!=n->params.end()) sz=pz->second;
    }
}

bool SceneGraph::set_node_param(const std::string& object_id, const std::string& node_id,
                                const std::string& param, float value) {
    auto it = objects_.find(object_id);
    if (it == objects_.end() || !it->second.geometry) return false;
    FRepNode* n = find_node_by_id(*it->second.geometry, node_id);
    if (!n) return false;
    n->params[param] = value;
    dirty_ = true;
    ++revision_;
    return true;
}

bool SceneGraph::get_node_param(const std::string& object_id, const std::string& node_id,
                                const std::string& param, float& out) const {
    auto it = objects_.find(object_id);
    if (it == objects_.end() || !it->second.geometry) return false;
    // find_node_by_id needs a mutable ref; we only read, so const_cast is safe here.
    FRepNode* n = find_node_by_id(const_cast<FRepNode&>(*it->second.geometry), node_id);
    if (!n) return false;
    auto p = n->params.find(param);
    if (p == n->params.end()) return false;
    out = p->second;
    return true;
}

} // namespace frep
