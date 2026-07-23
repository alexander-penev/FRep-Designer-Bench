#pragma once
// core/undo/undo_stack.hpp
//
// Lightweight undo/redo stack for SceneGraph operations.
//
// Design
// ──────
// A UndoCommand has two methods: apply() (do, or redo) and undo() (reverse).
// Each command captures the minimum state needed to be reversible — usually
// a copy of the old value plus a reference to the scene.
//
// UndoStack owns the commands. push_apply() does:
//   1. Run cmd->apply() to perform the change.
//   2. Discard any redo history past the cursor (since the new branch
//      makes the old redo path invalid).
//   3. Append cmd at the cursor.
//
// undo() moves the cursor back one and calls undo() on the popped command.
// redo() moves forward and re-applies.
//
// Covered scene-edit operations, each as an undoable command:
//   - add/remove object (AddObjectCommand / RemoveObjectCommand)
//   - set geometry (SetGeometryCommand) — node-graph edits, including primitive
//     parameter changes (sphere radius, box dimensions), arrive here as one
//     atomic command via sync_graph_to_scene
//   - set transform / rotation / scale (gizmo edits)
//   - set material (albedo, roughness, metallic)
//   - set per-object visibility
//   - add/remove light, set light parameters

#include "core/frep/scene.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace frep::undo {

class UndoCommand {
public:
    virtual ~UndoCommand() = default;

    // Run the change forward. Called both on the initial push and on redo.
    virtual void apply() = 0;
    // Reverse the change. Called on undo.
    virtual void undo() = 0;
    // Optional: short human-readable label for menus / status bar.
    virtual std::string description() const { return "(unnamed)"; }
};

// ─── Built-in commands ──────────────────────────────────────────────────────

// Adds an object to the scene.
class AddObjectCommand : public UndoCommand {
public:
    AddObjectCommand(SceneGraph& s, std::string id,
                     FRepNode::Ptr geom, Material mat)
        : s_(s), id_(std::move(id)), geom_(std::move(geom)), mat_(mat) {}

    void apply() override { s_.add_object(geom_, mat_); }
    void undo()  override { s_.remove_object(id_); }
    std::string description() const override { return "Add " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    FRepNode::Ptr geom_;
    Material mat_;
};

// Removes an object. Captures the SceneObject snapshot so undo can restore
// it exactly — geometry pointer, material, visibility.
class RemoveObjectCommand : public UndoCommand {
public:
    RemoveObjectCommand(SceneGraph& s, std::string id)
        : s_(s), id_(std::move(id))
    {
        auto it = s_.objects().find(id_);
        if (it != s_.objects().end()) snap_ = it->second;
    }

    void apply() override {
        // Re-snap on every apply in case the object was edited between
        // the constructor call and apply (push_apply does an immediate
        // apply, so this is a no-op in the typical case, but it makes
        // redo robust if other commands ran in between).
        auto it = s_.objects().find(id_);
        if (it != s_.objects().end()) snap_ = it->second;
        s_.remove_object(id_);
    }

    void undo() override {
        if (!snap_.geometry) return;
        s_.add_object(snap_.geometry, snap_.material);
        // Restore visibility — add_object defaults to visible=true.
        s_.set_visibility(id_, snap_.visible);
    }
    std::string description() const override { return "Remove " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    SceneObject snap_;
};

// Sets a per-object material. Captures the old material at construction so
// undo can restore it. apply() is idempotent (writes the new material;
// reading the old happens once).
// Swaps an object's geometry tree, preserving id + material. Used by
// the node graph editor — each user edit in the graph (add node,
// connect ports, change parameter, etc.) eventually triggers a
// set_geometry, and routing those through this command makes them
// undoable as a single atomic step. The old tree is held by
// shared_ptr; if the user undoes, we get exactly the previous tree
// back, not just a shallow copy.
class SetGeometryCommand : public UndoCommand {
public:
    SetGeometryCommand(SceneGraph& s, std::string id,
                       FRepNode::Ptr new_geom)
        : s_(s), id_(std::move(id)), new_(std::move(new_geom))
    {
        auto it = s_.objects().find(id_);
        if (it != s_.objects().end()) old_ = it->second.geometry;
    }

    void apply() override { s_.set_geometry(id_, new_); }
    void undo()  override { if (old_) s_.set_geometry(id_, old_); }
    std::string description() const override { return "Geometry on " + id_; }

private:
    SceneGraph&    s_;
    std::string    id_;
    FRepNode::Ptr  old_, new_;
};

class SetMaterialCommand : public UndoCommand {
public:
    SetMaterialCommand(SceneGraph& s, std::string id, Material new_mat)
        : s_(s), id_(std::move(id)), new_(new_mat)
    {
        auto it = s_.objects().find(id_);
        if (it != s_.objects().end()) old_ = it->second.material;
    }

    void apply() override { s_.set_material(id_, new_); }
    void undo()  override { s_.set_material(id_, old_); }
    std::string description() const override { return "Material on " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    Material old_, new_;
};

// Multi-object material edit, coalesced into one undo entry. Used when a
// material change in the editor is broadcast across a multi-selection:
// each affected object carries its own (old, new) pair so a single undo
// reverts every object, not just the primary one. Construct with the
// per-object baseline materials (captured at selection time) and the
// new materials to apply.
class SetMaterialsCommand : public UndoCommand {
public:
    struct Entry { std::string id; Material old_mat, new_mat; };

    SetMaterialsCommand(SceneGraph& s, std::vector<Entry> entries)
        : s_(s), entries_(std::move(entries)) {}

    void apply() override {
        for (const auto& e : entries_)
            if (s_.objects().count(e.id)) s_.set_material(e.id, e.new_mat);
    }
    void undo() override {
        for (const auto& e : entries_)
            if (s_.objects().count(e.id)) s_.set_material(e.id, e.old_mat);
    }
    std::string description() const override {
        return entries_.size() == 1
            ? "Material on " + entries_.front().id
            : "Material on " + std::to_string(entries_.size()) + " objects";
    }

private:
    SceneGraph&        s_;
    std::vector<Entry> entries_;
};

// Sets per-object visibility.
class SetVisibilityCommand : public UndoCommand {
public:
    SetVisibilityCommand(SceneGraph& s, std::string id, bool v)
        : s_(s), id_(std::move(id)), new_(v)
    {
        auto it = s_.objects().find(id_);
        if (it != s_.objects().end()) old_ = it->second.visible;
    }
    void apply() override { s_.set_visibility(id_, new_); }
    void undo()  override { s_.set_visibility(id_, old_); }
    std::string description() const override {
        return std::string(new_ ? "Show " : "Hide ") + id_;
    }

private:
    SceneGraph& s_;
    std::string id_;
    bool old_ = true, new_ = true;
};

// Sets an object's world-space translation (the gizmo). Captures the
// previous offset at construction so undo restores it exactly. Coalesces
// nothing — each committed drag / spinbox edit is one undo step.
class SetTransformCommand : public UndoCommand {
public:
    SetTransformCommand(SceneGraph& s, std::string id,
                        std::array<float, 3> t)
        : s_(s), id_(std::move(id)), new_(t)
    {
        old_ = s_.get_translation(id_);
    }
    void apply() override { s_.set_translation(id_, new_); }
    void undo()  override { s_.set_translation(id_, old_); }
    std::string description() const override { return "Move " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    std::array<float, 3> old_{0, 0, 0}, new_{0, 0, 0};
};

// Y-axis rotation gizmo edit (radians). Mirrors SetTransformCommand.
class SetRotationCommand : public UndoCommand {
public:
    SetRotationCommand(SceneGraph& s, std::string id, float angle_rad)
        : s_(s), id_(std::move(id)), new_(angle_rad)
    {
        old_ = s_.get_rotation_y(id_);
    }
    void apply() override { s_.set_rotation_y(id_, new_); }
    void undo()  override { s_.set_rotation_y(id_, old_); }
    std::string description() const override { return "Rotate " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    float old_ = 0.0f, new_ = 0.0f;
};

// Per-axis rotation (axis 0=X, 1=Y, 2=Z).
class SetRotationAxisCommand : public UndoCommand {
public:
    SetRotationAxisCommand(SceneGraph& s, std::string id, int axis, float angle_rad)
        : s_(s), id_(std::move(id)), axis_(axis), new_(angle_rad)
    {
        old_ = s_.get_rotation_axis(id_, axis_);
    }
    void apply() override { s_.set_rotation_axis(id_, axis_, new_); }
    void undo()  override { s_.set_rotation_axis(id_, axis_, old_); }
    std::string description() const override { return "Rotate " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    int axis_ = 1;
    float old_ = 0.0f, new_ = 0.0f;
};

// Uniform-scale gizmo edit. Mirrors SetTransformCommand.
class SetScaleCommand : public UndoCommand {
public:
    SetScaleCommand(SceneGraph& s, std::string id, float s_factor)
        : s_(s), id_(std::move(id)), new_(s_factor)
    {
        old_ = s_.get_scale(id_);
    }
    void apply() override { s_.set_scale(id_, new_); }
    void undo()  override { s_.set_scale(id_, old_); }
    std::string description() const override { return "Scale " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    float old_ = 1.0f, new_ = 1.0f;
};

// Per-axis (non-uniform) scale.
class SetScaleXYZCommand : public UndoCommand {
public:
    SetScaleXYZCommand(SceneGraph& s, std::string id, float sx, float sy, float sz)
        : s_(s), id_(std::move(id)), nx_(sx), ny_(sy), nz_(sz)
    {
        s_.get_scale_xyz(id_, ox_, oy_, oz_);
    }
    void apply() override { s_.set_scale_xyz(id_, nx_, ny_, nz_); }
    void undo()  override { s_.set_scale_xyz(id_, ox_, oy_, oz_); }
    std::string description() const override { return "Scale " + id_; }

private:
    SceneGraph& s_;
    std::string id_;
    float ox_=1, oy_=1, oz_=1, nx_=1, ny_=1, nz_=1;
};

// Sets a single parameter of a node inside an object's geometry tree — used by
// the Scene-tab property grid. Parameter-only (topology unchanged), so it can
// ride the incremental recompile path.
class SetParamCommand : public UndoCommand {
public:
    SetParamCommand(SceneGraph& s, std::string object_id, std::string node_id,
                    std::string param, float value)
        : s_(s), obj_(std::move(object_id)), node_(std::move(node_id)),
          param_(std::move(param)), new_(value)
    {
        has_old_ = s_.get_node_param(obj_, node_, param_, old_);
    }
    void apply() override { s_.set_node_param(obj_, node_, param_, new_); }
    void undo()  override { if (has_old_) s_.set_node_param(obj_, node_, param_, old_); }
    std::string description() const override { return "Set " + node_ + "." + param_; }

private:
    SceneGraph& s_;
    std::string obj_, node_, param_;
    float old_ = 0.0f, new_ = 0.0f;
    bool has_old_ = false;
};

// Adds a light. Captures the index so undo removes the right one.
class AddLightCommand : public UndoCommand {
public:
    AddLightCommand(SceneGraph& s, PointLight L) : s_(s), L_(L) {}
    void apply() override {
        s_.lights().push_back(L_);
        idx_ = static_cast<int>(s_.lights().size()) - 1;
    }
    void undo() override {
        auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size())) {
            Ls.erase(Ls.begin() + idx_);
        }
    }
    std::string description() const override { return "Add light"; }

private:
    SceneGraph& s_;
    PointLight L_;
    int idx_ = -1;
};

// Removes a light. Captures index + value so undo restores at the same slot.
class RemoveLightCommand : public UndoCommand {
public:
    RemoveLightCommand(SceneGraph& s, int idx) : s_(s), idx_(idx) {
        const auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size()))
            snap_ = Ls[idx_];
    }
    void apply() override {
        auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size())) {
            snap_ = Ls[idx_];
            Ls.erase(Ls.begin() + idx_);
        }
    }
    void undo() override {
        auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ <= static_cast<int>(Ls.size())) {
            Ls.insert(Ls.begin() + idx_, snap_);
        }
    }
    std::string description() const override { return "Remove light"; }

private:
    SceneGraph& s_;
    int idx_;
    PointLight snap_{};
};

// Sets a light's full state — used as the "atomic" capture from the GUI's
// editor on commit (e.g. when the user finishes dragging a spinbox).
class SetLightCommand : public UndoCommand {
public:
    SetLightCommand(SceneGraph& s, int idx, PointLight new_val)
        : s_(s), idx_(idx), new_(new_val)
    {
        const auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size())) old_ = Ls[idx_];
    }
    void apply() override {
        auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size())) Ls[idx_] = new_;
    }
    void undo() override {
        auto& Ls = s_.lights();
        if (idx_ >= 0 && idx_ < static_cast<int>(Ls.size())) Ls[idx_] = old_;
    }
    std::string description() const override { return "Edit light"; }

private:
    SceneGraph& s_;
    int idx_;
    PointLight old_{}, new_{};
};

// ─── UndoStack ──────────────────────────────────────────────────────────────

class UndoStack {
public:
    // Push a command and apply it. Discards any redo history past the cursor.
    void push_apply(std::unique_ptr<UndoCommand> cmd) {
        if (!cmd) return;
        cmd->apply();
        // Truncate redo history.
        if (cursor_ < static_cast<int>(stack_.size())) {
            stack_.erase(stack_.begin() + cursor_, stack_.end());
        }
        stack_.push_back(std::move(cmd));
        cursor_ = static_cast<int>(stack_.size());
        notify();
    }

    // Pushes a command without applying it (caller has already applied).
    // Useful when the change came from an external action that we want to
    // make reversible after the fact.
    void push_no_apply(std::unique_ptr<UndoCommand> cmd) {
        if (!cmd) return;
        if (cursor_ < static_cast<int>(stack_.size())) {
            stack_.erase(stack_.begin() + cursor_, stack_.end());
        }
        stack_.push_back(std::move(cmd));
        cursor_ = static_cast<int>(stack_.size());
        notify();
    }

    bool can_undo() const { return cursor_ > 0; }
    bool can_redo() const { return cursor_ < static_cast<int>(stack_.size()); }

    void undo() {
        if (!can_undo()) return;
        --cursor_;
        stack_[cursor_]->undo();
        notify();
    }
    void redo() {
        if (!can_redo()) return;
        stack_[cursor_]->apply();
        ++cursor_;
        notify();
    }

    // Drops the entire history. Use when a fresh scene is loaded.
    void clear() {
        stack_.clear();
        cursor_ = 0;
        notify();
    }

    std::string undo_description() const {
        return can_undo() ? stack_[cursor_ - 1]->description() : "";
    }
    std::string redo_description() const {
        return can_redo() ? stack_[cursor_]->description() : "";
    }

    // Observer: called whenever the stack state changes (push/undo/redo/clear).
    // Used by the GUI to refresh enable-state and labels on Undo/Redo actions.
    void set_change_observer(std::function<void()> f) {
        observer_ = std::move(f);
    }

    int size()   const { return static_cast<int>(stack_.size()); }
    int cursor() const { return cursor_; }

private:
    std::vector<std::unique_ptr<UndoCommand>> stack_;
    int cursor_ = 0;
    std::function<void()> observer_;
    void notify() { if (observer_) observer_(); }
};

} // namespace frep::undo
