#pragma once
// core/frep/scene.hpp
//
// SceneGraph — a container for the FRepNode tree, materials, camera and lights.
// This is the "program" of the visual language — the full image description.

#include "node.hpp"
#include "template_fn.hpp"
#include <array>
#include <memory>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace frep {

// ── Material ──────────────────────────────────────────────────────────────────
struct Material {
    // Procedural pattern type. The default Solid uses `albedo` directly.
    // Other patterns blend between `albedo` (primary color) and `albedo2`
    // (secondary color) based on a position-dependent function:
    //
    //   Checker    — 3D checkerboard, scale = `pattern_scale` (cells per
    //                world unit). Classic Larry Gritz checkerboard.
    //   Stripes    — alternating bands along the Y axis, period = 1 /
    //                pattern_scale.
    //   GradientY  — smooth linear gradient between albedo and albedo2,
    //                from y=-pattern_scale to y=+pattern_scale.
    //   Noise      — hash-based 3D value-noise; pattern_scale = frequency.
    //   Texture    — sample an RGBA8 image via triplanar projection. The
    //                texture's pixels live in `texture_rgba`; the
    //                `pattern_scale` field then maps world-space units to
    //                texture repetitions.
    //
    // All patterns are evaluated against the hit point's world coordinates,
    // so they "stick" to the surface naturally.
    enum class Pattern {
        Solid, Checker, Stripes, GradientY, Noise, Texture
    };

    std::array<float, 3> albedo        = {0.8f, 0.8f, 0.8f};
    std::array<float, 3> albedo2       = {0.1f, 0.1f, 0.1f};
    Pattern              pattern       = Pattern::Solid;
    float                pattern_scale = 1.0f;

    // ── Texture-only fields ────────────────────────────────────────────────
    // RGBA8 pixel data in row-major top-down order. Empty unless
    // `pattern == Pattern::Texture`. Loaded by Image (core/io/bmp_loader.hpp);
    // the renderer concatenates all textures into a single GPU buffer.
    std::vector<std::uint8_t> texture_rgba;
    int                       texture_width  = 0;
    int                       texture_height = 0;

    // Source file the texture was loaded from. When non-empty, scene_io
    // serializes this string instead of the pixel blob and reloads it
    // (via load_image) when the scene is opened. This keeps JSON scene
    // files compact even with large textures.
    std::string               texture_path;

    float                roughness    = 0.5f;
    float                metallic     = 0.0f;
    float                emission     = 0.0f;
    // Mirror reflection strength in [0, 1]. 0 = matte (no secondary
    // reflection ray), 1 = perfect mirror. The renderer casts one
    // reflection ray per hit when reflectivity > 0 and blends the
    // reflected colour in by this factor (modulated by a Fresnel term
    // so grazing angles reflect more). Both CPU JIT and GPU paths
    // honour it; recursion depth is capped at `max_bounces` in
    // TracerConfig to keep it bounded.
    float                reflectivity = 0.0f;

    // Value equality — used by the material editor to coalesce a
    // multi-object edit into one undo entry (objects whose material
    // didn't actually change are skipped). Compares every field,
    // including the texture pixel blob; cheap in practice since the
    // editor only compares the handful of objects in the selection.
    bool operator==(const Material& o) const {
        return albedo == o.albedo && albedo2 == o.albedo2
            && pattern == o.pattern && pattern_scale == o.pattern_scale
            && texture_rgba == o.texture_rgba
            && texture_width == o.texture_width
            && texture_height == o.texture_height
            && texture_path == o.texture_path
            && roughness == o.roughness && metallic == o.metallic
            && emission == o.emission && reflectivity == o.reflectivity;
    }
    bool operator!=(const Material& o) const { return !(*this == o); }
};

// ── SceneObject ───────────────────────────────────────────────────────────────
struct SceneObject {
    FRepNode::Ptr geometry;
    Material      material;
    bool          visible = true;
};

// ── Camera ────────────────────────────────────────────────────────────────────
struct Camera {
    enum class Projection {
        Perspective,    // standard pinhole — fov_deg controls FoV
        Orthographic    // parallel rays — ortho_size sets the view height
    };

    std::array<float, 3> position = { 0.0f, 2.0f, 7.0f };
    std::array<float, 3> target   = { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> up       = { 0.0f, 1.0f, 0.0f };

    Projection projection = Projection::Perspective;
    float fov_deg     = 60.0f;
    // Height of the view rectangle at the focal plane (in world units).
    // Width is derived from the image aspect ratio.
    float ortho_size  = 4.0f;
};

// ── PointLight ────────────────────────────────────────────────────────────────
struct PointLight {
    std::array<float, 3> pos       = {  5.0f, 10.0f,  5.0f };
    std::array<float, 3> color     = {  1.0f,  1.0f,  1.0f };
    float                intensity = 1.0f;
};

// ── SceneGraph ────────────────────────────────────────────────────────────────
class SceneGraph {
public:
    void add_object(FRepNode::Ptr geom, Material mat = {}) {
        std::string key = geom->id;
        objects_.emplace(key, SceneObject{std::move(geom), mat});
        dirty_ = true;
        ++revision_;
    }

    void remove_object(const std::string& id) {
        objects_.erase(id);
        dirty_ = true;
        ++revision_;
    }

    // ── User-defined template functions ──────────────────────────────────────
    // Named, scalar-parameterised sub-expressions callable from CustomExpr
    // bodies (see template_fn.hpp). Owned by the scene; nodes reference it.
    TemplateRegistry&       templates()       { return *templates_; }
    const TemplateRegistry& templates() const { return *templates_; }
    // Point every CustomExpr node (recursively, across all objects) at this
    // scene's registry so their bodies can call templates. Call after adding
    // templates and objects (deserialize_scene does this automatically).
    void wire_templates();

    void set_material(const std::string& id, Material mat) {
        if (auto it = objects_.find(id); it != objects_.end()) {
            it->second.material = mat;
            shader_dirty_ = true;
            ++revision_;
        }
    }

    // Swap an object's geometry (FRepNode tree) without changing its id,
    // material, or visibility. Used by the node graph editor when the
    // user rewires the tree — it edits the active object in place
    // rather than wiping the scene. Marks geometry as dirty so the
    // viewport recompiles.
    void set_geometry(const std::string& id, FRepNode::Ptr geometry) {
        if (auto it = objects_.find(id); it != objects_.end()) {
            it->second.geometry = std::move(geometry);
            dirty_ = true;
            ++revision_;
        }
    }

    // Toggles an object's visibility. The change is geometry-dirty,
    // because hidden objects are not included in the emitted scene_sdf.
    void set_visibility(const std::string& id, bool visible) {
        if (auto it = objects_.find(id); it != objects_.end()) {
            it->second.visible = visible;
            dirty_ = true;
            ++revision_;
        }
    }

    // ── Per-object translation (gizmo support) ────────────────────────────────
    // The editor exposes a world-space translation per object. Rather
    // than add a parallel transform field that every codegen/GLSL/BVH
    // path would have to learn about, we represent it as an implicit
    // TranslateNode wrapping the object's geometry root: every existing
    // path already handles TranslateNode, so translation "just works"
    // across CPU JIT, GPU, picking, and meshing with zero new plumbing.
    //
    // get_translation reads the offset back out (0,0,0 if the root is
    // not a TranslateNode). set_translation ensures the root is a
    // TranslateNode with the given offset, wrapping or unwrapping as
    // needed so repeated edits don't nest translates.
    std::array<float, 3> get_translation(const std::string& id) const {
        auto it = objects_.find(id);
        if (it == objects_.end()) return {0, 0, 0};
        const FRepNode* g = it->second.geometry.get();
        if (g && std::string(g->type_name()) == "Translate") {
            return { g->params.at("tx"), g->params.at("ty"), g->params.at("tz") };
        }
        return {0, 0, 0};
    }

    // Implemented out-of-line in scene.cpp (needs TranslateNode's full
    // definition, which would create an include cycle in this header).
    void set_translation(const std::string& id, std::array<float, 3> t);

    // Rotation (Y axis) and uniform scale gizmo. Stored as nested
    // transform wrappers in canonical T·R·S order alongside translation
    // (see scene.cpp). Identity = 0 rad / scale 1.0; setting identity
    // unwraps the node so the tree stays clean. get_* read the current
    // value back (identity if the wrapper isn't present).
    void  set_rotation_y(const std::string& id, float angle_rad);
    void  set_scale(const std::string& id, float s);
    float get_rotation_y(const std::string& id) const;
    float get_scale(const std::string& id) const;
    // Per-axis rotation gizmo (axis: 0=X, 1=Y, 2=Z). Keeps a canonical
    // RotateX→RotateY→RotateZ chain so the three axes are independent.
    void  set_rotation_axis(const std::string& id, int axis, float angle_rad);
    float get_rotation_axis(const std::string& id, int axis) const;
    // Per-axis (non-uniform) scale gizmo.
    void  set_scale_xyz(const std::string& id, float sx, float sy, float sz);
    void  get_scale_xyz(const std::string& id, float& sx, float& sy, float& sz) const;

    // Set a single parameter of a node (found by node_id) inside an object's
    // geometry tree, bumping the revision so the render updates. This is a
    // parameter-only change (topology unchanged), so it can ride the incremental
    // path when supported. Returns false if the object or node isn't found.
    bool  set_node_param(const std::string& object_id, const std::string& node_id,
                         const std::string& param, float value);
    bool  get_node_param(const std::string& object_id, const std::string& node_id,
                         const std::string& param, float& out) const;

    // ── Accessors ─────────────────────────────────────────────────────────────
    const auto& objects()  const { return objects_; }

    // Mutable object map — used by instance resolution to rebind InstanceNode
    // target pointers in place. Prefer the const objects() elsewhere.
    auto& objects_mutable() { return objects_; }

    // Look up a single object by id, or nullptr if absent. Convenience for the
    // GUI (instancing, deformation) which works with the primary selection id.
    const SceneObject* find_object(const std::string& id) const {
        auto it = objects_.find(id);
        return it == objects_.end() ? nullptr : &it->second;
    }
    Camera&       camera()       { return cam_; }
    const Camera& camera() const { return cam_; }
    std::vector<PointLight>&       lights()       { return lights_; }
    const std::vector<PointLight>& lights() const { return lights_; }

    // ── Dirty tracking ────────────────────────────────────────────────────────
    bool geom_dirty()   const { return dirty_; }
    bool shader_dirty() const { return shader_dirty_; }
    void clear_dirty()        { dirty_ = false; shader_dirty_ = false; }

    // Monotonic mutation counter. Incremented on every structural or
    // material mutation routed through the setter methods (add/remove/
    // set_material/set_geometry/set_visibility/set_translation). Used by
    // the real-time renderer's temporal-accumulation logic to detect
    // when the scene changed and the running average must restart.
    //
    // NOTE: direct edits through the non-const lights()/camera()
    // accessors do NOT bump this (they bypass the setters); the renderer
    // hashes those fields separately. Geometry parameters edited
    // in-place on a node (without going through set_geometry) likewise
    // won't bump it — callers that mutate nodes directly should pair the
    // edit with one of the setters, as the GUI does.
    std::uint64_t revision() const noexcept { return revision_; }

    // ── Hash of the whole scene (for the incremental cache) ──────────────────
    std::size_t scene_hash() const noexcept {
        std::size_t h = 0;
        for (auto& [id, obj] : objects_)
            h = h * 2654435761ull ^ obj.geometry->structural_hash();
        return h;
    }

    // ── Hash of the scene *structure* only (ignores parameter values) ────────
    // Used by IncrementalCompiler to detect the "structure unchanged, only
    // parameters changed" case — useful for future per-parameter incremental
    // strategies (e.g. emitting parameters as runtime arguments instead of
    // baked-in constants, which would let one JIT-ed module serve many
    // parameter sets without recompilation).
    std::size_t structure_hash() const noexcept {
        std::size_t h = 0;
        for (auto& [id, obj] : objects_)
            h = h * 2654435761ull ^ obj.geometry->structure_hash();
        return h;
    }

private:
    std::unordered_map<std::string, SceneObject> objects_;
    // Heap-allocated so a node's raw pointer to it (set by wire_templates)
    // survives SceneGraph moves/copies (load_scene returns by value).
    std::shared_ptr<TemplateRegistry> templates_ =
        std::make_shared<TemplateRegistry>();
    Camera                  cam_;
    std::vector<PointLight> lights_ = { PointLight{} };
    bool dirty_        = false;
    bool shader_dirty_ = false;
    std::uint64_t revision_ = 0;
};

} // namespace frep
