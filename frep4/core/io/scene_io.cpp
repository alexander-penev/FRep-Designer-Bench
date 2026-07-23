// core/io/scene_io.cpp
#include <unordered_set>

#include "core/io/scene_io.hpp"

#include "core/frep/custom_expr.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/instance.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/io/json.hpp"
#include "core/io/png_loader.hpp"
#include "core/plugin/plugin_api.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace frep::io {

namespace fs = std::filesystem;

// Rewrites an absolute or already-relative texture path as a path
// relative to `base_dir`, when the result stays within the same
// directory tree. Returns the original string if base_dir is empty,
// if the input is already a non-absolute path, or if relative_path
// throws (which it can do on Windows when the inputs come from
// different drives).
//
// We deliberately do NOT rewrite paths that escape `base_dir` via
// "..": those tend to surprise the user when scenes are moved.
static std::string rewrite_relative(const std::string& tex_path,
                                    const std::string& base_dir)
{
    if (tex_path.empty() || base_dir.empty()) return tex_path;
    try {
        fs::path p(tex_path);
        if (!p.is_absolute()) return tex_path;  // already relative
        fs::path rel = fs::relative(p, fs::path(base_dir));
        if (rel.empty()) return tex_path;
        // Keep relative only if it stays inside base_dir.
        std::string s = rel.generic_string();
        if (s.starts_with("..")) return tex_path;
        return s;
    } catch (...) {
        return tex_path;
    }
}

// Resolves a possibly-relative texture path against `base_dir`. If
// the path is already absolute (or base_dir is empty) it's returned
// unchanged. The output is always passed through filesystem::path
// normalisation so backslashes-on-Windows / forward-slashes-on-Unix
// don't trip up the loader.
static std::string resolve_against(const std::string& tex_path,
                                   const std::string& base_dir)
{
    if (tex_path.empty() || base_dir.empty()) return tex_path;
    try {
        fs::path p(tex_path);
        if (p.is_absolute()) return tex_path;
        return (fs::path(base_dir) / p).lexically_normal().string();
    } catch (...) {
        return tex_path;
    }
}

using json::Value;
using json::Object;
using json::Array;

// ── base64 (for embedded texture pixels) ─────────────────────────────────────
static const char* kB64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const std::vector<std::uint8_t>& in) {
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    std::size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        std::uint32_t n = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out += kB64[(n >> 18) & 63]; out += kB64[(n >> 12) & 63];
        out += kB64[(n >> 6) & 63];  out += kB64[n & 63];
    }
    if (i < in.size()) {
        std::uint32_t n = in[i] << 16;
        bool two = (i + 1 < in.size());
        if (two) n |= in[i + 1] << 8;
        out += kB64[(n >> 18) & 63];
        out += kB64[(n >> 12) & 63];
        out += two ? kB64[(n >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

static std::vector<std::uint8_t> base64_decode(const std::string& in) {
    int rev[256]; for (int i = 0; i < 256; ++i) rev[i] = -1;
    for (int i = 0; i < 64; ++i) rev[(unsigned char)kB64[i]] = i;
    std::vector<std::uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    std::uint32_t buf = 0; int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int v = rev[(unsigned char)c];
        if (v < 0) continue;
        buf = (buf << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((buf >> bits) & 0xFF); }
    }
    return out;
}


static Value node_to_json(const FRepNode& node) {
    Object o;
    o["type"] = std::string(node.type_name());
    o["id"]   = node.id;

    // params
    Object params;
    for (const auto& [k, v] : node.params)
        params[k] = static_cast<double>(v);
    o["params"] = std::move(params);

    // An Instance references another object's geometry by id — it must NOT
    // serialize its (shared) child inline, or the file would carry a full copy
    // and load would break the sharing. Store only the target id; the child is
    // rebound from the scene after load (resolve_instances).
    if (node.kind == NodeKind::Instance) {
        o["target_id"] = static_cast<const InstanceNode&>(node).target_id();
        return Value(std::move(o));
    }

    // children (recursive)
    if (!node.children.empty()) {
        Array kids;
        for (const auto& c : node.children)
            kids.push_back(node_to_json(*c));
        o["children"] = std::move(kids);
    }

    // CustomExpr also carries a text expression.
    // (No dynamic_cast — the core lib is compiled with -fno-rtti because of LLVM.)
    if (std::string(node.type_name()) == "CustomExpr") {
        // CustomExprNode keeps the expression behind a separate getter.
        // Accessed via static_cast, safe because type_name guarantees the type.
        const auto& ce = static_cast<const CustomExprNode&>(node);
        o["expr"] = ce.expression();
        // A finite bounds hint (set on instanced prototypes for the RTX BVH)
        // is round-tripped; an infinite default is omitted.
        auto b = ce.aabb();
        if (std::isfinite(b.min_x) && std::isfinite(b.max_x))
            o["aabb"] = Array{b.min_x, b.min_y, b.min_z, b.max_x, b.max_y, b.max_z};
    }

    return Value(std::move(o));
}

std::string serialize_scene(const SceneGraph& scene,
                            const std::string& base_dir,
                            bool embed_textures)
{
    Object root;
    root["version"] = std::string("4.0");

    // camera
    const auto& cam = scene.camera();
    Object cam_obj;
    cam_obj["position"]   = Array{cam.position[0], cam.position[1], cam.position[2]};
    cam_obj["target"]     = Array{cam.target[0],   cam.target[1],   cam.target[2]};
    cam_obj["up"]         = Array{cam.up[0],       cam.up[1],       cam.up[2]};
    cam_obj["fov_deg"]    = static_cast<double>(cam.fov_deg);
    cam_obj["projection"] = std::string(
        cam.projection == Camera::Projection::Orthographic
            ? "orthographic" : "perspective");
    cam_obj["ortho_size"] = static_cast<double>(cam.ortho_size);
    root["camera"] = std::move(cam_obj);

    // objects
    Array objects;
    for (const auto& [id, obj] : scene.objects()) {
        Object jo;
        jo["id"]      = id;
        jo["visible"] = obj.visible;

        Object mat;
        mat["albedo"]  = Array{obj.material.albedo[0],
                               obj.material.albedo[1],
                               obj.material.albedo[2]};
        mat["albedo2"] = Array{obj.material.albedo2[0],
                               obj.material.albedo2[1],
                               obj.material.albedo2[2]};
        // Pattern stored as a string so the JSON stays readable. Texture
        // pixels are NOT serialized — they'd balloon the file. The
        // recommended workflow is to keep textures alongside scenes
        // (future: a `texture_path` field referencing a BMP/PNG).
        const char* pat_name = "Solid";
        switch (obj.material.pattern) {
            case Material::Pattern::Solid:     pat_name = "Solid"; break;
            case Material::Pattern::Checker:   pat_name = "Checker"; break;
            case Material::Pattern::Stripes:   pat_name = "Stripes"; break;
            case Material::Pattern::GradientY: pat_name = "GradientY"; break;
            case Material::Pattern::Noise:     pat_name = "Noise"; break;
            case Material::Pattern::Texture:   pat_name = "Texture"; break;
        }
        mat["pattern"]       = std::string(pat_name);
        mat["pattern_scale"] = static_cast<double>(obj.material.pattern_scale);
        if (!obj.material.texture_path.empty())
            mat["texture_path"] = rewrite_relative(obj.material.texture_path,
                                                   base_dir);
        // Embed the texture pixels directly when asked to, or whenever the
        // material has pixels but no file path to load them from (procedural
        // textures, or textures built in-memory). Without this a procedural
        // texture is silently lost across serialize/deserialize — which
        // matters for distributed rendering, where the remote node rebuilds
        // the scene purely from the message and has no access to local files.
        if (obj.material.pattern == Material::Pattern::Texture &&
            !obj.material.texture_rgba.empty() &&
            (embed_textures || obj.material.texture_path.empty())) {
            mat["texture_width"]  = static_cast<double>(obj.material.texture_width);
            mat["texture_height"] = static_cast<double>(obj.material.texture_height);
            mat["texture_rgba_b64"] = base64_encode(obj.material.texture_rgba);
        }
        mat["roughness"]     = static_cast<double>(obj.material.roughness);
        mat["metallic"]      = static_cast<double>(obj.material.metallic);
        mat["emission"]      = static_cast<double>(obj.material.emission);
        mat["reflectivity"]  = static_cast<double>(obj.material.reflectivity);
        jo["material"] = std::move(mat);

        jo["geometry"] = node_to_json(*obj.geometry);
        objects.push_back(Value(std::move(jo)));
    }
    root["objects"] = std::move(objects);

    // user-defined template functions (name, scalar params, body expression)
    if (!scene.templates().empty()) {
        Array tmpls;
        for (const auto& t : scene.templates().all()) {
            Object to;
            to["name"] = t.name;
            Array ps;
            for (const auto& p : t.params) ps.push_back(Value(p));
            to["params"] = std::move(ps);
            to["body"]   = t.body_src;
            tmpls.push_back(Value(std::move(to)));
        }
        root["templates"] = std::move(tmpls);
    }

    // lights
    Array lights;
    for (const auto& L : scene.lights()) {
        Object lo;
        lo["pos"]       = Array{L.pos[0], L.pos[1], L.pos[2]};
        lo["color"]     = Array{L.color[0], L.color[1], L.color[2]};
        lo["intensity"] = static_cast<double>(L.intensity);
        lights.push_back(Value(std::move(lo)));
    }
    root["lights"] = std::move(lights);

    return Value(std::move(root)).dump(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deserialize: JSON → FRepNode
//
// Factory: type-name -> constructor. Each constructor takes (params, children,
// id, optional expr) and returns a ready FRepNode::Ptr.
//
// A NodeDeserializer carries the optional PluginRegistry so unknown built-in
// types can be looked up as plugin-registered primitives.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

float param_or(const Object& p, const std::string& key, float def) {
    auto it = p.find(key);
    if (it == p.end()) return def;
    return it->second.as_float();
}

struct NodeDeserializer {
    const plugin::PluginRegistry* registry;

    // Helper: parses the children array.
    std::vector<FRepNode::Ptr> parse_children(const Value& jv) {
        std::vector<FRepNode::Ptr> kids;
        if (jv.has("children")) {
            for (const auto& cj : jv["children"].as_array())
                kids.push_back(json_to_node(cj));
        }
        return kids;
    }

    FRepNode::Ptr json_to_node(const Value& jv) {
        if (!jv.is_object())
            throw std::runtime_error("scene_io: node must be a JSON object");

        const std::string type = jv["type"].as_string();
        const std::string id   = jv.has("id") ? jv["id"].as_string() : type;
        const Object& params   = jv.has("params") ? jv["params"].as_object() : Object{};
        auto kids              = parse_children(jv);

        // ── Primitives ────────────────────────────────────────────────────────
        if (type == "Sphere") {
            return std::make_shared<SphereNode>(param_or(params, "r", 1.0f), id);
        }
        if (type == "Box") {
            return std::make_shared<BoxNode>(
                param_or(params, "hx", 1.0f),
                param_or(params, "hy", 1.0f),
                param_or(params, "hz", 1.0f), id);
        }
        if (type == "Plane") {
            return std::make_shared<PlaneNode>(
                param_or(params, "nx", 0.0f),
                param_or(params, "ny", 1.0f),
                param_or(params, "nz", 0.0f),
                param_or(params, "d",  0.0f), id);
        }

        // ── Operations ────────────────────────────────────────────────────────
        if (type == "Union") {
            if (kids.size() != 2) throw std::runtime_error("Union needs 2 children");
            return std::make_shared<UnionNode>(kids[0], kids[1], id);
        }
        if (type == "Intersection") {
            if (kids.size() != 2) throw std::runtime_error("Intersection needs 2 children");
            return std::make_shared<IntersectionNode>(kids[0], kids[1], id);
        }
        if (type == "Difference") {
            if (kids.size() != 2) throw std::runtime_error("Difference needs 2 children");
            return std::make_shared<DifferenceNode>(kids[0], kids[1], id);
        }
        if (type == "SmoothUnion") {
            if (kids.size() != 2) throw std::runtime_error("SmoothUnion needs 2 children");
            return std::make_shared<SmoothUnionNode>(
                kids[0], kids[1], param_or(params, "k", 0.4f), id);
        }
        if (type == "Negate") {
            if (kids.size() != 1) throw std::runtime_error("Negate needs 1 child");
            return std::make_shared<NegateNode>(kids[0], id);
        }

        // ── Transforms ────────────────────────────────────────────────────────
        if (type == "Translate") {
            if (kids.size() != 1) throw std::runtime_error("Translate needs 1 child");
            return std::make_shared<TranslateNode>(
                kids[0],
                param_or(params, "tx", 0.0f),
                param_or(params, "ty", 0.0f),
                param_or(params, "tz", 0.0f), id);
        }
        if (type == "Scale") {
            if (kids.size() != 1) throw std::runtime_error("Scale needs 1 child");
            // Non-uniform if sx/sy/sz present; else fall back to a uniform "s"
            // (old files, and the uniform GUI path).
            if (params.count("sx") || params.count("sy") || params.count("sz")) {
                float s = param_or(params, "s", 1.0f);
                return std::make_shared<ScaleNode>(
                    kids[0], param_or(params, "sx", s),
                    param_or(params, "sy", s), param_or(params, "sz", s), id);
            }
            return std::make_shared<ScaleNode>(
                kids[0], param_or(params, "s", 1.0f), id);
        }
        if (type == "RotateY") {
            if (kids.size() != 1) throw std::runtime_error("RotateY needs 1 child");
            return std::make_shared<RotateYNode>(
                kids[0], param_or(params, "a", 0.0f), id);
        }
        if (type == "RotateX") {
            if (kids.size() != 1) throw std::runtime_error("RotateX needs 1 child");
            return std::make_shared<RotateXNode>(
                kids[0], param_or(params, "a", 0.0f), id);
        }
        if (type == "RotateZ") {
            if (kids.size() != 1) throw std::runtime_error("RotateZ needs 1 child");
            return std::make_shared<RotateZNode>(
                kids[0], param_or(params, "a", 0.0f), id);
        }

        // ── Non-linear deformations ───────────────────────────────────────────
        // These were initially routed through the plugin registry, but
        // they're shipped as core types now — they appear in built-in
        // demos and gallery scenes, so the registry shouldn't be
        // required to round-trip them.
        if (type == "TwistY") {
            if (kids.size() != 1) throw std::runtime_error("TwistY needs 1 child");
            return std::make_shared<TwistYNode>(
                kids[0], param_or(params, "k", 1.0f), id);
        }
        if (type == "BendXY") {
            if (kids.size() != 1) throw std::runtime_error("BendXY needs 1 child");
            return std::make_shared<BendXYNode>(
                kids[0], param_or(params, "k", 0.5f), id);
        }
        if (type == "TaperY") {
            if (kids.size() != 1) throw std::runtime_error("TaperY needs 1 child");
            return std::make_shared<TaperYNode>(
                kids[0],
                param_or(params, "t", 0.5f),
                param_or(params, "h", 2.0f), id);
        }

        // ── Instance ──────────────────────────────────────────────────────────
        // References another object's geometry by id. Created unresolved (null
        // target) here because the target object may not be parsed yet; the
        // caller runs resolve_instances(scene) once all objects exist.
        if (type == "Instance") {
            std::string tgt = jv.has("target_id") ? jv["target_id"].as_string() : "";
            return std::make_shared<InstanceNode>(nullptr, tgt, id);
        }

        // ── CustomExpr ────────────────────────────────────────────────────────
        if (type == "CustomExpr") {
            std::string expr = jv.has("expr") ? jv["expr"].as_string() : "0.0";
            auto ce = std::make_shared<CustomExprNode>(expr, id);
            if (jv.has("aabb")) {   // finite bounds hint for the RTX BVH
                const auto& a = jv["aabb"].as_array();
                if (a.size() == 6)
                    ce->set_bounds({a[0].as_float(), a[1].as_float(), a[2].as_float(),
                                    a[3].as_float(), a[4].as_float(), a[5].as_float()});
            }
            return ce;
        }

        // ── Plugin-based primitives ───────────────────────────────────────────
        // For node types that are not built in (Torus, Octahedron, Capsule, etc.),
        // ask the plugin registry. The plugin's param_names() lists the parameter
        // keys in declaration order; we read them from the JSON params object in
        // that order and pass the resulting vector to the plugin's create().
        if (registry) {
            if (const auto* slot = registry->find_primitive_by_type_name(type)) {
                std::vector<float> ordered;
                ordered.reserve(slot->param_names.size());
                for (std::size_t i = 0; i < slot->param_names.size(); ++i) {
                    std::string key{slot->param_names[i]};
                    float def = i < slot->param_defaults.size()
                                ? slot->param_defaults[i] : 0.0f;
                    ordered.push_back(param_or(params, key, def));
                }
                return slot->create(ordered, id);
            }
        }

        throw std::runtime_error(
            "scene_io: unknown node type '" + type + "'. "
            "Plugin-based types require the plugin to be registered (pass a "
            "PluginRegistry* to deserialize_scene/load_scene).");
    }
};

} // namespace

// Rebind every InstanceNode's shared target pointer from the scene, by its
// target_id. Instances may be nested inside transforms, so we walk each object's
// whole tree. A target may itself contain instances, which is fine as long as
// the reference graph is acyclic; a cycle (A instances B, B instances A, or a
// self-reference) would make codegen recurse forever, so we detect it and leave
// the offending instance unresolved (renders empty) rather than hang.
//
// Two passes: first collect object roots by id, then walk and rebind. Cycle
// detection runs a DFS over the *instance reference* graph (id -> ids it
// instances) and refuses to bind any instance that participates in a cycle.
static void collect_instances(FRepNode* n, std::vector<InstanceNode*>& out) {
    if (!n) return;
    if (n->kind == NodeKind::Instance)
        out.push_back(static_cast<InstanceNode*>(n));
    for (auto& c : n->children)
        if (c) collect_instances(c.get(), out);
}

// Does binding `inst` (in object `owner_id`) to its target create a cycle?
// Walks target -> its instances -> ... looking for a path back to owner_id.
static bool creates_cycle(const SceneGraph& scene, const std::string& owner_id,
                          const std::string& target_id) {
    if (target_id == owner_id) return true;
    std::vector<std::string> stack{target_id};
    std::unordered_set<std::string> seen;
    while (!stack.empty()) {
        std::string cur = stack.back(); stack.pop_back();
        if (cur == owner_id) return true;
        if (!seen.insert(cur).second) continue;
        const SceneObject* o = scene.find_object(cur);
        if (!o || !o->geometry) continue;
        std::vector<InstanceNode*> insts;
        collect_instances(o->geometry.get(), insts);
        for (auto* in : insts) stack.push_back(in->target_id());
    }
    return false;
}

void resolve_instances(SceneGraph& scene, const plugin::PluginRegistry*) {
    for (auto& [oid, obj] : scene.objects_mutable()) {
        if (!obj.geometry) continue;
        std::vector<InstanceNode*> insts;
        collect_instances(obj.geometry.get(), insts);
        for (auto* in : insts) {
            const std::string& tgt = in->target_id();
            if (tgt.empty() || creates_cycle(scene, oid, tgt)) {
                in->rebind(nullptr);            // dangling / cyclic -> empty
                continue;
            }
            const SceneObject* t = scene.find_object(tgt);
            in->rebind(t ? t->geometry : nullptr);
        }
    }
}

std::vector<std::string>
find_dependent_instances(const SceneGraph& scene, const std::string& target_id) {
    std::vector<std::string> deps;
    for (const auto& [oid, obj] : scene.objects()) {
        if (oid == target_id || !obj.geometry) continue;
        std::vector<InstanceNode*> insts;
        collect_instances(obj.geometry.get(), insts);
        for (auto* in : insts)
            if (in->target_id() == target_id) { deps.push_back(oid); break; }
    }
    return deps;
}

SceneGraph deserialize_scene(const std::string& json_text,
                             const plugin::PluginRegistry* registry,
                             const std::string& base_dir)
{
    Value root = json::Value::parse(json_text);
    if (!root.is_object())
        throw std::runtime_error("scene_io: root must be a JSON object");

    SceneGraph scene;
    NodeDeserializer deser{registry};

    // camera
    if (root.has("camera")) {
        const Value& cj = root["camera"];
        auto read3 = [](const Value& arr) -> std::array<float, 3> {
            const auto& a = arr.as_array();
            return {a[0].as_float(), a[1].as_float(), a[2].as_float()};
        };
        if (cj.has("position")) scene.camera().position = read3(cj["position"]);
        if (cj.has("target"))   scene.camera().target   = read3(cj["target"]);
        if (cj.has("up"))       scene.camera().up        = read3(cj["up"]);
        if (cj.has("fov_deg"))  scene.camera().fov_deg   = cj["fov_deg"].as_float();
        if (cj.has("ortho_size"))
            scene.camera().ortho_size = cj["ortho_size"].as_float();
        if (cj.has("projection")) {
            const std::string& mode = cj["projection"].as_string();
            scene.camera().projection =
                (mode == "orthographic" || mode == "ortho")
                    ? Camera::Projection::Orthographic
                    : Camera::Projection::Perspective;
        }
    }

    // user-defined template functions — added before wiring the object nodes.
    if (root.has("templates")) {
        for (const auto& tj : root["templates"].as_array()) {
            std::string name = tj["name"].as_string();
            std::vector<std::string> params;
            if (tj.has("params"))
                for (const auto& p : tj["params"].as_array())
                    params.push_back(p.as_string());
            std::string body = tj.has("body") ? tj["body"].as_string() : std::string();
            std::string err = scene.templates().add(name, std::move(params), body);
            if (!err.empty())
                throw std::runtime_error("scene_io: " + err);
        }
    }

    // objects
    if (root.has("objects")) {
        for (const auto& oj : root["objects"].as_array()) {
            auto geom = deser.json_to_node(oj["geometry"]);

            Material mat;
            if (oj.has("material")) {
                const Value& mj = oj["material"];
                if (mj.has("albedo")) {
                    const auto& a = mj["albedo"].as_array();
                    mat.albedo = {a[0].as_float(), a[1].as_float(), a[2].as_float()};
                }
                if (mj.has("albedo2")) {
                    const auto& a = mj["albedo2"].as_array();
                    mat.albedo2 = {a[0].as_float(), a[1].as_float(), a[2].as_float()};
                }
                if (mj.has("pattern")) {
                    std::string p = mj["pattern"].as_string();
                    if      (p == "Checker")   mat.pattern = Material::Pattern::Checker;
                    else if (p == "Stripes")   mat.pattern = Material::Pattern::Stripes;
                    else if (p == "GradientY") mat.pattern = Material::Pattern::GradientY;
                    else if (p == "Noise")     mat.pattern = Material::Pattern::Noise;
                    else if (p == "Texture")   mat.pattern = Material::Pattern::Texture;
                    else                       mat.pattern = Material::Pattern::Solid;
                }
                if (mj.has("pattern_scale"))
                    mat.pattern_scale = mj["pattern_scale"].as_float();
                // Embedded pixels take priority over a file path: they are the
                // authoritative bytes the scene was saved with (and the only
                // source for a procedural texture, or on a remote distributed
                // node with no access to the original file).
                bool embedded_ok = false;
                if (mj.has("texture_rgba_b64") && mj.has("texture_width") &&
                    mj.has("texture_height")) {
                    auto pixels = base64_decode(mj["texture_rgba_b64"].as_string());
                    int w = mj["texture_width"].as_int();
                    int h = mj["texture_height"].as_int();
                    if (w > 0 && h > 0 &&
                        pixels.size() == static_cast<std::size_t>(w) * h * 4) {
                        mat.texture_rgba   = std::move(pixels);
                        mat.texture_width  = w;
                        mat.texture_height = h;
                        if (mj.has("texture_path"))
                            mat.texture_path = mj["texture_path"].as_string();
                        embedded_ok = true;
                    }
                }
                if (!embedded_ok && mj.has("texture_path")) {
                    // Save the path AS WRITTEN (likely relative) so a
                    // subsequent save can re-emit it relative again,
                    // rather than baking in this machine's absolute
                    // location. The renderer only needs the in-memory
                    // pixels and ignores texture_path itself.
                    mat.texture_path = mj["texture_path"].as_string();

                    // Load the pixels. We try in three steps, falling
                    // through silently until one succeeds:
                    //
                    //   1. As-is (raw path). Handles absolute paths AND
                    //      paths that are relative to the current
                    //      working directory rather than to the .frep
                    //      file — e.g. examples/06_textured_objects.json
                    //      stores "examples/textures/wood.bmp" assuming
                    //      the user runs frep_designer from the project
                    //      root. We try this FIRST so existing scenes
                    //      keep working.
                    //
                    //   2. Resolved against base_dir (.frep file's
                    //      directory). Handles the modern portable
                    //      layout where the scene file ships next to
                    //      its textures as a self-contained package —
                    //      see frep4-v4.0.1 release notes on
                    //      "relative-path resolution".
                    //
                    //   3. Last resort: bare filename portion against
                    //      base_dir. Handles the case where the
                    //      original author had textures in a sibling
                    //      directory ("textures/wood.bmp") but the
                    //      user copied just the .frep and the texture
                    //      files into a flat directory.
                    //
                    // The renderer falls back to solid `albedo` only
                    // if all three fail.
                    io::Image img;
                    img = io::load_image(mat.texture_path);
                    if (img.empty() && !base_dir.empty()) {
                        std::string resolved =
                            resolve_against(mat.texture_path, base_dir);
                        if (resolved != mat.texture_path)
                            img = io::load_image(resolved);
                    }
                    if (img.empty() && !base_dir.empty()) {
                        try {
                            fs::path p(mat.texture_path);
                            std::string flat =
                                (fs::path(base_dir) / p.filename())
                                    .lexically_normal().string();
                            img = io::load_image(flat);
                        } catch (...) {}
                    }
                    if (!img.empty()) {
                        mat.texture_rgba   = std::move(img.rgba);
                        mat.texture_width  = img.width;
                        mat.texture_height = img.height;
                    }
                }
                if (mj.has("roughness")) mat.roughness = mj["roughness"].as_float();
                if (mj.has("metallic"))  mat.metallic  = mj["metallic"].as_float();
                if (mj.has("emission"))  mat.emission  = mj["emission"].as_float();
                if (mj.has("reflectivity")) mat.reflectivity = mj["reflectivity"].as_float();
            }

            scene.add_object(geom, mat);

            // visibility — set_visibility if the object is hidden
            if (oj.has("visible") && !oj["visible"].as_bool())
                scene.set_visibility(geom->id, false);
        }
    }

    if (root.has("lights")) {
        auto& Ls = scene.lights();
        Ls.clear();
        for (const auto& lj : root["lights"].as_array()) {
            PointLight L;
            if (lj.has("pos")) {
                const auto& p = lj["pos"].as_array();
                L.pos = {p[0].as_float(), p[1].as_float(), p[2].as_float()};
            }
            if (lj.has("color")) {
                const auto& c = lj["color"].as_array();
                L.color = {c[0].as_float(), c[1].as_float(), c[2].as_float()};
            }
            if (lj.has("intensity"))
                L.intensity = lj["intensity"].as_float();
            Ls.push_back(L);
        }
    }

    resolve_instances(scene, registry);
    scene.wire_templates();   // point every CustomExpr node at scene.templates()
    return scene;
}

// ─────────────────────────────────────────────────────────────────────────────
// File wrappers
// ─────────────────────────────────────────────────────────────────────────────
bool save_scene(const SceneGraph& scene, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    // Use the directory containing the scene file as the base for
    // making texture paths portable. parent_path on a bare filename
    // (no slashes) is the empty string, which serialize_scene treats
    // as "no rewriting" — correct.
    std::string base_dir;
    try {
        base_dir = fs::path(path).parent_path().string();
    } catch (...) {}
    f << serialize_scene(scene, base_dir);
    return f.good();
}

SceneGraph load_scene(const std::string& path,
                      const plugin::PluginRegistry* registry) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("scene_io: cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string base_dir;
    try {
        base_dir = fs::path(path).parent_path().string();
    } catch (...) {}
    return deserialize_scene(ss.str(), registry, base_dir);
}

FRepNode::Ptr clone_node(const FRepNode& node,
                         const std::string& new_id,
                         const plugin::PluginRegistry* registry)
{
    // Serialize the subtree to a JSON value, then rewrite the id fields
    // so the clone doesn't collide with the original. The root takes
    // exactly `new_id`; each descendant keeps its relative structure
    // under that prefix. Finally reconstruct through the same
    // NodeDeserializer the loader uses, so plugin nodes work too.
    Value jv = node_to_json(node);

    // Recursive id-rewrite: root → new_id; child[i] → parent_id + "/cN".
    // Using the structural path as the suffix keeps ids stable and
    // unique regardless of the original names (two duplicated children
    // could otherwise share a name).
    std::function<void(Value&, const std::string&)> rewrite =
        [&](Value& v, const std::string& id) {
            if (!v.is_object()) return;
            Object& o = v.as_object();
            o["id"] = id;
            if (o.count("children")) {
                Array& kids = o["children"].as_array();
                for (std::size_t i = 0; i < kids.size(); ++i)
                    rewrite(kids[i], id + "/c" + std::to_string(i));
            }
        };
    rewrite(jv, new_id);

    NodeDeserializer deser{registry};
    return deser.json_to_node(jv);
}

} // namespace frep::io
