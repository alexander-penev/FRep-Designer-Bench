#pragma once
// core/exec/parity_scenes.hpp
//
// A library of small, focused scenes for cross-path equivalence testing.
// Each scene exercises one feature (a primitive, a CSG op, a transform, a
// deformation, a shading model, a material pattern, a mesh, …) so that path
// equivalence can be demonstrated across the whole feature surface, not just
// one busy scene. Used by:
//   - tests/test_path_parity.cpp  (CPU_IR vs CPU_IR self-check in the sandbox,
//     and the harness that runs on hardware against GPU paths)
//   - frep_multipath / a parity driver, to measure CPU↔GLSL on real hardware.
//
// Scenes are deliberately tiny (one or two objects + a floor) and share one
// camera/lighting setup, so a divergence localizes to the feature under test.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/frep/custom_expr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace frep::parity {

// Consistent camera + a single warm light, matching the equivalence work
// (the warm light is what surfaced the per-light-colour bug, so keeping it
// here keeps that path exercised).
inline void setup(SceneGraph& s) {
    s.camera().position = {3.5f, 2.5f, 4.5f};
    s.camera().target   = {0, 0, 0};
    s.camera().up       = {0, 1, 0};
    s.camera().fov_deg  = 45.0f;

    auto& L = s.lights(); L.clear();
    L.push_back({{6.0f, 8.0f, 5.0f}, {1.0f, 0.97f, 0.9f}, 1.0f});  // warm white
}

inline SceneGraph with_floor() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.6f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);
    return s;
}

// ── individual feature scenes ───────────────────────────────────────────────

inline SceneGraph scene_sphere() {
    auto s = with_floor();
    s.add_object(std::make_shared<SphereNode>(1.0f, "sph"),
                 Material{{0.85f, 0.3f, 0.25f}});
    setup(s); return s;
}

inline SceneGraph scene_box() {
    auto s = with_floor();
    s.add_object(std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "box"),
                 Material{{0.3f, 0.6f, 0.85f}});
    setup(s); return s;
}

inline SceneGraph scene_union() {
    auto s = with_floor();
    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "a"), -0.4f, 0, 0, "ta");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "b"), 0.4f, 0, 0, "tb");
    s.add_object(std::make_shared<UnionNode>(a, b, "u"),
                 Material{{0.4f, 0.8f, 0.4f}});
    setup(s); return s;
}

inline SceneGraph scene_intersection() {
    auto s = with_floor();
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "b"), 0.4f, 0.4f, 0, "tb");
    s.add_object(std::make_shared<IntersectionNode>(a, b, "i"),
                 Material{{0.8f, 0.6f, 0.3f}});
    setup(s); return s;
}

inline SceneGraph scene_difference() {
    auto s = with_floor();
    auto a = std::make_shared<SphereNode>(1.0f, "a");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b"), 0.5f, 0.5f, 0.5f, "tb");
    s.add_object(std::make_shared<DifferenceNode>(a, b, "d"),
                 Material{{0.7f, 0.4f, 0.8f}});
    setup(s); return s;
}

inline SceneGraph scene_smooth_union() {
    auto s = with_floor();
    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "a"), -0.5f, 0, 0, "ta");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.7f, "b"), 0.5f, 0, 0, "tb");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.3f, "su"),
                 Material{{0.3f, 0.7f, 0.7f}});
    setup(s); return s;
}

inline SceneGraph scene_twist() {
    auto s = with_floor();
    auto box = std::make_shared<BoxNode>(0.5f, 1.0f, 0.5f, "b");
    s.add_object(std::make_shared<TwistYNode>(box, 1.5f, "tw"),
                 Material{{0.85f, 0.7f, 0.2f}});
    setup(s); return s;
}

inline SceneGraph scene_bend() {
    auto s = with_floor();
    auto box = std::make_shared<BoxNode>(0.4f, 1.0f, 0.4f, "b");
    s.add_object(std::make_shared<BendXYNode>(box, 0.8f, "bend"),
                 Material{{0.4f, 0.7f, 0.85f}});
    setup(s); return s;
}

// Voxelized-mesh path: polygonize a sphere, then sample it back as a MeshSDF.
// Exercises the mesh SDF code on CPU_IR / GPU_IR (codegen) and GPU_GLSL
// (sample_mesh emitter); the RT path skips it like textures (no descriptor
// plumbing yet). Resolution kept modest so the build-time voxelization is fast.
inline SceneGraph scene_mesh() {
    SceneGraph src;
    src.add_object(std::make_shared<SphereNode>(0.8f, "s"));
    mesh::MarchingCubesParams mc;
    mc.rx = mc.ry = mc.rz = 32;
    mesh::Mesh m = mesh::extract_iso_mesh(src, mc);

    auto s = with_floor();
    s.add_object(std::make_shared<MeshSDFNode>(m, 32, "mesh"),
                 Material{{0.8f, 0.5f, 0.4f}});
    setup(s); return s;
}

inline SceneGraph scene_taper() {
    auto s = with_floor();
    auto box = std::make_shared<BoxNode>(0.6f, 1.0f, 0.6f, "b");
    s.add_object(std::make_shared<TaperYNode>(box, 0.5f, 2.0f, "tp"),
                 Material{{0.5f, 0.5f, 0.85f}});
    setup(s); return s;
}

inline SceneGraph scene_rotate() {
    auto s = with_floor();
    auto box = std::make_shared<BoxNode>(0.9f, 0.5f, 0.5f, "b");
    s.add_object(std::make_shared<RotateYNode>(box, 0.6f, "rot"),
                 Material{{0.8f, 0.5f, 0.5f}});
    setup(s); return s;
}

inline SceneGraph scene_scale() {
    auto s = with_floor();
    auto sph = std::make_shared<SphereNode>(1.0f, "s");
    s.add_object(std::make_shared<ScaleNode>(sph, 1.3f, "sc"),
                 Material{{0.6f, 0.6f, 0.3f}});
    setup(s); return s;
}

inline SceneGraph scene_customexpr() {
    // Gyroid clipped to a sphere — exercises the CustomExpr path on both
    // CPU codegen and the GLSL emitter.
    auto s = with_floor();
    auto g = std::make_shared<CustomExprNode>(
        "abs( sin(2*x)*cos(2*y) + sin(2*y)*cos(2*z) + sin(2*z)*cos(2*x) ) - 0.4",
        "gy");
    auto clip = std::make_shared<SphereNode>(1.4f, "clip");
    s.add_object(std::make_shared<IntersectionNode>(g, clip, "gx"),
                 Material{{0.5f, 0.2f, 0.55f}});
    setup(s); return s;
}

// ── material pattern scenes (procedural, no image textures) ──────────────────

inline SceneGraph scene_checker() {
    auto s = with_floor();
    Material m{{0.9f, 0.9f, 0.9f}};
    m.pattern = Material::Pattern::Checker;
    m.albedo2 = {0.1f, 0.1f, 0.1f};
    m.pattern_scale = 4.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "s"), m);
    setup(s); return s;
}

inline SceneGraph scene_stripes() {
    auto s = with_floor();
    Material m{{0.85f, 0.7f, 0.2f}};
    m.pattern = Material::Pattern::Stripes;
    m.albedo2 = {0.4f, 0.1f, 0.5f};
    m.pattern_scale = 8.0f;
    s.add_object(std::make_shared<SphereNode>(1.0f, "s"), m);
    setup(s); return s;
}

inline SceneGraph scene_gradient() {
    auto s = with_floor();
    Material m{{0.9f, 0.4f, 0.2f}};
    m.pattern = Material::Pattern::GradientY;
    m.albedo2 = {0.2f, 0.3f, 0.8f};
    s.add_object(std::make_shared<SphereNode>(1.0f, "s"), m);
    setup(s); return s;
}

inline SceneGraph scene_blinnphong() {
    // Same geometry as scene_sphere; the shading model is chosen via
    // TracerConfig at render time, so the harness renders this scene under
    // ShadingModel::BlinnPhong on both paths.
    return scene_sphere();
}

// ── image texture scene (now that all three paths sample textures) ───────────

inline SceneGraph scene_texture() {
    // A red/blue checkerboard image texture on a sphere, triplanar-mapped.
    // Exercises the image-texture path that is now implemented on CPU_IR
    // (embedded constant + IR triplanar sampling), GPU_IR (shared IR) and
    // GPU_GLSL (SSBO + shader sampling). The three must agree within the
    // shared-IR floor, same as every other feature.
    auto s = with_floor();
    Material m{{0.8f, 0.8f, 0.8f}};
    m.pattern = Material::Pattern::Texture;
    m.pattern_scale = 3.0f;
    const int tw = 8, th = 8;
    m.texture_width = tw; m.texture_height = th;
    m.texture_rgba.assign(tw * th * 4, 0);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            int i = (y * tw + x) * 4;
            bool c = (x + y) & 1;
            m.texture_rgba[i + 0] = c ? 210 : 40;
            m.texture_rgba[i + 1] = c ? 60  : 90;
            m.texture_rgba[i + 2] = c ? 40  : 210;
            m.texture_rgba[i + 3] = 255;
        }
    s.add_object(std::make_shared<SphereNode>(1.0f, "s"), m);
    setup(s); return s;
}

// ── the registry ─────────────────────────────────────────────────────────────

struct NamedScene { const char* name; SceneGraph (*make)(); };

inline std::vector<NamedScene> all_scenes() {
    return {
        {"sphere",        scene_sphere},
        {"box",           scene_box},
        {"union",         scene_union},
        {"intersection",  scene_intersection},
        {"difference",    scene_difference},
        {"smooth_union",  scene_smooth_union},
        {"twist",         scene_twist},
        {"bend",          scene_bend},
        {"taper",         scene_taper},
        {"rotate",        scene_rotate},
        {"scale",         scene_scale},
        {"customexpr",    scene_customexpr},
        {"checker",       scene_checker},
        {"stripes",       scene_stripes},
        {"gradient",      scene_gradient},
        {"mesh",          scene_mesh},
        {"texture",       scene_texture},
    };
}

} // namespace frep::parity
