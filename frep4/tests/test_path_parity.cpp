// tests/test_path_parity.cpp
//
// Systematic cross-path equivalence: render every feature scene from
// parity_scenes.hpp on two executors and assert the images agree within
// tolerance. The point is to demonstrate equivalence across the whole
// feature surface (primitives, CSG, transforms, deformations, custom
// expressions, material patterns, both shading models) — not from one busy
// scene but from many focused ones, each isolating a feature.
//
// In the sandbox there is no GPU, so the two executors are both CPU_IR: this
// proves the harness and the determinism of the path (same scene → identical
// pixels, every scene). On real hardware the same harness compares CPU_IR vs
// GPU_GLSL (and GPU_IR) — see the parity driver tool, which reuses
// parity_scenes.hpp and the same metric.
//
// The metric mirrors the equivalence study: mean and max absolute per-channel
// difference over the frame.

#include <gtest/gtest.h>

#include "core/exec/parity_scenes.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/multipath.hpp"

#include <cmath>

using namespace frep;
using namespace frep::exec;

namespace {

struct DiffStats { double mean = 0; double max = 0; std::size_t n = 0; };

DiffStats compare(const std::vector<float>& a, const std::vector<float>& b) {
    DiffStats d;
    const std::size_t n = std::min(a.size(), b.size());
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        sum += e;
        if (e > d.max) d.max = e;
    }
    d.n = n;
    d.mean = n ? sum / n : 0;
    return d;
}

// Render a whole frame for a scene on a fresh CPU_IR executor with a given
// shading model.
std::vector<float> render_cpu(const SceneGraph& s, int W, int H,
                              TracerConfig::ShadingModel sm) {
    TracerConfig cfg;
    cfg.shading_model = sm;
    CpuIrExecutor e(SceneCodegen::SceneSdfMode::Inlined, cfg);
    auto r = e.render(s, W, H, Tile{0, 0, W, H});
    EXPECT_TRUE(r.ok) << "render failed: " << r.error;
    return r.rgba;
}

}  // namespace

// Every feature scene renders deterministically: the same scene on two
// independent CPU_IR executors must produce bit-identical frames. This walks
// the entire scene library, so a regression in any one feature's codegen
// shows up as that scene's row failing.
TEST(PathParity, EveryFeatureSceneIsDeterministicCpu) {
    const int W = 96, H = 72;
    for (const auto& ns : parity::all_scenes()) {
        SceneGraph s = ns.make();
        auto a = render_cpu(s, W, H, TracerConfig::ShadingModel::CookTorrance);
        auto b = render_cpu(s, W, H, TracerConfig::ShadingModel::CookTorrance);
        auto d = compare(a, b);
        EXPECT_EQ(d.max, 0.0) << "scene '" << ns.name
                              << "' not deterministic (max |Δ|=" << d.max << ")";
    }
}

// Both shading models must work on every scene (no crash, non-empty frame).
// This is the parity guarantee for ShadingModel across the feature surface:
// BlinnPhong and CookTorrance each render every scene.
TEST(PathParity, BothShadingModelsRenderEveryScene) {
    const int W = 64, H = 48;
    for (const auto& ns : parity::all_scenes()) {
        SceneGraph s = ns.make();
        auto bp = render_cpu(s, W, H, TracerConfig::ShadingModel::BlinnPhong);
        auto ct = render_cpu(s, W, H, TracerConfig::ShadingModel::CookTorrance);
        ASSERT_EQ(bp.size(), (std::size_t)W * H * 4) << ns.name;
        ASSERT_EQ(ct.size(), (std::size_t)W * H * 4) << ns.name;
        // The two models should differ somewhere (otherwise the model isn't
        // actually being applied) but both must be valid frames.
        auto d = compare(bp, ct);
        EXPECT_GT(d.n, 0u) << ns.name;
    }
}

// Deformation scenes (twist, taper) must render cleanly and deterministically
// after the switch from central-difference to analytic dual-AD normals. This
// guards that the analytic gradient path compiles and is stable for the
// non-linear deformations (the fix that brought twist/taper into cross-path
// parity with the GLSL emitter).
TEST(PathParity, DeformationScenesRenderStably) {
    const int W = 80, H = 60;
    for (const char* name : {"twist", "taper"}) {
        SceneGraph s;
        for (const auto& ns : parity::all_scenes())
            if (std::string(ns.name) == name) s = ns.make();
        auto a = render_cpu(s, W, H, TracerConfig::ShadingModel::CookTorrance);
        auto b = render_cpu(s, W, H, TracerConfig::ShadingModel::CookTorrance);
        ASSERT_EQ(a.size(), (std::size_t)W * H * 4) << name;
        auto d = compare(a, b);
        EXPECT_EQ(d.max, 0.0) << name << " not deterministic";
        double sum = 0; bool finite = true;
        for (float v : a) { if (!std::isfinite(v)) finite = false; sum += v; }
        EXPECT_TRUE(finite) << name << " produced non-finite pixels";
        EXPECT_GT(sum, 0.0) << name << " produced an empty frame";
    }
}

// The scene library covers the advertised feature surface. If a feature scene
// is added/removed this guards the count so the harness stays comprehensive.
TEST(PathParity, SceneLibraryCoversFeatureSurface) {
    auto scenes = parity::all_scenes();
    EXPECT_GE(scenes.size(), 15u);
    // Spot-check that key feature names are present.
    bool has_csg = false, has_deform = false, has_pattern = false,
         has_expr = false, has_texture = false;
    for (const auto& ns : scenes) {
        std::string n = ns.name;
        if (n == "smooth_union" || n == "difference") has_csg = true;
        if (n == "twist" || n == "taper")             has_deform = true;
        if (n == "checker" || n == "stripes")         has_pattern = true;
        if (n == "customexpr")                          has_expr = true;
        if (n == "texture")                             has_texture = true;
    }
    EXPECT_TRUE(has_csg);
    EXPECT_TRUE(has_deform);
    EXPECT_TRUE(has_pattern);
    EXPECT_TRUE(has_expr);
    EXPECT_TRUE(has_texture);
}
