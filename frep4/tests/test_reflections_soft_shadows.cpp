// tests/test_reflections_soft_shadows.cpp
//
// Regression + smoke tests for the v4.2.0 shader features on the CPU
// JIT path:
//   - mirror reflections (TracerConfig.max_bounces > 0 + per-material
//     Material.reflectivity), implemented as a reflection bounce loop
//     in emit_tracer that reuses the factored-out shade_hit helper and
//     the new emit_scene_reflectivity lookup.
//   - soft shadows (TracerConfig.shadow_samples > 1) via unrolled
//     jittered shadow rays.
//
// The contract these tests assert is the *legal* one: the compile-and-
// render IR is valid (verifyModule passes inside SceneCodegen::verify_fn)
// and compile_if_changed returns a usable function pointer for every
// combination of the new knobs. We don't pixel-compare here — that's
// validated on real hardware against the GLSL emitter. The point is to
// guarantee the new IR paths don't crash LLVM or fail verification,
// across the matrix of (reflections on/off) × (soft shadows on/off).

#include "core/compiler/incremental.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/scene.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace {

using namespace frep;

// A small scene: a reflective sphere sitting above a large matte floor,
// lit by a single point light. This is the canonical "mirror ball on a
// plane" setup that exercises reflection rays that both hit another
// object (the floor) and escape to the sky.
SceneGraph make_reflective_scene() {
    SceneGraph s;

    auto floor = std::make_shared<BoxNode>(6.0f, 0.1f, 6.0f, "floor");
    Material mf;
    mf.albedo       = {0.6f, 0.6f, 0.6f};
    mf.reflectivity = 0.2f;   // slightly reflective floor
    s.add_object(floor, mf);

    auto ball = std::make_shared<SphereNode>(1.0f, "ball");
    Material mb;
    mb.albedo       = {0.9f, 0.3f, 0.25f};
    mb.metallic     = 1.0f;
    mb.roughness    = 0.1f;
    mb.reflectivity = 0.8f;   // strongly reflective sphere
    s.add_object(ball, mb);

    PointLight L;
    L.pos = {4.0f, 6.0f, 4.0f};
    s.lights().push_back(L);

    return s;
}

} // anonymous namespace

TEST(CpuReflections, CompilesWithReflectionsDisabled) {
    // Baseline: max_bounces = 0 (default). The reflection IR must not be
    // emitted at all, and the scene compiles exactly as it did pre-4.2.
    auto scene = make_reflective_scene();
    TracerConfig cfg;
    cfg.max_bounces = 0;
    IncrementalCompiler compiler(cfg);
    auto result = compiler.compile_if_changed(scene);
    EXPECT_TRUE(result.has_value())
        << (result.has_value() ? "" : result.error());
}

TEST(CpuReflections, CompilesWithSingleBounce) {
    auto scene = make_reflective_scene();
    TracerConfig cfg;
    cfg.max_bounces = 1;
    IncrementalCompiler compiler(cfg);
    auto result = compiler.compile_if_changed(scene);
    EXPECT_TRUE(result.has_value())
        << (result.has_value() ? "" : result.error());
}

TEST(CpuReflections, CompilesWithMultipleBounces) {
    // Mirror-in-mirror: several bounces. Exercises the unrolled bounce
    // block chain and the inter-bounce origin/normal threading.
    auto scene = make_reflective_scene();
    for (int bounces : {2, 3, 4}) {
        TracerConfig cfg;
        cfg.max_bounces = bounces;
        IncrementalCompiler compiler(cfg);
        auto result = compiler.compile_if_changed(scene);
        EXPECT_TRUE(result.has_value())
            << "bounces=" << bounces << ": "
            << (result.has_value() ? "" : result.error());
    }
}

TEST(CpuSoftShadows, CompilesWithMultipleSamples) {
    // Soft shadows alone (no reflections).
    auto scene = make_reflective_scene();
    for (int samples : {2, 4, 8, 16}) {
        TracerConfig cfg;
        cfg.shadow_samples      = samples;
        cfg.shadow_light_radius = 0.4f;
        IncrementalCompiler compiler(cfg);
        auto result = compiler.compile_if_changed(scene);
        EXPECT_TRUE(result.has_value())
            << "samples=" << samples << ": "
            << (result.has_value() ? "" : result.error());
    }
}

TEST(CpuReflections, CompilesWithReflectionsAndSoftShadows) {
    // The full combination: both features on at once. This is the
    // configuration most likely to expose an IR-dominance bug, since
    // the soft-shadow loop lives inside shade_hit which is itself
    // called from both the primary path and each reflection bounce.
    auto scene = make_reflective_scene();
    TracerConfig cfg;
    cfg.max_bounces         = 3;
    cfg.shadow_samples      = 8;
    cfg.shadow_light_radius = 0.3f;
    IncrementalCompiler compiler(cfg);
    auto result = compiler.compile_if_changed(scene);
    EXPECT_TRUE(result.has_value())
        << (result.has_value() ? "" : result.error());
}

TEST(CpuReflections, ZeroReflectivityMaterialsStillCompile) {
    // A scene where reflections are enabled globally but no material is
    // actually reflective. The bounce loop should early-out at runtime
    // (refl <= 0.001), but it must still compile cleanly.
    SceneGraph s;
    auto ball = std::make_shared<SphereNode>(1.0f, "ball");
    Material m;
    m.albedo       = {0.5f, 0.5f, 0.9f};
    m.reflectivity = 0.0f;   // not reflective
    s.add_object(ball, m);

    TracerConfig cfg;
    cfg.max_bounces = 2;
    IncrementalCompiler compiler(cfg);
    auto result = compiler.compile_if_changed(s);
    EXPECT_TRUE(result.has_value())
        << (result.has_value() ? "" : result.error());
}
