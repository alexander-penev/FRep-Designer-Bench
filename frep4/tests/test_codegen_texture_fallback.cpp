// tests/test_codegen_texture_fallback.cpp
//
// Originally a regression test for a crash in `emit_scene_material` when a
// scene contained a `Material::Pattern::Texture` object (the pattern switch
// had no Texture case, fell through with `t` unset, and segfaulted in LLVM).
//
// As of the texture-parity work, the CPU JIT path now actually *samples*
// image textures (triplanar nearest, matching the GLSL emitter) instead of
// falling back to solid albedo. These tests therefore check two things:
//   1. textured scenes still compile without crashing (the original contract)
//   2. a rendered textured surface shows the texture's colour variation, not
//      a single solid colour (the new behaviour)
// The cross-path equivalence (CPU sampled texture ≈ GPU_GLSL sampled texture)
// is checked on hardware via frep_parity_check with texture scenes.

#include "core/compiler/incremental.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/multipath.hpp"
#include "core/io/bmp_loader.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace {

// Build a one-sphere scene whose material is in Texture mode with a
// real RGBA8 buffer. The buffer is small (4×4) — we don't care about
// the rendered pixels, only that the compile succeeds without a
// segfault.
frep::SceneGraph make_textured_scene() {
    using namespace frep;
    SceneGraph s;
    auto sphere = std::make_shared<SphereNode>(1.0f, "tex_sphere");

    Material m;
    m.pattern        = Material::Pattern::Texture;
    m.pattern_scale  = 1.0f;
    m.albedo         = {0.9f, 0.7f, 0.2f};  // fallback colour
    m.texture_width  = 4;
    m.texture_height = 4;
    m.texture_rgba.resize(4 * 4 * 4, 0xCC);  // arbitrary content

    s.add_object(sphere, m);
    return s;
}

} // anonymous namespace

TEST(CodegenTextureFallback, CompilesWithoutCrash) {
    // The single test rep: compile a Texture-pattern scene. The bug
    // manifested as a hard crash inside LLVM during emit_scene_material;
    // if we reach the EXPECT line, the regression is gone.
    auto scene = make_textured_scene();
    frep::IncrementalCompiler compiler;
    EXPECT_NO_FATAL_FAILURE({
        compiler.compile_if_changed(scene);
    });
}

TEST(CodegenTextureFallback, SamplesTextureColourOnCpu) {
    // The behavioural contract is now that the CPU JIT samples the image
    // texture (not solid albedo). Render a sphere with a high-contrast
    // red/blue checkerboard texture head-on. A solid-albedo fallback would
    // paint every lit surface pixel the same hue; a sampled texture varies
    // across the surface. We check that the hue (R−B per pixel) takes a
    // range of values over the lit surface — impossible for a solid colour.
    using namespace frep;
    using namespace frep::exec;

    SceneGraph s;
    io::Image tx; tx.width = 4; tx.height = 4; tx.rgba.assign(4 * 4 * 4, 0);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            int i = (y * 4 + x) * 4;
            bool c = (x + y) & 1;
            tx.rgba[i + 0] = c ? 220 : 30;   // R
            tx.rgba[i + 1] = 30;             // G
            tx.rgba[i + 2] = c ? 30 : 220;   // B
            tx.rgba[i + 3] = 255;
        }
    Material m;
    m.pattern = Material::Pattern::Texture;
    m.texture_rgba = tx.rgba; m.texture_width = 4; m.texture_height = 4;
    m.pattern_scale = 3.0f;
    s.add_object(std::make_shared<SphereNode>(1.2f, "tex"), m);
    s.camera().position = {2.0f, 1.5f, 3.0f};
    s.camera().target   = {0, 0, 0};

    CpuIrExecutor e;
    auto r = e.render(s, 96, 96, Tile{0, 0, 96, 96});
    ASSERT_TRUE(r.ok) << r.error;

    // Collect R−B over lit surface pixels; a textured surface spans a range,
    // a solid one collapses to a single value.
    float lo = 1e9f, hi = -1e9f;
    int lit = 0;
    for (int i = 0; i < 96 * 96; ++i) {
        float R = r.rgba[i * 4 + 0], G = r.rgba[i * 4 + 1], B = r.rgba[i * 4 + 2];
        if (R + G + B < 0.05f) continue;          // skip background/shadow
        ++lit;
        float hue = R - B;
        lo = std::min(lo, hue);
        hi = std::max(hue, hi);
    }
    ASSERT_GT(lit, 200) << "object not in frame";
    // Texture has red (R−B≈+0.75) and blue (R−B≈−0.75) texels; even after
    // shading the lit surface must show a spread far wider than a solid
    // colour's (which would be ~0).
    EXPECT_GT(hi - lo, 0.2f)
        << "hue range " << (hi - lo) << " too narrow — texture not sampled";
}

TEST(CodegenTextureFallback, MixedSolidAndTexturePatternsCompile) {
    // The real scene that surfaced the crash had three objects: one
    // solid (floor), two textured (wbt, mbt). Cover that shape here
    // — multiple textured objects interleaved with non-textured ones
    // shouldn't cause the codegen switch to misalign.
    using namespace frep;
    SceneGraph s;

    auto floor = std::make_shared<BoxNode>(5.0f, 0.1f, 5.0f, "floor");
    Material mf;
    mf.pattern = Material::Pattern::Solid;
    mf.albedo  = {0.5f, 0.5f, 0.5f};
    s.add_object(floor, mf);

    for (int i = 0; i < 2; ++i) {
        auto sph = std::make_shared<SphereNode>(0.8f,
            std::string("tex_") + char('a' + i));
        Material m;
        m.pattern        = Material::Pattern::Texture;
        m.albedo         = {0.8f, 0.4f, 0.1f};
        m.texture_width  = 4;
        m.texture_height = 4;
        m.texture_rgba.resize(4 * 4 * 4, 0x80);
        s.add_object(sph, m);
    }

    frep::IncrementalCompiler compiler;
    EXPECT_NO_FATAL_FAILURE({
        compiler.compile_if_changed(s);
    });
}
