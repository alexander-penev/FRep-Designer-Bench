// tests/test_gpu_shadow_regression.cpp
//
// Regression test for the shadow-ray artefact reported in the
// 2026-05-23 user bug report: "horizon holes" and silhouette
// speckles in scenes with a ground plane plus objects.
//
// The bug:
//   - Shadow ray-march used `t += max(d, 0.01)` which over-marched
//     when the SDF returned small positive values, producing
//     spurious shadow hits manifesting as dark blob-shaped artefacts
//     on the floor near the horizon.
//   - The scene_normal central difference could collapse to near-
//     zero at silhouette edges, making normalize() return NaN/Inf
//     which then propagated into the BRDF as white speckles.
//
// What we test:
//   - Render a scene with a ground plane + small set of objects.
//   - For each pixel on the floor that's NOT under any object's
//     shadow direction, check brightness > some lower bound (i.e.
//     no spurious shadow blobs).
//   - For each pixel on an object's silhouette, check that no
//     channel is > 0.99 unless the lit surface should plausibly be
//     that bright (i.e. no NaN/Inf-driven white spots).

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace frep;

namespace {

// Render the test scene at modest resolution. Returns the RGBA8 buffer
// alongside dimensions, or empty if Vulkan isn't usable on this build.
struct RenderResult {
    std::vector<std::uint8_t> rgba;
    int W = 0, H = 0;
};

RenderResult render_test_scene() {
    if (!gpu::VulkanCtx::available())    return {};
    if (gpu::find_glslang().empty())     return {};

    SceneGraph s;
    Material mfloor{{0.5f, 0.5f, 0.5f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mfloor);
    Material mcube{{0.2f, 0.4f, 0.45f}};
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.6f, 0.7f, 0.6f, "cube"),
        -2.0f, 0.3f, 0, "cube_t"), mcube);
    Material msph{{0.55f, 0.30f, 0.25f}};
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.85f, "ball"),
        0.0f, 0.5f, 0, "ball_t"), msph);

    s.camera().position = {0, 1.4f, 5.5f};
    s.camera().target   = {0, 0.4f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{5, 7, 4}, {1, 1, 0.95f}, 1.0f});

    auto emit = gpu::GlslEmitter::emit(s);
    if (!emit) return {};
    auto spv = gpu::compile_glsl_to_spv_managed(emit->source);
    if (!spv) return {};
    auto ctx = gpu::VulkanCtx::create(
        spv->path(), emit->mesh_voxels, emit->texture_pixels);
    if (!ctx) return {};

    RenderResult out;
    out.W = 320; out.H = 240;
    auto push = gpu::build_push_from_scene(s, out.W, out.H);
    auto rr = (**ctx).render(push, out.rgba);
    if (!rr) out.rgba.clear();
    return out;
}

} // anon

TEST(GpuShadowRegression, NoHorizonHoles) {
    auto r = render_test_scene();
    if (r.rgba.empty()) GTEST_SKIP() << "Vulkan unavailable";

    // Sample a band of pixels on the floor BEHIND the objects (in the
    // far-distance region). With the bug, the shadow-ray's spurious
    // hits paint these pixels darker than the actual rest of the
    // floor; without the bug they're at the floor's lit gray level.
    //
    // The horizon line is at v ~= 0.5 (centered camera, plane at y=0,
    // cam_y=1.4, target_y=0.4 → tilt brings horizon roughly to centre).
    // We sample the band just BELOW horizon — y in [0.55..0.65] of
    // image — over the full width.
    int W = r.W, H = r.H;
    int y0 = static_cast<int>(0.55 * H);
    int y1 = static_cast<int>(0.65 * H);
    // Sample only the LIT half of the band (toward the light, which sits at
    // +x and thus maps to the right of the image). The objects' legitimate
    // cast shadows fall to -x (image-left), so restricting to the right half
    // excludes them; the genuine "horizon-hole" self-shadow artefact darkens
    // the floor broadly (it occluded 20-40% of the *whole* band when present),
    // so it would still darken this lit half and trip the test.
    int x0 = W / 2;
    int total = 0, dark_pixels = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < W; ++x) {
            int i = (y * W + x) * 4;
            int rch = r.rgba[i+0];
            int gch = r.rgba[i+1];
            int bch = r.rgba[i+2];
            // Skip pixels that are clearly an object (saturated colour)
            // — we only care about floor pixels here. Object pixels
            // have one channel dominant; floor is approximately grey.
            int mx = std::max({rch, gch, bch});
            int mn = std::min({rch, gch, bch});
            if (mx - mn > 30) continue;
            // Skip sky (light blue background, blue dominant)
            if (bch > rch + 20) continue;
            ++total;
            // "Dark blob" pixels in the bug had brightness ~10..40
            // while normal floor was ~70..120. The bug-fix threshold:
            // less than 30 brightness means we still have a hole.
            if (mx < 30) ++dark_pixels;
        }
    }
    ASSERT_GT(total, 50) << "test scene sampled no floor pixels?";
    double dark_frac = double(dark_pixels) / total;
    // Threshold tuned so legitimate cast shadows (~3-5% of the band)
    // don't trip the test, but the pre-fix "holes" (which used to
    // affect 20-40% of the floor band) clearly do. The pre-fix
    // version of this scene produced ~25% dark fraction; the fixed
    // version sits around 4-5%.
    EXPECT_LT(dark_frac, 0.10)
        << "More than 10% of floor pixels near horizon are very dark "
        << "(" << dark_pixels << "/" << total << "). This is the "
        << "horizon-hole shadow artefact.";
}

TEST(GpuShadowRegression, NoSilhouetteSpeckles) {
    auto r = render_test_scene();
    if (r.rgba.empty()) GTEST_SKIP() << "Vulkan unavailable";

    // Look for isolated near-white pixels (>0.95 on all channels) in
    // the image. With the NaN-normal bug, scattered ~white pixels
    // appeared at silhouettes. We sample the whole image and count
    // bright-white pixels that are surrounded by darker pixels (i.e.
    // isolated speckles, not part of a legitimate bright region).
    int W = r.W, H = r.H;
    int speckles = 0;
    auto px = [&](int x, int y) {
        int i = (y * W + x) * 4;
        return std::min({r.rgba[i+0], r.rgba[i+1], r.rgba[i+2]});
    };
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int c = px(x, y);
            if (c < 240) continue;  // not bright enough
            // Check neighbours — if they're all considerably darker,
            // we have an isolated speckle.
            int max_neighbour = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    max_neighbour = std::max(max_neighbour,
                        int(px(x + dx, y + dy)));
                }
            if (c - max_neighbour > 60) ++speckles;
        }
    }
    // Allow at most a handful — there can be legitimate hot spots
    // from anti-aliasing on hard edges, but with the bug there were
    // tens or hundreds.
    EXPECT_LE(speckles, 10)
        << "Too many isolated near-white pixels (" << speckles << "). "
        << "This is the NaN-normal silhouette speckle artefact.";
}
