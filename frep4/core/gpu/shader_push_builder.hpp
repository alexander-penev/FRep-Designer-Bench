#pragma once
// core/gpu/shader_push_builder.hpp
//
// Helper for filling a ShaderPush struct from a SceneGraph. Used by
// CLI tools, the GUI viewport, and tests to avoid duplicating the
// boilerplate. Builds the orthonormal camera frame from
// `camera.position` and `camera.target`, copies up to 4 scene lights
// into the `lights[]` array (or supplies a default key light if the
// scene has none), and writes the resolution + FOV.

#include "core/frep/scene.hpp"
#include "core/gpu/vulkan_ctx.hpp"

#include <algorithm>
#include <cmath>

namespace frep::gpu {

inline ShaderPush build_push_from_scene(const SceneGraph& s, int width, int height,
                                        float /*fov_radians_unused*/ = 1.2f)
{
    ShaderPush p{};
    const auto& cam = s.camera();
    float fwd[3] = {
        cam.target[0] - cam.position[0],
        cam.target[1] - cam.position[1],
        cam.target[2] - cam.position[2]};
    float L = std::sqrt(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    if (L > 1e-6f) { fwd[0]/=L; fwd[1]/=L; fwd[2]/=L; }
    float wup[3] = {0, 1, 0};
    float right[3] = {
        fwd[1]*wup[2] - fwd[2]*wup[1],
        fwd[2]*wup[0] - fwd[0]*wup[2],
        fwd[0]*wup[1] - fwd[1]*wup[0]};
    L = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (L > 1e-6f) { right[0]/=L; right[1]/=L; right[2]/=L; }
    float up[3] = {
        right[1]*fwd[2] - right[2]*fwd[1],
        right[2]*fwd[0] - right[0]*fwd[2],
        right[0]*fwd[1] - right[1]*fwd[0]};

    for (int i = 0; i < 3; ++i) {
        p.cam_pos[i]   = cam.position[i];
        p.cam_fwd[i]   = fwd[i];
        p.cam_right[i] = right[i];
        p.cam_up[i]    = up[i];
    }

    const auto& lights = s.lights();
    int n_lights = static_cast<int>(std::min<std::size_t>(lights.size(), 4u));
    if (n_lights == 0) {
        p.light_count   = 1.0f;
        p.lights[0][0]  = 5;  p.lights[0][1] = 7;
        p.lights[0][2]  = 5;  p.lights[0][3] = 1.0f;
        p.light_colors[0][0] = 1.0f; p.light_colors[0][1] = 1.0f;
        p.light_colors[0][2] = 1.0f;
    } else {
        p.light_count = static_cast<float>(n_lights);
        for (int i = 0; i < n_lights; ++i) {
            p.lights[i][0] = lights[i].pos[0];
            p.lights[i][1] = lights[i].pos[1];
            p.lights[i][2] = lights[i].pos[2];
            p.lights[i][3] = lights[i].intensity;
            p.light_colors[i][0] = lights[i].color[0];
            p.light_colors[i][1] = lights[i].color[1];
            p.light_colors[i][2] = lights[i].color[2];
        }
    }

    // Camera projection — branch into perspective or ortho fields. The
    // hard-coded `fov_radians = 1.2` default is now ignored; we always
    // use scene.camera().fov_deg (converted to radians) for perspective
    // and scene.camera().ortho_size for ortho. The GLSL emitter's
    // main() reads pc.projection_mode and chooses the matching ray
    // origin / direction formula.
    constexpr float kPi = 3.14159265358979323846f;
    p.projection_mode = (cam.projection == Camera::Projection::Orthographic)
                            ? 1.0f : 0.0f;
    p.fov_scale       = std::tan(0.5f * cam.fov_deg * kPi / 180.0f);
    p.ortho_size      = cam.ortho_size;
    p.sphere_radius   = 0.9f;  // legacy field; unused by emitted shaders
    p.width           = width;
    p.height          = height;
    return p;
}

// Variant that uses an explicit camera + light set, no scene needed.
// Used by the hand-written sphere_trace.comp test harness.
inline ShaderPush build_push_simple(const float cam_pos[3], const float cam_target[3],
                                    const float light_pos[3], int W, int H,
                                    float fov_radians = 1.2f)
{
    ShaderPush p{};
    float fwd[3] = {
        cam_target[0] - cam_pos[0],
        cam_target[1] - cam_pos[1],
        cam_target[2] - cam_pos[2]};
    float L = std::sqrt(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    if (L > 1e-6f) { fwd[0]/=L; fwd[1]/=L; fwd[2]/=L; }
    float wup[3] = {0, 1, 0};
    float right[3] = {
        fwd[1]*wup[2] - fwd[2]*wup[1],
        fwd[2]*wup[0] - fwd[0]*wup[2],
        fwd[0]*wup[1] - fwd[1]*wup[0]};
    L = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (L > 1e-6f) { right[0]/=L; right[1]/=L; right[2]/=L; }
    float up[3] = {
        right[1]*fwd[2] - right[2]*fwd[1],
        right[2]*fwd[0] - right[0]*fwd[2],
        right[0]*fwd[1] - right[1]*fwd[0]};
    for (int i = 0; i < 3; ++i) {
        p.cam_pos[i]   = cam_pos[i];
        p.cam_fwd[i]   = fwd[i];
        p.cam_right[i] = right[i];
        p.cam_up[i]    = up[i];
    }
    p.light_count   = 1.0f;
    p.lights[0][0]  = light_pos[0];
    p.lights[0][1]  = light_pos[1];
    p.lights[0][2]  = light_pos[2];
    p.lights[0][3]  = 1.0f;
    // White light tint. The emitted shader multiplies every light's
    // contribution by pc.light_colors[i].xyz; without this the zero-initialised
    // colour makes all shading black (a blank render). build_push_from_scene
    // sets the same default — this variant must match.
    p.light_colors[0][0] = 1.0f;
    p.light_colors[0][1] = 1.0f;
    p.light_colors[0][2] = 1.0f;
    p.fov_scale     = std::tan(0.5f * fov_radians);
    p.sphere_radius = 0.9f;
    p.width         = W;
    p.height        = H;
    // Default tile = whole frame (tile_x1/y1 == 0 the shader reads as
    // width/height). A caller rendering a sub-region overwrites these.
    p.tile_x0 = 0; p.tile_y0 = 0;
    p.tile_x1 = 0; p.tile_y1 = 0;
    return p;
}

} // namespace frep::gpu
