// core/gpu/rtx_shaders.cpp — see rtx_shaders.hpp.
//
// We reuse GlslEmitter and lift its shared region — everything from the start
// of the helper/SDF declarations up to (but not including) `void main()` — so
// the RT intersection/closest-hit shaders evaluate byte-identical scene_sdf /
// scene_albedo / shade code. The compute `main()` (with its image bindings and
// dispatch indexing) is dropped; the RT stages provide their own entry points.
//
// Scope: scenes whose shared region is self-contained in terms of
// descriptor bindings (no MeshSDF / texture storage buffers yet). Those add
// set/binding plumbing that differs between compute and RT and are layered on
// in a later step; emit_rt_shaders reports if it sees them so we fail loudly
// rather than silently diverging.

#include "core/gpu/rtx_shaders.hpp"
#include "core/gpu/glsl_emitter.hpp"

#include <sstream>
#include <vector>
#include <cstring>

namespace frep::gpu {

namespace {

// Pull out the portion of the compute shader between the end of the push/
// bindings preamble and `void main()`. We keep the helper + scene_sdf +
// scene_albedo + shade functions; we drop the `#version`, layout/binding
// declarations (RT stages declare their own), and main().
//
// The shared functions only reference push-constant values via `pc.*`. The RT
// stages declare a compatible `pc` block, so the lifted code compiles
// unchanged.
std::string lift_shared_region(const std::string& comp_src,
                               std::string& error) {
    // The self-contained math/shading region runs up to main(). It normally
    // starts at the "float scene_sdf" forward-decl, but mesh/texture helper
    // functions (sample_mesh_*, _unpack_rgb) are emitted *before* that decl, and
    // scene_sdf calls them — so if any are present we must start the lift at the
    // earliest such helper, or the lifted region references undefined functions.
    std::size_t main_pos = comp_src.find("void main()");
    std::size_t start = comp_src.find("float scene_sdf");
    for (const char* helper : {"float sample_mesh_", "vec3 _unpack_rgb",
                               "vec3 sample_texture", "float sample_texture",
                               // Instancing Level 2: the shared _inst_fn_N /
                               // _inst_grad_fn_N subprograms are emitted before
                               // scene_sdf and called by it, so the lift must
                               // start at the earliest of them or the RT shaders
                               // reference undefined instance functions.
                               "float _inst_fn_", "Dual _inst_grad_fn_",
                               // User template functions frep_tmpl_<name> are
                               // likewise emitted before scene_sdf and called by
                               // it, so the lift must include them too.
                               "float frep_tmpl_"}) {
        std::size_t h = comp_src.find(helper);
        if (h != std::string::npos && h < start) start = h;
    }
    if (start == std::string::npos || main_pos == std::string::npos ||
        start >= main_pos) {
        error = "rtx_shaders: could not locate shared region in compute source";
        return {};
    }
    return comp_src.substr(start, main_pos - start);
}

// Lift the exact push-constant block from the compute source, so the RT
// stages use a byte-identical layout (any divergence would read camera/light
// data at the wrong offsets and silently break parity).
std::string lift_push_block(const std::string& comp_src, std::string& error) {
    std::size_t p = comp_src.find("layout(push_constant)");
    if (p == std::string::npos) {
        error = "rtx_shaders: no push_constant block in compute source";
        return {};
    }
    std::size_t end = comp_src.find("} pc;", p);
    if (end == std::string::npos) {
        error = "rtx_shaders: malformed push_constant block";
        return {};
    }
    return comp_src.substr(p, (end + 5) - p) + "\n";
}

// Emit an intersection shader that sphere-traces `shared`'s scene_sdf_v exactly
// like the compute path. Used for both the single-BLAS path and each per-group
// per-group shader (where `shared` is the group's own sub-tree SDF).
inline std::string emit_intersection(const char* hdr, const std::string& push,
                                     const std::string& shared,
                                     const TracerConfig& cfg) {
    std::ostringstream s;
    s << hdr << push << shared
      << "hitAttributeEXT vec3 hit_pos;\n"
      << "void main() {\n"
      << "    vec3 ro = gl_WorldRayOriginEXT;\n"
      << "    vec3 rd = gl_WorldRayDirectionEXT;\n"
      << "    float t = max(gl_RayTminEXT, 0.001);\n"
      << "    float tmax = min(gl_RayTmaxEXT, " << cfg.max_dist << ");\n";
    // Interval pre-skip (the RTX analog of the compute path's tile cull): bound
    // the field over the AABB of this ray's [t,tmax] segment; if the interval
    // doesn't straddle 0 the surface can't be in this segment, so skip the whole
    // sphere-trace. Emitted only when interval culling is on and sdf_ival was
    // lifted into `shared`. The hardware BVH already does broad-phase between
    // objects; this trims empty space *within* an object's AABB.
    const bool interval_on = cfg.cull_slabs > 0 &&
        (cfg.cull_method == TracerConfig::CullMethod::Interval ||
         cfg.cull_method == TracerConfig::CullMethod::Auto);
    if (interval_on && shared.find("sdf_ival") != std::string::npos) {
        s << "    {\n"
          << "        vec3 pa = ro + rd * t;\n"
          << "        vec3 pb = ro + rd * tmax;\n"
          << "        vec3 lo = min(pa, pb);\n"
          << "        vec3 hi = max(pa, pb);\n"
          << "        vec2 fi = sdf_ival(lo, hi);\n"
          << "        if (fi.x > 0.0 || fi.y < 0.0) return;   // segment can't contain the surface\n"
          << "    }\n";
    }
    s << "    bool hit = false; float last_d = 1e30; vec3 p = ro;\n"
      << "    float step_len = 0.0; float omega = " << cfg.over_relax << ";\n"
      << "    for (int i = 0; i < " << cfg.max_steps << "; ++i) {\n"
      << "        if (t > tmax) break;\n"
      << "        p = ro + rd * t;\n"
      << "        float d = scene_sdf_v(p);\n"
      << "        float radius = d * " << cfg.safety_factor << ";\n"
      << "        bool sor_fail = (omega > 1.0) && ((radius + last_d) < step_len);\n"
      << "        step_len = sor_fail ? (step_len * (1.0 - omega)) : (radius * omega);\n"
      << "        omega = sor_fail ? 1.0 : omega;\n"
      << "        if (d < " << cfg.epsilon << " && !sor_fail) { hit = true; last_d = d; break; }\n"
      << "        last_d = d;\n"
      << "        t += step_len;\n"
      << "    }\n"
      << "    if (!hit && t <= tmax && last_d < " << (cfg.epsilon * 80.0f)
      <<       ") hit = true;\n"
      << "    if (hit) { hit_pos = p; reportIntersectionEXT(t, 0u); }\n"
      << "}\n";
    return s.str();
}

}  // namespace

std::expected<RtShaderSet, std::string>
emit_rt_shaders(const SceneGraph& scene, const TracerConfig& cfg) {
    auto comp = GlslEmitter::emit(scene, cfg);
    if (!comp) return std::unexpected("rtx_shaders: " + comp.error());

    // Both mesh and textures are now supported on the RT path via dedicated
    // bindings (2=texture, 3=mesh); 0=TLAS, 1=output image are taken.
    std::string err;
    std::string shared = lift_shared_region(comp->source, err);
    if (!err.empty()) return std::unexpected(err);
    std::string push = lift_push_block(comp->source, err);
    if (!err.empty()) return std::unexpected(err);

    // Buffer declarations the shared region references live in the compute
    // bindings block (before scene_sdf), so they aren't lifted — prepend them
    // here with RT-specific binding numbers.
    std::string buf_decls;
    if (comp->texture_count > 0) {
        buf_decls +=
            "layout(std430, set = 0, binding = 2) readonly buffer TextureData {\n"
            "    uint texture_pixels[];\n"
            "} tex_data;\n";
    }
    if (comp->mesh_count > 0) {
        buf_decls +=
            "layout(std430, set = 0, binding = 3) readonly buffer MeshData {\n"
            "    float mesh_voxels[];\n"
            "} mesh_data;\n";
    }
    // shared region + its buffer declarations, used by rint/rchit/rgen.
    std::string shared_block = buf_decls + shared;

    RtShaderSet set;
    set.shared_glsl = shared_block;

    // Repack RGBA8 texture bytes into uint32 texels for the binding-2 buffer
    // (the GLSL declares `uint texture_pixels[]`). 4 bytes → 1 uint, little-
    // endian, matching the compute path's raw-byte upload + _unpack_rgb.
    if (comp->texture_count > 0 && !comp->texture_pixels.empty()) {
        const auto& bytes = comp->texture_pixels;
        std::size_t n = (bytes.size() + 3) / 4;
        set.texture_pixels.resize(n, 0u);
        std::memcpy(set.texture_pixels.data(), bytes.data(), bytes.size());
    }
    // Mesh voxels pass through as floats (binding-3 buffer).
    if (comp->mesh_count > 0)
        set.mesh_voxels = comp->mesh_voxels;

    const char* hdr =
        "#version 460\n"
        "#extension GL_EXT_ray_tracing : require\n";

    // ── Ray generation ──────────────────────────────────────────────────────
    // One ray per pixel. Build a camera ray from the push-constant basis,
    // matching how the compute path forms primary rays (perspective branch),
    // and traceRayEXT into the TLAS. Payload carries the shaded colour back.
    {
        std::ostringstream s;
        s << hdr
          << push
          << shared_block  // for sky_color_s (analytic scenes have no extra bindings)
          << "layout(location = 0) rayPayloadEXT vec3 prd;\n"
          << "layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;\n"
          << "layout(set = 0, binding = 1, rgba32f) uniform image2D out_img;\n"
          << "void main() {\n"
          << "    ivec2 pix = ivec2(gl_LaunchIDEXT.xy);\n"
          << "    int W = pc.width, H = pc.height;\n"
          << "    // Identical to the compute emitter's primary-ray formula:\n"
          << "    // pixel index (no +0.5 center offset), u scaled by aspect,\n"
          << "    // v flipped. Any deviation shifts the silhouette by a\n"
          << "    // fraction of a pixel and shows up as an edge fringe.\n"
          << "    float u = (2.0 * float(pix.x) / float(W) - 1.0)\n"
          << "            * (float(W) / float(H));\n"
          << "    float v = 1.0 - 2.0 * float(pix.y) / float(H);\n"
          << "    vec3 ro, rd;\n"
          << "    if (pc.projection_mode > 0.5) {\n"
          << "        ro = pc.cam_pos\n"
          << "           + pc.cam_right * (u * pc.ortho_size)\n"
          << "           + pc.cam_up    * (v * pc.ortho_size);\n"
          << "        rd = normalize(pc.cam_fwd);\n"
          << "    } else {\n"
          << "        ro = pc.cam_pos;\n"
          << "        rd = normalize(\n"
          << "            pc.cam_fwd\n"
          << "            + pc.cam_right * (u * pc.fov_scale)\n"
          << "            + pc.cam_up    * (v * pc.fov_scale));\n"
          << "    }\n"
          << "    // Seed the payload with the sky colour computed from the NDC\n"
          << "    // vertical coord (s = 0.5 + 0.5*v) — exactly the compute\n"
          << "    // path's primary-ray sky. On a miss the payload keeps this;\n"
          << "    // on a hit closest-hit overwrites it with the shaded colour.\n"
          << "    prd = sky_color_s(0.5 + 0.5 * v);\n"
          << "    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,\n"
          << "        ro, 0.001, rd, 1000.0, 0);\n"
          << "    imageStore(out_img, pix, vec4(prd, 1.0));\n"
          << "}\n";
        set.rgen = s.str();
    }

    // ── Intersection: sphere-trace scene_sdf inside the AABB ────────────────
    set.rint = emit_intersection(hdr, push, shared_block, cfg);

    // ── Closest hit: normal + shade ─────────────────────────────────────────
    {
        std::ostringstream s;
        s << hdr
          << push
          << shared_block
          << "layout(location = 0) rayPayloadInEXT vec3 prd;\n"
          << "hitAttributeEXT vec3 hit_pos;\n"
          << "void main() {\n"
          << "    vec3 p = hit_pos;\n"
          << "    // Same normal + shade as the compute path: scene_normal()\n"
          << "    // (lifted from the emitter) and shade(p, n, -rayDir).\n"
          << "    vec3 n = scene_normal(p);\n"
          << "    prd = shade(p, n, -gl_WorldRayDirectionEXT);\n"
          << "}\n";
        set.rchit = s.str();
    }

    // ── Miss: keep the sky the raygen already wrote into the payload ────────
    {
        std::ostringstream s;
        s << hdr
          << "layout(location = 0) rayPayloadInEXT vec3 prd;\n"
          << "void main() {\n"
          << "    // raygen seeded prd with sky_color_s(0.5+0.5*v); a miss\n"
          << "    // leaves it untouched so the NDC-based gradient is exact.\n"
          << "}\n";
        set.rmiss = s.str();
    }

    return set;
}

std::expected<RtGroupShaderSet, std::string>
emit_rt_group_shaders(const SceneGraph& full_scene,
                      const std::vector<SceneGraph>& group_scenes,
                      const TracerConfig& cfg) {
    if (group_scenes.empty())
        return std::unexpected("rtx_shaders: no groups to emit");

    // Shared stages come from the FULL scene, so sky/shade/normal are identical
    // to the other paths and to the single-BLAS path.
    auto full = GlslEmitter::emit(full_scene, cfg);
    if (!full) return std::unexpected("rtx_shaders(group): " + full.error());
    if (full->mesh_count > 0 || full->texture_count > 0)
        return std::unexpected("rtx_shaders(group): mesh/texture not supported "
                               "on the RT path yet");

    std::string err;
    std::string full_shared = lift_shared_region(full->source, err);
    if (!err.empty()) return std::unexpected(err);
    std::string push = lift_push_block(full->source, err);
    if (!err.empty()) return std::unexpected(err);

    const char* hdr =
        "#version 460\n"
        "#extension GL_EXT_ray_tracing : require\n";

    RtGroupShaderSet out;
    out.shared_glsl = full_shared;

    // Reuse the single-BLAS emitter to produce the shared raygen/closest-hit/
    // miss (identical to that path), then drop its single intersection shader.
    auto base = emit_rt_shaders(full_scene, cfg);
    if (!base) return std::unexpected(base.error());
    out.rgen  = base->rgen;
    out.rchit = base->rchit;
    out.rmiss = base->rmiss;

    // One intersection shader per group, each lifted from that group's own
    // scene so it sphere-traces ONLY that group's sub-tree SDF.
    out.rint_per_group.reserve(group_scenes.size());
    for (std::size_t i = 0; i < group_scenes.size(); ++i) {
        auto g = GlslEmitter::emit(group_scenes[i], cfg);
        if (!g) return std::unexpected("rtx_shaders(group " + std::to_string(i) +
                                       "): " + g.error());
        if (g->mesh_count > 0 || g->texture_count > 0)
            return std::unexpected("rtx_shaders(group " + std::to_string(i) +
                                   "): mesh/texture not supported");
        std::string gshared = lift_shared_region(g->source, err);
        if (!err.empty()) return std::unexpected(err);
        // push block is identical across groups (same camera/lights), so reuse
        // the full scene's; the group only changes scene_sdf.
        out.rint_per_group.push_back(emit_intersection(hdr, push, gshared, cfg));
    }
    return out;
}

}  // namespace frep::gpu
