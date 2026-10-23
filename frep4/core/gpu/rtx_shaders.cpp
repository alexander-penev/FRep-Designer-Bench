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
#include "core/frep/scene_bounds.hpp"

#include <cstdio>

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
// like the compute path. Used for the single-BLAS path, for each per-group
// shader (where `shared` is the group's own sub-tree SDF), and — with
// `object_space` set — for each prototype on the instanced path.
//
// Object-space mode. On the instanced path the BLAS holds the prototype's own
// bounds and the TLAS instance carries the placement, so the traversal unit has
// already transformed the ray by the time this shader runs: gl_ObjectRay*
// is the world ray expressed in the prototype's coordinates. Marching there
// instead of in world space is what lets every placement of a shape share one
// shader, since the shader no longer knows or cares where the copy stands.
//
// Two conversions make that exact rather than approximate:
//
//   * gl_ObjectRayDirectionEXT is NOT renormalised — it is the world direction
//     pushed through world->object, so its length is the object units per world
//     unit along the ray. That is deliberate in the specification: it makes the
//     ray parameter t identical in both spaces, so the t we report is the world
//     t the hardware expects, with no rescaling.
//   * distances, however, do differ by that factor. An object-space step of `d`
//     advances t by d/|objDir|, and the world-space hit epsilon corresponds to
//     an object-space epsilon of eps*|objDir|. Both fall out of the same scalar.
//
// For a rigid placement (translation and rotation, the common case) the factor
// is exactly one and the arithmetic reduces to the world-space form.
//
// Object-space mode also carries the surface NORMAL out through the hit
// attribute, which is not an optimisation of this shader but of the next one.
// The closest-hit stage needs a normal, and taking it from the whole scene's
// field means differentiating the whole scene: at 256 instances that single
// call is 152k of the closest-hit shader's 194k SPIR-V words, because the
// analytic-gradient form carries a dual number through every term. Here the
// prototype's field is already inlined and already being marched, so the
// gradient is bounded by the prototype and costs a few more evaluations of
// something small. Attribute storage is 24 bytes against a 32-byte device
// limit.
//
// `carry_normal` is independent of `object_space`: the per-group path benefits
// from it too, and for the same reason. Its intersection shader marches the
// group's own sub-tree, so the gradient it can produce is the group's, and near
// a hit on group i the union's gradient IS group i's -- exactly the argument
// that makes the instanced case exact. Applying it to both is also what lets
// the two constructions be compared on instancing alone rather than on
// instancing plus this.
inline std::string emit_intersection(const char* hdr, const std::string& push,
                                     const std::string& shared,
                                     const TracerConfig& cfg,
                                     bool object_space = false,
                                     bool carry_normal = false,
                                     const FRepNode::AABB* clip_box = nullptr) {
    std::ostringstream s;
    s << hdr << push << shared;
    if (carry_normal)
        // Not named `hit`: the march below already has a local `bool hit`,
        // and the shadowed name turns hit.pos into a swizzle of a boolean.
        s << "struct RtHit { vec3 pos; vec3 nrm; };\n"
          << "hitAttributeEXT RtHit hit_attr;\n";
    else
        s << "hitAttributeEXT vec3 hit_pos;\n";
    s << "void main() {\n";
    if (object_space) {
        s << "    vec3 ro = gl_ObjectRayOriginEXT;\n"
          << "    vec3 rd = gl_ObjectRayDirectionEXT;   // unnormalised: |rd| = object units per world unit\n"
          << "    float sc = length(rd);\n"
          << "    if (sc < 1e-20) return;               // degenerate placement\n"
          << "    float dt_per_d = 1.0 / sc;            // object distance -> ray parameter\n";
    } else {
        s << "    vec3 ro = gl_WorldRayOriginEXT;\n"
          << "    vec3 rd = gl_WorldRayDirectionEXT;\n"
          << "    const float sc = 1.0, dt_per_d = 1.0;\n";
    }
    s << "    float t = max(gl_RayTminEXT, 0.001);\n"
      << "    float tmax = min(gl_RayTmaxEXT, " << cfg.max_dist << ");\n"
      // Distance thresholds live in the space we march in.
      << "    float eps = " << cfg.epsilon << " * sc;\n";
    // The whole-scene ray-box clip the compute and IR paths march within
    // (scene_bounds.hpp), so this path's march starts at the same t as theirs
    // and converges to the same hit: the BVH only gates which rays reach this
    // shader, it does not narrow where they start. World space only; a group
    // or instance marches its own box in object space.
    if (clip_box && !object_space) {
        // Same literal formatting as the compute emitter's flit(), so both
        // shaders clip at bit-identical planes.
        auto lit = [](float v) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.9g", v);
            std::string r(buf);
            if (r.find('.') == std::string::npos && r.find('e') == std::string::npos) r += ".0";
            return r;
        };
        s << "    {\n"
          << "        vec3 inv = 1.0 / rd;\n"
          << "        vec3 t1 = (vec3(" << lit(clip_box->min_x) << ", " << lit(clip_box->min_y) << ", "
          << lit(clip_box->min_z) << ") - ro) * inv;\n"
          << "        vec3 t2 = (vec3(" << lit(clip_box->max_x) << ", " << lit(clip_box->max_y) << ", "
          << lit(clip_box->max_z) << ") - ro) * inv;\n"
          << "        vec3 tmn = min(t1, t2);\n"
          << "        vec3 tmx = max(t1, t2);\n"
          << "        t    = max(t, max(max(tmn.x, tmn.y), tmn.z));\n"
          << "        tmax = min(tmax, min(min(tmx.x, tmx.y), tmx.z));\n"
          << "    }\n";
    }
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
      << "        if (d < eps && !sor_fail) { hit = true; last_d = d; break; }\n"
      << "        last_d = d;\n"
      << "        t += step_len * dt_per_d;\n"
      << "    }\n"
      << "    if (!hit && t <= tmax && last_d < eps * 80.0) hit = true;\n";
    // The closest-hit shader shades in world space, so the position attribute
    // has to be a world point even when the march ran in object space. t is the
    // same parameter in both, so the world point is just the world ray
    // evaluated at it.
    if (carry_normal)
        s << "    if (hit) {\n"
          << "        hit_attr.pos = " << (object_space
                 ? "gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * t"
                 : "p") << ";\n"
          << "        hit_attr.nrm = scene_normal(p);   // in the space we marched\n"
          << "        reportIntersectionEXT(t, 0u);\n"
          << "    }\n";
    else
        s << "    if (hit) { hit_pos = p; reportIntersectionEXT(t, 0u); }\n";
    s << "}\n";
    return s.str();
}

// Swap the body of a function out of emitted GLSL, keeping its signature. Used
// to redirect the shadow march at the acceleration structure. The alternative
// -- an emitter option threaded through TracerConfig -- would put a
// ray-tracing concern inside the compute emitter, which nothing else there
// knows about. `signature` must be the exact emitted prototype line; a miss
// returns false rather than silently leaving the original in place, because a
// silently unpatched shadow march would still *work*, just at O(N), and the
// whole point is that it must not.
bool replace_function_body(std::string& src, const std::string& signature,
                           const std::string& body) {
    std::size_t at = src.find(signature);
    if (at == std::string::npos) return false;
    std::size_t open = src.find('{', at + signature.size() - 1);
    if (open == std::string::npos) return false;
    int depth = 0;
    std::size_t i = open;
    for (; i < src.size(); ++i) {
        if (src[i] == '{') ++depth;
        else if (src[i] == '}') { if (--depth == 0) break; }
    }
    if (i >= src.size()) return false;
    src.replace(open + 1, i - open - 1, "\n" + body);
    return true;
}

// Closest-hit for the instanced path, in two forms.
//
// Both take the surface normal from the hit attribute rather than
// differentiating the scene, because the intersection shader already
// differentiated the prototype it hit. That alone is most of the stage: the
// analytic-gradient form carries a dual number through every term, so at 256
// instances one scene_normal call was 152k of 194k SPIR-V words. It is exact --
// near a hit on instance i the union's gradient IS instance i's gradient.
//
// The BOUNDED form goes further and is emitted from the prototype's field, so
// albedo and roughness come from the prototype's own material (which is why
// factoring makes material part of a prototype's identity) and nothing in the
// stage grows with the object count. Shadows then cannot be local -- a column
// must shadow its neighbour -- so the emitted shadow march is replaced by a
// second ray through the acceleration structure, which stops at the first hit
// and skips shading.
//
// That substitution is only legitimate when the shadow being replaced is a
// visibility test. With shadow_samples > 1 the emitter already writes a binary
// per-sample test and the softness comes from sampling the light's disk, so a
// first-hit ray is exactly equivalent. With a single sample it writes Inigo
// Quilez's penumbra estimate instead, which reads the *distance* to the nearest
// surface along the ray -- something no first-hit query returns. Measured, the
// substitution costs 4.5% of pixels beyond tolerance on a mixed scene, so the
// caller only asks for the bounded form when the shadow model permits it.
//
// Ambient occlusion is the residual approximation in the bounded form: it too
// wants a distance rather than a yes/no answer, and here it reads the
// prototype's field, so a copy standing very close to another will not darken
// the crease between them. Measured at 1.7% of pixels on the same scene.
inline std::string emit_instanced_closest_hit(const char* hdr,
                                              const std::string& push,
                                              const std::string& proto_shared,
                                              bool redirect_shadow,
                                              bool rotate_normal,
                                              std::string& error) {
    std::string shared = proto_shared;
    if (redirect_shadow) {
        const std::string sig = "float _shadow_ray(vec3 origin, vec3 dir, float max_t)";
        const std::string body =
            "    // Occlusion through the TLAS instead of a march over the whole\n"
            "    // field. TerminateOnFirstHit stops at the first occluder and\n"
            "    // SkipClosestHit means no shading runs for it; the payload is\n"
            "    // seeded blocked and only the shadow miss shader clears it.\n"
            "    _shadow_vis = 0.0;\n"
            "    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT\n"
            "                    | gl_RayFlagsTerminateOnFirstHitEXT\n"
            "                    | gl_RayFlagsSkipClosestHitShaderEXT,\n"
            "                0xFF, 0, 0, 1, origin, 0.001, dir, max_t, 1);\n"
            "    return _shadow_vis;\n";
        if (!replace_function_body(shared, sig, body)) {
            error = "rtx_shaders: could not redirect _shadow_ray at the TLAS";
            return {};
        }
    }

    std::ostringstream s;
    s << hdr
      << push;
    // Declared before the lifted region, which references both, and only when
    // the shadow march was actually redirected -- an unused acceleration
    // structure binding would still have to be declared and written.
    if (redirect_shadow)
        s << "layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;\n"
          << "layout(location = 1) rayPayloadEXT float _shadow_vis;\n";
    s << shared
      << "layout(location = 0) rayPayloadInEXT vec3 prd;\n"
      << "struct RtHit { vec3 pos; vec3 nrm; };\n"
      << "hitAttributeEXT RtHit hit_attr;\n"
      << "void main() {\n"
      << "    vec3 p = hit_attr.pos;\n";
    if (rotate_normal)
        s << "    // A normal transforms by the inverse transpose of object->world.\n"
          << "    // gl_WorldToObjectEXT IS that inverse, so multiplying the normal\n"
          << "    // on the LEFT by it applies the transpose: n * M == transpose(M) * n.\n"
          << "    vec3 n = hit_attr.nrm * mat3(gl_WorldToObjectEXT);\n";
    else
        s << "    vec3 n = hit_attr.nrm;   // already world space\n";
    s << "    float nl = length(n);\n"
      << "    n = nl > 1e-20 ? n / nl : vec3(0.0, 1.0, 0.0);\n"
      << "    prd = shade(p, n, -gl_WorldRayDirectionEXT);\n"
      << "}\n";
    return s.str();
}

// The RT path consumes instancing structurally -- one BLAS and one shader per
// prototype, placed by the TLAS -- so it does not also want the emitter's
// Level 2 sharing, which turns each repeated shape into a subprogram CALL
// inside the field. That trade is the wrong way round here: measured on a
// 256-instance grid it shrank the closest-hit shader by 7% and halved the
// trace, because the shadow and occlusion marches evaluate the field in a hot
// loop where a call does not inline away. Recognition still happens upstream;
// only this consumer opts out of the code-size half of it.
inline TracerConfig rt_cfg(const TracerConfig& cfg) {
    TracerConfig c = cfg;
    c.instance_shared_subprograms = false;
    return c;
}

// Miss shader for the shadow ray. Reached only when nothing was hit between the
// surface and the light, which is exactly the definition of unoccluded.
inline std::string emit_shadow_miss(const char* hdr) {
    std::ostringstream s;
    s << hdr
      << "layout(location = 1) rayPayloadInEXT float _shadow_vis;\n"
      << "void main() { _shadow_vis = 1.0; }\n";
    return s.str();
}

}  // namespace

std::expected<RtShaderSet, std::string>
emit_rt_shaders(const SceneGraph& scene, const TracerConfig& cfg_in) {
    const TracerConfig cfg = rt_cfg(cfg_in);
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
          << "layout(set = 0, binding = 1, rgba8) uniform image2D out_img;\n"
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
    // The whole-scene clip box, when the scene has one (scene_bounds.hpp).
    FRepNode::AABB clip{};
    const bool clipped = cfg.bbox_clip && scene_clip_box(scene, clip);
    set.rint = emit_intersection(hdr, push, shared_block, cfg, false, false,
                                 clipped ? &clip : nullptr);

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

// Shared body of the group and instanced emitters: the two differ only in
// whether each intersection shader marches in world or in object space.
static std::expected<RtGroupShaderSet, std::string>
emit_rt_per_group(const SceneGraph& full_scene,
                  const std::vector<SceneGraph>& group_scenes,
                  const TracerConfig& cfg_in, bool object_space) {
    const TracerConfig cfg = rt_cfg(cfg_in);
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
    out.rmiss = base->rmiss;
    // The per-group path keeps one closest-hit stage over the whole scene. The
    // instanced path emits one per prototype below, from the prototype's own
    // field, and needs a second miss shader for the occlusion ray those use.
    // Whether the closest-hit stage can be emitted from the prototype's field
    // instead of the scene's. It can only when the shadow term is a visibility
    // test -- off, or multi-sampled, where the emitter already writes a binary
    // per-sample march. A single-sample soft shadow is a distance estimate and
    // no first-hit ray reproduces it, so that config keeps a whole-scene stage
    // (still without the scene gradient, which the hit attribute now carries).
    const bool redirect_shadow = object_space && cfg.enable_shadows &&
                                 cfg.shadow_samples > 1;
    const bool bounded_chit = object_space &&
                              (!cfg.enable_shadows || cfg.shadow_samples > 1);
    out.rchit = std::string{};
    if (redirect_shadow) out.rmiss_shadow = emit_shadow_miss(hdr);

    // Not bounded: one whole-scene closest-hit, shared by every hit group. The
    // per-group path lands here too -- it keeps the whole-scene shading but
    // still takes its normal from the attribute, which is where four fifths of
    // that stage used to be.
    if (!bounded_chit) {
        std::string cerr;
        std::string ch = emit_instanced_closest_hit(hdr, push, full_shared,
                                                    /*redirect_shadow=*/false,
                                                    /*rotate_normal=*/object_space,
                                                    cerr);
        if (!cerr.empty()) return std::unexpected(cerr);
        out.rchit_per_group.push_back(std::move(ch));
    }

    // One intersection shader per group, each lifted from that group's own
    // scene so it sphere-traces ONLY that group's sub-tree SDF.
    out.rint_per_group.reserve(group_scenes.size());
    if (bounded_chit) out.rchit_per_group.reserve(group_scenes.size());
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
        out.rint_per_group.push_back(
            emit_intersection(hdr, push, gshared, cfg, object_space,
                              /*carry_normal=*/true));
        if (bounded_chit) {
            std::string cerr;
            std::string ch = emit_instanced_closest_hit(hdr, push, gshared,
                                                        redirect_shadow,
                                                        /*rotate_normal=*/true,
                                                        cerr);
            if (!cerr.empty())
                return std::unexpected(cerr + " (prototype " + std::to_string(i) + ")");
            out.rchit_per_group.push_back(std::move(ch));
        }
    }
    return out;
}

std::expected<RtGroupShaderSet, std::string>
emit_rt_group_shaders(const SceneGraph& full_scene,
                      const std::vector<SceneGraph>& group_scenes,
                      const TracerConfig& cfg) {
    return emit_rt_per_group(full_scene, group_scenes, cfg, /*object_space=*/false);
}

std::expected<RtGroupShaderSet, std::string>
emit_rt_instanced_shaders(const SceneGraph& full_scene,
                          const std::vector<SceneGraph>& proto_scenes,
                          const TracerConfig& cfg) {
    return emit_rt_per_group(full_scene, proto_scenes, cfg, /*object_space=*/true);
}

}  // namespace frep::gpu
