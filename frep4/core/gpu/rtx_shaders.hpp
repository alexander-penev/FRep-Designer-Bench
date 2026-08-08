// core/gpu/rtx_shaders.hpp
//
// Ray-tracing shader generation for the GpuRtx path.
//
// The whole point of the RTX path is that it renders the *same* implicit SDF
// as the other three. To guarantee that, the intersection shader sphere-traces
// the exact `scene_sdf` GLSL the compute path uses: we take the GlslEmitter
// output and extract its shared region (the scene_sdf / scene_albedo / shade
// helpers, everything before `void main()`), then wrap it in the four RT
// stages. No second SDF emitter, so no chance of divergence.
//
// Four stages for a naive single-AABB scene:
//   raygen (.rgen)        — camera ray per pixel, traceRayEXT into the TLAS
//   intersection (.rint)  — sphere-trace scene_sdf inside the AABB, report t
//   closest-hit (.rchit)  — normal + scene_albedo + shade at the hit point
//   miss (.rmiss)         — background colour
//
// Each stage is returned as GLSL text targeting GL_EXT_ray_tracing; the
// executor compiles them to SPIR-V with glslang and links them into
// an RT pipeline.

#pragma once

#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"   // TracerConfig

#include <expected>
#include <string>

namespace frep::gpu {

struct RtShaderSet {
    std::string rgen;   // ray generation
    std::string rint;   // intersection (SDF sphere trace)
    std::string rchit;  // closest hit (shade)
    std::string rmiss;  // miss (background)

    // The shared GLSL region (scene_sdf + scene_albedo + shade helpers) that
    // was extracted from the compute emitter and embedded into rint/rchit.
    // Kept for diagnostics / tests.
    std::string shared_glsl;

    // RGBA8 texture pixels (packed as uint32, one per texel) for textured
    // scenes, to be bound at RT binding 2. Empty for analytic scenes. The
    // executor uploads these and passes them to rtx_trace.
    std::vector<std::uint32_t> texture_pixels;

    // Mesh voxel grid (float SDF samples) for mesh scenes, bound at RT
    // binding 3. Empty for non-mesh scenes.
    std::vector<float> mesh_voxels;
};

// Generate the RT shader set for a scene. Reuses GlslEmitter internally so the
// SDF/material/shading code is byte-identical to the compute path.
std::expected<RtShaderSet, std::string>
emit_rt_shaders(const SceneGraph& scene, const TracerConfig& cfg = {});

// ── Per-CSG-group shaders for the multi-BLAS broad-phase ─────────────────────
//
// One shared raygen/miss/closest-hit, plus one intersection shader per group,
// each sphere-tracing ONLY that group's sub-tree SDF (not the whole scene).
// The TLAS holds one instance per group pointing at the group's BLAS, and the
// SBT gives each instance its own hit record (its intersection shader), so a
// ray that enters group G's box runs G's intersection shader and evaluates
// only G's SDF. That's where the RT broad-phase pays off: groups a ray misses
// are never sphere-traced at all.
struct RtGroupShaderSet {
    std::string rgen;                       // shared
    std::string rchit;                      // shared (per-group path)
    std::string rmiss;                      // shared
    std::vector<std::string> rint_per_group; // one intersection shader / group
    std::string shared_glsl;                // diagnostics

    // Instanced path only. The closest-hit stage is emitted per prototype from
    // that prototype's own field, so it does not grow with the object count;
    // `rchit` above is left empty in that case. `rmiss_shadow` answers the
    // second ray type those shaders trace for occlusion — empty when shadows
    // are off, in which case no second miss record is needed.
    std::vector<std::string> rchit_per_group;
    std::string rmiss_shadow;
};

// Emit per-group shaders. `group_scenes[i]` is a SceneGraph holding only group
// i's sub-tree (caller builds these from partition_csg_groups). The shared
// stages are lifted from the *full* scene so sky/shade/normal match the other
// paths; each intersection shader is lifted from its group scene so it only
// evaluates that group's SDF.
std::expected<RtGroupShaderSet, std::string>
emit_rt_group_shaders(const SceneGraph& full_scene,
                      const std::vector<SceneGraph>& group_scenes,
                      const TracerConfig& cfg = {});

// ── Per-prototype shaders for the instanced broad-phase ──────────────────────
//
// Same shape as emit_rt_group_shaders, but each intersection shader marches in
// its prototype's OBJECT space (gl_ObjectRayOriginEXT / gl_ObjectRayDirectionEXT)
// rather than in world space. That is the whole difference between "one shader
// per placement" and "one shader per shape": a world-space shader has the
// placement baked into its SDF, so twenty-five columns need twenty-five
// shaders, while an object-space one is written once and the TLAS instance
// transform puts each copy where it belongs.
//
// `proto_scenes[i]` holds prototype i's body at the origin of its own space
// (from factor_rt_instances + make_proto_scenes). rint_per_group is then
// indexed by prototype, matching the SBT record offset each placement carries.
std::expected<RtGroupShaderSet, std::string>
emit_rt_instanced_shaders(const SceneGraph& full_scene,
                          const std::vector<SceneGraph>& proto_scenes,
                          const TracerConfig& cfg = {});

}  // namespace frep::gpu
