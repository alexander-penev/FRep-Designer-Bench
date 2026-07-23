// core/gpu/rtx_query.hpp
//
// The ray-QUERY path for GpuRtx — the counterpart to the SBT ray-tracing
// pipeline (rtx_pipeline / rtx_shaders).
//
// Both paths use the SAME acceleration structure (RtAccel's TLAS) and the SAME
// sphere-trace math (scene_sdf / scene_normal / shade lifted from GlslEmitter).
// The difference is WHERE the march runs:
//
//   SBT pipeline (rtx_shaders.emit_intersection): the whole sphere-march lives
//     inside an *intersection shader*. The RT scheduler expects short
//     intersection shaders (a quadric = a few ops); a 100-step march there runs
//     at poor occupancy and the RT cores stall — so on a single-primitive scene
//     the path is ~5x the plain GLSL compute march (measured).
//
//   Ray query (this file): a plain COMPUTE shader runs rayQueryEXT for the
//     hardware BVH broad phase (which box?) and marches scene_sdf INLINE in the
//     compute shader at full occupancy — the same code the GLSL path runs, plus
//     hardware culling. This removes the intersection-shader tax.
//
// Requires VK_KHR_ray_query (RtxCtx::has_ray_query()). Single-BLAS (whole-scene
// box) scenes only for now — enough to isolate the intersection-shader tax on
// the case where the SBT path loses.

#pragma once

#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"
#include "core/gpu/rtx_pipeline.hpp"   // RtPushConstants, RtRenderResult
#include "core/gpu/rtx_shaders.hpp"    // TracerConfig, SceneGraph
#include "core/frep/scene.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace frep::gpu {

// Emit the ray-query compute shader for `scene`. Per pixel: build the primary
// ray (identical formula to the SBT raygen), rayQueryEXT against the TLAS,
// sphere-trace scene_sdf inside the candidate AABB, then scene_normal + shade —
// byte-identical to the SBT intersection+closest-hit, only the stage differs.
std::expected<std::string, std::string>
emit_ray_query_compute(const SceneGraph& scene, const TracerConfig& cfg);

// Trace via the ray-query compute pipeline against `accel` (reuse the SAME TLAS
// the SBT path builds — this is an apples-to-apples pipeline swap). `comp` is
// the compiled compute SPIR-V. Timing breakdown mirrors rtx_trace: pipeline_ms
// (build), trace_ms (dispatch submit->fence), readback_ms.
std::expected<RtRenderResult, std::string>
rtx_query_trace(const RtxCtx& ctx, const RtAccel& accel,
                const std::vector<std::uint32_t>& comp,
                const RtPushConstants& pc, int width, int height);

}  // namespace frep::gpu
