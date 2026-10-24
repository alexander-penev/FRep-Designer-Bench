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
//     inside an *intersection shader*, reached through the pipeline's shader
//     stages.
//
//   Ray query (this file): a plain COMPUTE shader runs rayQueryEXT for the
//     hardware BVH broad phase (which box?) and marches scene_sdf INLINE in the
//     compute shader — the same code the GLSL path runs, plus hardware culling.
//
// What the two measure (RTX 2080, 512², GPU timestamps, October 2026): the
// same within noise -- sphere 0.056 against 0.057 ms of shader time, gyroid
// 0.063 against 0.070, colonnade 0.210 against 0.237 -- and both a fixed
// 0.02-0.03 ms above the compute path. The cost is the hardware ray cast
// itself, not the pipeline's stage transitions, which is what this control
// establishes; see docs/PERFORMANCE_TUNING.md. The executor runs this path
// under FREP_RTX_MODE=query (rtx_query_trace_cached below).
//
// Requires VK_KHR_ray_query (RtxCtx::has_ray_query()). Single-BLAS scenes
// (one box, or the shell cells of rtx_cells.hpp).

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
// With `cells` the candidate boxes are the field's shell cells (rtx_cells.hpp),
// read from the box list bound at binding 2; otherwise the march is clipped to
// the whole-scene box exactly as the compute path clips it.
std::expected<std::string, std::string>
emit_ray_query_compute(const SceneGraph& scene, const TracerConfig& cfg,
                       bool cells = false);

// Trace via the ray-query compute pipeline against `accel` (reuse the SAME TLAS
// the SBT path builds — this is an apples-to-apples pipeline swap). `comp` is
// the compiled compute SPIR-V. Timing breakdown mirrors rtx_trace: pipeline_ms
// (build), trace_ms (dispatch submit->fence), readback_ms.
std::expected<RtRenderResult, std::string>
rtx_query_trace(const RtxCtx& ctx, const RtAccel& accel,
                const std::vector<std::uint32_t>& comp,
                const RtPushConstants& pc, int width, int height);

// The cached ray-query trace: the counterpart of rtx_trace_cached for the
// compute pipeline. The pipeline, layouts and module live in `cache` while the
// SPIR-V is unchanged, and the frame resources (image, readback, command
// buffer, descriptor set, fence) are reused across frames exactly as the SBT
// path reuses them, so the two can be compared frame for frame. The box list
// is bound at binding 2 when `accel` holds shell cells. Defined next to the
// SBT trace (rtx_pipeline.cpp) because it shares its frame-resource helpers.
std::expected<RtRenderResult, std::string>
rtx_query_trace_cached(const RtxCtx& ctx, const RtAccel& accel,
                       RtxPipelineCache& cache,
                       const std::vector<std::uint32_t>& comp,
                       const RtPushConstants& pc, int width, int height,
                       bool want_rgba8 = false, bool float_image = false);

}  // namespace frep::gpu
