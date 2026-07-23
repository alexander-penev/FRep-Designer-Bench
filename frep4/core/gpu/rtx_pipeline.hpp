// core/gpu/rtx_pipeline.hpp
//
// RT pipeline + shader binding table + trace dispatch for the GpuRtx path
// Given a built acceleration structure (RtAccel) and the four RT
// shader SPIR-V modules, this creates the ray-tracing pipeline, assembles the
// SBT, binds the TLAS + output image, records vkCmdTraceRaysKHR, and reads the
// rendered image back to host floats (RGBA).
//
// Full-frame render (no tiling on the RT path). The
// executor crops to the requested tile after readback, mirroring how the
// compute path started before tile-limited dispatch.

#pragma once

#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace frep::gpu {

// Push-constant payload mirroring the GLSL `Push` block the shaders use. Laid
// out to match the compute emitter's block byte-for-byte (std430-ish scalar
// rules with the vec3+float packing the emitter relies on).
struct RtPushConstants {
    float cam_pos[3];    float fov_scale;
    float cam_fwd[3];    float sphere_radius;
    float cam_right[3];  float light_count;
    float cam_up[3];     float pad_w;
    float lights[4][4];
    float light_colors[4][4];
    std::int32_t width;
    std::int32_t height;
    float projection_mode;
    float ortho_size;
    float accum_blend;
    float frame_seed;
    std::int32_t tile_x0, tile_y0, tile_x1, tile_y1;
};

struct RtRenderResult {
    std::vector<float> rgba;  // width*height*4, row-major top-down
    int width  = 0;
    int height = 0;

    // Timing breakdown (ms), so the paper can separate the actual GPU trace
    // cost from the per-frame setup. A real
    // renderer amortizes pipeline_ms/sbt across frames; only trace_ms is the
    // recurring per-frame GPU cost.
    double pipeline_ms = 0.0;  // shader modules + RT pipeline + SBT build
    double trace_ms    = 0.0;  // vkCmdTraceRaysKHR submit→fence (the real work)
    double readback_ms = 0.0;  // image → host buffer copy + map
};

// Build the pipeline once per (shader set) and render. We keep it
// simple: one call does create+render+teardown. `rgen/rint/rchit/rmiss` are
// SPIR-V byte blobs (already compiled). Returns the full-frame image.
//
// `texture_pixels` (RGBA8 packed as uint32, one per pixel) is bound at RT
// binding 2 when non-empty; `mesh_voxels` (float SDF grid) at binding 3. Both
// match the buffers declared in the lifted shared region. Empty for analytic
// scenes.
std::expected<RtRenderResult, std::string>
rtx_trace(const RtxCtx& ctx, const RtAccel& accel,
          const std::vector<std::uint32_t>& rgen,
          const std::vector<std::uint32_t>& rint,
          const std::vector<std::uint32_t>& rchit,
          const std::vector<std::uint32_t>& rmiss,
          const RtPushConstants& pc, int width, int height,
          const std::vector<std::uint32_t>& texture_pixels = {},
          const std::vector<float>& mesh_voxels = {});

// Multi-BLAS trace. `accel` must be from RtAccel::build_groups with one
// instance per group; `rint_per_group[i]` is group i's intersection shader
// SPIR-V. The pipeline gets one hit group per group (shared closest-hit +
// per-group intersection), and the SBT hit region has one record per group so
// instance i (SBT offset i) runs its own intersection shader. raygen/closest-
// hit/miss are shared. This is the path where RT broad-phase culls groups a ray
// misses, so it should scale far better than the O(N) flat-union compute path.
std::expected<RtRenderResult, std::string>
rtx_trace_groups(const RtxCtx& ctx, const RtAccel& accel,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                 const std::vector<std::uint32_t>& rchit,
                 const std::vector<std::uint32_t>& rmiss,
                 const RtPushConstants& pc, int width, int height);

// Amortized RT setup across frames. Creating shader modules + the ray-tracing
// pipeline + SBT is the bulk of pipeline_ms; for an interactive viewport where
// only the camera moves (shaders unchanged), it's wasteful to rebuild them each
// frame. An RtxPipelineCache holds those shader-dependent objects and is reused
// while the SPIR-V is identical; it's keyed by a hash of the four shader blobs
// (+ the has_tex/has_mesh binding shape), and self-rebuilds when that changes.
//
// The caller owns the cache and must call .release(device) before the RtxCtx's
// device is destroyed (or let ~RtxPipelineCache do it while the device lives).
// Per-frame work — descriptor pool/set (bound to the TLAS + output image),
// the output image, and readback — is NOT cached and is redone each trace.
struct RtxPipelineCache {
    // Opaque cached handles (Vulkan types kept as void* to avoid leaking the
    // Vulkan headers into callers that only forward the cache).
    void*        pipeline = nullptr;        // VkPipeline
    void*        pipeline_layout = nullptr; // VkPipelineLayout
    void*        descriptor_layout = nullptr; // VkDescriptorSetLayout
    void*        shader_modules[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<void*> group_rint_modules;  // multi-BLAS: one VkShaderModule per group
    void*        sbt_buffer = nullptr;      // VkBuffer
    void*        sbt_memory = nullptr;      // VkDeviceMemory
    std::uint64_t sbt_base = 0;             // VkDeviceAddress of SBT
    std::uint32_t handle_size = 0;          // aligned group handle size
    std::uint64_t key = 0;                  // shader-set hash this cache is valid for
    bool          valid = false;

    // Per-frame render target, reused across frames while (w,h) are unchanged.
    // The rgba32f storage image and its readback buffer are two ~W*H*16-byte
    // allocations that otherwise recur every frame; caching them means a steady
    // camera pays neither the alloc nor the descriptor rewrite. Recreated on a
    // resize; released with the rest of the cache. Only used for analytic scenes
    // (no tex/mesh bindings) — those keep the per-frame path.
    void*         frame_image = nullptr;       // VkImage
    void*         frame_image_mem = nullptr;   // VkDeviceMemory
    void*         frame_image_view = nullptr;  // VkImageView
    void*         frame_readback = nullptr;    // VkBuffer
    void*         frame_readback_mem = nullptr;// VkDeviceMemory
    int           frame_w = 0, frame_h = 0;    // extent the above were built for

    // Release all cached GPU objects. Safe to call repeatedly; no-op if empty.
    void release(const RtxCtx& ctx);
    ~RtxPipelineCache() = default;  // caller must release() explicitly
};

// Cached variant of rtx_trace: identical result, but reuses `cache`'s pipeline
// + SBT when the shader set matches (rebuilding only on a shader change). Pass
// the same cache object across frames of an interactive session. With a warm
// cache, pipeline_ms drops to ~0 and only trace_ms + readback_ms recur.
std::expected<RtRenderResult, std::string>
rtx_trace_cached(const RtxCtx& ctx, const RtAccel& accel,
                 RtxPipelineCache& cache,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::uint32_t>& rint,
                 const std::vector<std::uint32_t>& rchit,
                 const std::vector<std::uint32_t>& rmiss,
                 const RtPushConstants& pc, int width, int height,
                 const std::vector<std::uint32_t>& texture_pixels = {},
                 const std::vector<float>& mesh_voxels = {});

// Cached variant of rtx_trace_groups: reuses the multi-BLAS pipeline + SBT +
// shader modules from `cache` while the group shader set is unchanged (only the
// per-frame descriptors/image/readback recur), so pipeline_ms drops to ~0 after
// the first frame. Same cache object type as the single-BLAS path.
std::expected<RtRenderResult, std::string>
rtx_trace_groups_cached(const RtxCtx& ctx, const RtAccel& accel,
                        RtxPipelineCache& cache,
                        const std::vector<std::uint32_t>& rgen,
                        const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                        const std::vector<std::uint32_t>& rchit,
                        const std::vector<std::uint32_t>& rmiss,
                        const RtPushConstants& pc, int width, int height);

}  // namespace frep::gpu
