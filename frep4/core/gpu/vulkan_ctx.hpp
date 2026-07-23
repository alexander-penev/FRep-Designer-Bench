#pragma once
// core/gpu/vulkan_ctx.hpp
//
// Minimal Vulkan compute context for running a sphere-tracing shader on
// a real GPU (or a software Vulkan driver such as Mesa's llvmpipe).
//
// Scope of this PoC:
//   - Instance (no surface — we're headless)
//   - First physical device with a compute-capable queue family
//   - Logical device + queue
//   - Storage image (RGBA8) for the render target
//   - Compute pipeline loaded from a precompiled .spv file
//   - Descriptor set bound to the storage image
//   - Push constants for camera + scene parameters
//   - Dispatch + readback to a CPU buffer
//
// What's deliberately omitted (no need for a CPU-style readback flow):
//   - Swapchain, surfaces, presentation queues
//   - Multiple queues, async compute
//   - Pipeline caching
//   - Validation layers in release (enabled in debug only)
//   - Memory budgeting beyond what we directly allocate
//
// API: construct, call render(), get an RGBA8 buffer back. The struct is
// not copyable; destruction tears down the Vulkan objects in the right
// order via RAII wrappers (deleter struct).

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace frep::gpu {

// Layout MUST match the `Push` block in gpu/sphere_trace.comp and
// the equivalent layout emitted by GlslEmitter. The struct stays
// under the Vulkan-guaranteed 128-byte push constant limit:
//
//   cam_pos+pad   16 B
//   cam_fwd+pad   16 B
//   cam_right+pad 16 B
//   cam_up+pad    16 B
//   lights[4]     64 B  (vec4 each: xyz = pos, w = intensity)
//   misc fields    8 B
//  ───────────────────
//   total        136 B → too big, so misc moves into the w channels of
//                       the camera vectors / unused slots.
//
// Final layout (128 B exactly):
//   cam_pos    + fov_scale         (16)
//   cam_fwd    + sphere_radius     (16)   — kept for shader compat
//   cam_right  + light_count_float (16)   — float so layout stays std140
//   cam_up     + width_float       (16)
//   lights[4]  (vec4 pos+intensity)(64)
//                                  ──
//                                  128
// The width / height pair is sent as float-encoded ints because std140
// requires 16-byte alignment for the trailing scalar.
struct ShaderPush {
    float cam_pos[3];        float fov_scale     = 1.0f;
    float cam_fwd[3];        float sphere_radius = 1.0f;
    float cam_right[3];      float light_count   = 1.0f;
    float cam_up[3];         float pad_w         = 0.0f;
    // Up to 4 point lights. Unused slots have intensity = 0.
    float lights[4][4]      = {};
    // Per-light colour (rgb in xyz; w unused). Parallel to lights[]. The
    // CPU/IR path tints each light's contribution by its colour; the GLSL
    // path previously ignored it (treated every light as white), which made
    // GLSL systematically brighter in the green/blue channels for a warm
    // light — a constant per-pixel offset over every lit surface. Packed
    // here so the GLSL shader can apply the same tint.
    float light_colors[4][4] = {};
    std::int32_t width      = 0;
    std::int32_t height     = 0;
    // Camera projection mode. 0.0 = perspective (rays from cam_pos
    // diverge through fov_scale), 1.0 = orthographic (rays parallel
    // along cam_fwd, offsets within an `ortho_size` window). A float
    // rather than int because some drivers convert ints to float for
    // push constants and the conversion can introduce subtle bugs.
    // GLSL emitter branches on `pc.projection_mode > 0.5`.
    float projection_mode   = 0.0f;
    // Half-width of the orthographic view in world units. Ignored
    // when projection_mode == 0. Set from scene.camera().ortho_size
    // by build_push_from_scene.
    float ortho_size        = 1.0f;
    // Temporal accumulation blend factor for real-time denoising.
    // result = mix(previous_pixel, new_pixel, accum_blend).
    //   1.0  → use the new frame outright (reset / first frame / moving)
    //   1/n  → running average over n frames (static camera, frame n)
    // The compute shader reads the previous value from the storage
    // image and writes the blended result back in place — each
    // invocation touches only its own pixel, so there's no cross-pixel
    // race. Set by build_push_from_scene from the renderer's frame
    // accumulation counter. Ignored by the offscreen path (which sets
    // it to 1.0).
    float accum_blend       = 1.0f;
    // Per-frame seed offset for stochastic sampling (soft shadows). It
    // changes every frame so that temporal accumulation actually
    // averages *different* jittered samples — without it, a static
    // camera would re-render an identical jitter pattern each frame and
    // accumulation would converge to nothing. Set from the renderer's
    // accumulation frame counter; the offscreen path leaves it at 0.
    float frame_seed        = 0.0f;
    // ── Tile rendering ──────────────────────────────────────────────────────
    // Sub-region of the frame this dispatch should compute, in pixel
    // coordinates: [tile_x0, tile_x1) × [tile_y0, tile_y1). The shader adds
    // (tile_x0, tile_y0) to its local invocation id to get the absolute
    // pixel, computes the ray from the FULL (width,height) frame so the
    // sub-region matches the corresponding part of a full render, and
    // writes into a tile-sized output buffer. Default {0,0,width,height}
    // (set by build_push_from_scene) = whole frame, so existing callers are
    // unaffected. int32 to match width/height (same std140 treatment).
    std::int32_t tile_x0 = 0;
    std::int32_t tile_y0 = 0;
    std::int32_t tile_x1 = 0;   // 0 means "use width"  (whole frame)
    std::int32_t tile_y1 = 0;   // 0 means "use height" (whole frame)
};
static_assert(sizeof(ShaderPush) <= 256,
    "Vulkan push constants typically limited to 128–256B");

// Diagnostics returned alongside the rendered pixels.
struct GpuRenderStats {
    std::string device_name;     // e.g. "llvmpipe (LLVM 20.1.2, 256 bits)"
    double      init_ms  = 0;    // instance + device + pipeline setup (total)
    double      render_ms = 0;   // dispatch + wait + readback
    int         width    = 0;
    int         height   = 0;

    // Breakdown of init_ms into its phases. The pipeline phase is the
    // prime suspect for the init blow-up on large shaders: it's where the
    // driver compiles SPIR-V to native GPU code, which can dominate (and
    // grows with shader size) on a heavily-unrolled scene. Splitting it
    // out confirms whether a multi-second/minute init is driver shader
    // compilation rather than buffer upload or device setup.
    double      init_device_ms   = 0;  // instance + physical/logical device + queue
    double      init_shader_ms   = 0;  // vkCreateShaderModule (SPIR-V ingest)
    double      init_pipeline_ms = 0;  // vkCreateComputePipelines (DRIVER COMPILE)
    double      init_buffers_ms  = 0;  // descriptor/storage buffer alloc + upload
    double      init_misc_ms     = 0;  // command pool, fence, image, the rest
};

class VulkanCtx {
public:
    // Construct + initialize. Loads the shader from `spv_path` (a path to
    // a precompiled .spv file). Returns either a ready-to-use context, or
    // a human-readable error string.
    //
    // If `mesh_voxels` is non-empty, an additional storage buffer is
    // allocated at descriptor binding 1 and the data is uploaded once.
    // If `texture_pixels` is non-empty, another storage buffer is
    // allocated at binding 2 holding packed RGBA8 image data.
    // The shader is expected to declare matching std430 buffers; the
    // GlslEmitter does this automatically when the scene contains the
    // corresponding nodes / materials.
    static std::expected<std::unique_ptr<VulkanCtx>, std::string>
    create(const std::string& spv_path,
           const std::vector<float>& mesh_voxels = {},
           const std::vector<std::uint8_t>& texture_pixels = {},
           const std::vector<float>& params = {});

    ~VulkanCtx();
    VulkanCtx(const VulkanCtx&)            = delete;
    VulkanCtx& operator=(const VulkanCtx&) = delete;

    // Renders into a CPU-side buffer of size width * height * 4 bytes
    // (RGBA8). The buffer is reallocated if the size changes.
    std::expected<void, std::string>
    render(const ShaderPush& push, std::vector<std::uint8_t>& out_rgba);

    // Re-upload the runtime parameter buffer (binding 3); a cheap memcpy into
    // the persistently-mapped buffer, so an interactive parameter edit reuses
    // the compiled pipeline. No-op if created without a parameter buffer.
    void update_params(const std::vector<float>& values);

    const GpuRenderStats& stats() const { return stats_; }

    // Quick availability probe — returns true if any Vulkan device exists.
    // Useful for tests that should skip when Vulkan is missing.
    static bool available();

private:
    VulkanCtx() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    GpuRenderStats stats_;
};

} // namespace frep::gpu
