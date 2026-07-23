// gui/vulkan_viewport.cpp
//
// QVulkanWindow-based real-time viewport — the F4 "true real-time" path.
//
// This file ships:
//   1. A runtime capability probe (vulkan_viewport_available()) that
//      decides whether the swapchain path is usable on this build.
//   2. ComputeBlitRenderer — a QVulkanWindowRenderer that compiles the
//      scene to GLSL → SPV → compute pipeline, dispatches into a
//      private storage image at swapchain dimensions, then vkCmdBlitImage's
//      the result into the current swapchain image and presents.
//   3. A graceful container widget that handles the case where the
//      probe says yes but actual swapchain creation fails (which can
//      happen in containerised environments where a Vulkan driver
//      exists but no displayable surface is reachable).
//
// VERIFICATION CONSTRAINT
// ───────────────────────
// The development sandbox exposes only Mesa's `llvmpipe` software
// Vulkan driver, which `vulkan_viewport_available()` deliberately
// rejects — QVulkanWindow assumes a real swapchain capable of
// presenting to a window surface, which `llvmpipe` does not provide in
// the headless container. The implementation in this file therefore
// compiles and links cleanly but cannot be live-tested in CI; real
// hardware (or a desktop Mesa stack with a visible display) is needed
// to verify the actual rendering output.
//
// LAYOUT MANAGEMENT
// ─────────────────
// QVulkanWindow does NOT use the default render pass for us — when we
// override `startNextFrame()`, we own the command buffer. At entry to
// `startNextFrame()`, the swapchain image is in `PRESENT_SRC_KHR`
// layout (Qt presented the previous frame). Before vkQueuePresentKHR
// is called by `frameReady()`, the swapchain image must again be in
// `PRESENT_SRC_KHR`. The flow we use:
//
//   1. Transition storage image to GENERAL (oldLayout=UNDEFINED on
//      first frame, GENERAL on subsequent ones since we never leave
//      that layout once initialised).
//   2. Bind compute pipeline + descriptor set + push constants.
//   3. Dispatch ⌈w/8⌉ × ⌈h/8⌉ workgroups (matches local_size_x=8 in
//      sphere_trace.comp / GlslEmitter's emitted shader).
//   4. Barrier: storage image GENERAL → TRANSFER_SRC_OPTIMAL.
//   5. Barrier: swapchain image UNDEFINED → TRANSFER_DST_OPTIMAL
//      (UNDEFINED because we discard whatever was there).
//   6. vkCmdBlitImage from storage image to swapchain image.
//   7. Barrier: swapchain image TRANSFER_DST_OPTIMAL → PRESENT_SRC_KHR.
//   8. Call window_->frameReady() then window_->requestUpdate() to
//      keep frames flowing.

// Pull in vulkan.h BEFORE any Qt header that might transitively
// include qvulkaninstance.h. Qt's QVulkanInstance defines
// VK_NO_PROTOTYPES before its own include of vulkan.h to force
// driver-loaded function pointer access through QVulkanFunctions —
// but we use the loader-resolved global symbols (vkCreateInstance
// etc.) directly. Including vulkan.h first means its include guard
// is set and Qt's later attempt becomes a no-op, leaving the global
// symbols defined.
#include <vulkan/vulkan.h>

#include "gui/vulkan_viewport.hpp"
#include "gui/frep_vulkan_window.hpp"
#include "gui/viewport.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/picker.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/gpu/vulkan_ctx.hpp"

#include <QLabel>
#include <QMouseEvent>
#include <QPoint>
#include <QTimer>
#include <QVBoxLayout>
#include <QVulkanInstance>
#include <QVulkanWindow>
#include <QVulkanWindowRenderer>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace frep::gui {

// The original `namespace { ... }` block here has been opened up.
// FRepVulkanWindow (declared in gui/frep_vulkan_window.hpp) needs to
// hold a pointer to ComputeBlitRenderer, so the renderer can no
// longer be hidden in an anonymous namespace — it needs to be name-
// addressable from the header. Everything else in this translation
// unit (probe_real_vulkan_device, helper utilities, g_last_status,
// the renderer class itself) is now under frep::gui rather than
// anon-frep::gui, with `static` linkage where appropriate.

// Cached status message; populated by create() and read by last_status().
QString g_last_status;

// Decide whether to enable the real-time path. We're conservative — the
// Mesa llvmpipe software driver can technically expose VK_KHR_swapchain
// via the X11 surface extension on a real desktop, but in headless
// containers it cannot, and we don't want to crash MainWindow's startup.
bool probe_real_vulkan_device() {
    VkInstance inst = VK_NULL_HANDLE;
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "frep-realtime-probe";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS)
        return false;

    std::uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(inst, &pd_count, nullptr);
    if (pd_count == 0) { vkDestroyInstance(inst, nullptr); return false; }

    std::vector<VkPhysicalDevice> pds(pd_count);
    vkEnumeratePhysicalDevices(inst, &pd_count, pds.data());

    bool found_hw = false;
    for (auto pd : pds) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
            found_hw = true;
            break;
        }
    }
    vkDestroyInstance(inst, nullptr);
    return found_hw;
}

// Read an entire file into a byte buffer suitable for VkShaderModule.
// SPV is uint32-aligned on disk; the vector<uint32_t> output keeps
// alignment correct for vkCreateShaderModule's pCode field.
static std::vector<std::uint32_t> read_spv_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize n = f.tellg();
    if (n <= 0 || (n & 3)) return {};   // size must be a multiple of 4
    f.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> code(static_cast<std::size_t>(n / 4));
    if (!f.read(reinterpret_cast<char*>(code.data()), n)) return {};
    return code;
}

// Picks a memory type that satisfies both the required type bits
// (returned by vkGetImageMemoryRequirements) and the requested
// property flags (e.g. DEVICE_LOCAL_BIT). Returns UINT32_MAX on
// failure — caller must check.
static std::uint32_t pick_memory_type(VkPhysicalDevice phys,
                                      std::uint32_t   type_bits,
                                      VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

// Convenience: vkCmdPipelineBarrier wrapping a single image memory barrier.
static void image_barrier(VkCommandBuffer       cb,
                          VkImage               image,
                          VkImageLayout         old_layout,
                          VkImageLayout         new_layout,
                          VkAccessFlags         src_access,
                          VkAccessFlags         dst_access,
                          VkPipelineStageFlags  src_stage,
                          VkPipelineStageFlags  dst_stage)
{
    VkImageMemoryBarrier ib{};
    ib.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib.oldLayout           = old_layout;
    ib.newLayout           = new_layout;
    ib.srcAccessMask       = src_access;
    ib.dstAccessMask       = dst_access;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image               = image;
    ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ib.subresourceRange.levelCount = 1;
    ib.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0,
                         0, nullptr, 0, nullptr, 1, &ib);
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeBlitRenderer — actual F-Rep real-time renderer.
//
// Maintains the compute pipeline + intermediate storage image; on each
// frame, dispatches the scene SDF compute shader into the storage image
// and blits the result into the current swapchain image. Scene-structure
// changes (detected via SceneGraph::structural_hash composition) trigger
// a pipeline rebuild at the start of the next frame.
//
// All Vulkan handles are owned here; clean-up follows the canonical
// QVulkanWindow lifecycle: initResources/releaseResources for device-
// level objects, initSwapChainResources/releaseSwapChainResources for
// resolution-dependent ones (the storage image).
// ─────────────────────────────────────────────────────────────────────────────
class ComputeBlitRenderer : public QVulkanWindowRenderer {
public:
    ComputeBlitRenderer(QVulkanWindow* w, SceneGraph* scene)
        : window_(w), scene_(scene) {}

    void initResources() override {
        device_ = window_->device();
        phys_   = window_->physicalDevice();
        // The compute queue used by QVulkanWindow is the same as the
        // graphics+present queue; descriptors and commands all go
        // through `window_->graphicsQueue()` and `currentCommandBuffer()`.
        graphics_qfam_ = static_cast<std::uint32_t>(
            window_->graphicsQueueFamilyIndex());

        // One-time diagnostic dump: physical device + swapchain format
        // so we can verify after release what driver actually got used.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        const char* type_str = "?";
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   type_str = "discrete";   break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: type_str = "integrated"; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    type_str = "virtual";    break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            type_str = "CPU (software)"; break;
            default: break;
        }
        std::fprintf(stderr,
            "VulkanViewport: device='%s' type=%s api=%u.%u.%u\n",
            props.deviceName, type_str,
            VK_VERSION_MAJOR(props.apiVersion),
            VK_VERSION_MINOR(props.apiVersion),
            VK_VERSION_PATCH(props.apiVersion));
        std::fprintf(stderr,
            "VulkanViewport: swapchain format=%d colorSpace=%d depth=%d sample-count=%d\n",
            int(window_->colorFormat()),
            // QVulkanWindow doesn't expose colorSpace directly, but the
            // format enum carries SRGB info: format 43 = B8G8R8A8_SRGB,
            // 44 = R8G8B8A8_SRGB, 50 = B8G8R8A8_UNORM, ... — see VkFormat.
            -1,
            int(window_->depthStencilFormat()),
            int(window_->sampleCountFlagBits()));

        if (!rebuild_pipeline_from_scene()) {
            // Pipeline build failed — startNextFrame will detect
            // pipeline_ == VK_NULL_HANDLE and fall back to a clear.
            std::fprintf(stderr,
                "VulkanViewport: pipeline init failed: %s\n",
                build_status_.c_str());
        } else {
            std::fprintf(stderr,
                "VulkanViewport: pipeline built OK "
                "(scene_hash=0x%zx, mesh_buffer=%d, tex_buffer=%d)\n",
                scene_hash_, int(needs_mesh_buffer_), int(needs_tex_buffer_));
        }

        // ── Timestamp query pool ─────────────────────────────────────────────
        // Two timestamps per frame (start and end of the dispatch +
        // blit chain) give us a per-frame GPU-side render time. The
        // pool needs at least 2 slots; we use 4 (= 2 × MAX_FRAMES_IN_FLIGHT)
        // so frame N can record while frame N-2 is being read back.
        VkQueueFamilyProperties qfp[8];
        std::uint32_t qfp_count = 8;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qfp_count, qfp);
        if (qfp_count > graphics_qfam_
            && qfp[graphics_qfam_].timestampValidBits > 0)
        {
            VkQueryPoolCreateInfo qpi{};
            qpi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = TIMESTAMP_POOL_SIZE;
            if (vkCreateQueryPool(device_, &qpi, nullptr, &query_pool_)
                == VK_SUCCESS)
            {
                VkPhysicalDeviceProperties pp{};
                vkGetPhysicalDeviceProperties(phys_, &pp);
                ns_per_tick_ = pp.limits.timestampPeriod;
                std::fprintf(stderr,
                    "VulkanViewport: timestamp pool ready "
                    "(%u slots, %.2f ns/tick)\n",
                    TIMESTAMP_POOL_SIZE, ns_per_tick_);
            }
        }
    }

    void initSwapChainResources() override {
        const QSize sz = window_->swapChainImageSize();
        const int N = ssaa_factor_;
        std::fprintf(stderr,
            "VulkanViewport: swapchain resources created %dx%d (SSAA %dx)\n",
            sz.width(), sz.height(), N);
        create_storage_image(sz.width() * N, sz.height() * N);
    }

    // Update SSAA factor (1=off, 2=2x2, 3=3x3). Forces a swapchain
    // resource rebuild on the next frame so the storage image resizes
    // to the new SSAA dimensions.
    void set_ssaa(int n) {
        // The real-time path downsamples with a single bilinear blit,
        // which only box-filters correctly at even factors (see the
        // blit step). Snap odd factors up to the next even one so SSAA
        // always has a visible effect: 1→1 (off), 2→2, 3→4.
        n = std::clamp(n, 1, 4);
        if (n == 3) n = 4;
        if (n == ssaa_factor_) return;
        ssaa_factor_ = n;
        if (storage_image_ != VK_NULL_HANDLE) {
            // Force a swapchain-resource rebuild on the next frame.
            // We can't call releaseSwapChainResources here directly
            // (it's called by Qt on its own schedule); instead we
            // tear down + re-allocate the storage image now.
            if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
            destroy_storage_image();
            const QSize sz = window_->swapChainImageSize();
            create_storage_image(sz.width() * n, sz.height() * n);
            if (desc_set_ != VK_NULL_HANDLE)
                write_descriptor_image();
        }
    }

    void releaseSwapChainResources() override {
        destroy_storage_image();
    }

    void releaseResources() override {
        if (query_pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, query_pool_, nullptr);
            query_pool_ = VK_NULL_HANDLE;
        }
        destroy_pipeline();
    }

    void startNextFrame() override {
        VkCommandBuffer cb = window_->currentCommandBuffer();

        // Try to read the previous frame's GPU timing (if any). We
        // index the query pool by `frame_index_ % POOL_FRAMES`, so the
        // entry we're reading is the one written 2 frames ago — long
        // enough that the GPU has finished with it and we won't stall.
        if (query_pool_ != VK_NULL_HANDLE && frame_index_ >= POOL_FRAMES) {
            std::uint32_t slot_base =
                ((frame_index_ - POOL_FRAMES) % POOL_FRAMES) * 2;
            std::uint64_t ts[2] = {0, 0};
            VkResult qr = vkGetQueryPoolResults(
                device_, query_pool_, slot_base, 2,
                sizeof(ts), ts, sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (qr == VK_SUCCESS && ts[1] > ts[0]) {
                last_gpu_ns_ = static_cast<double>(ts[1] - ts[0])
                             * static_cast<double>(ns_per_tick_);
            }
        }

        // If the scene's structure changed since the pipeline was
        // compiled, rebuild it now. This is heavy (~5-50 ms on real HW)
        // so we only do it when `structural_hash` actually drifts.
        std::size_t cur_hash = compute_scene_hash();
        if (cur_hash != scene_hash_ && device_ != VK_NULL_HANDLE) {
            // We're inside a frame, so we need to wait for any in-flight
            // GPU work on the previous pipeline before destroying it.
            vkDeviceWaitIdle(device_);
            destroy_pipeline();
            rebuild_pipeline_from_scene();
            // Re-bind the storage image to the new descriptor set, since
            // the pool was torn down and re-created.
            if (storage_view_ != VK_NULL_HANDLE)
                write_descriptor_image();
        }

        if (pipeline_ == VK_NULL_HANDLE || storage_image_ == VK_NULL_HANDLE) {
            // Fallback: at least clear the swapchain to a recognisable
            // colour so the user sees the window is alive and the
            // pipeline failure is logged but not crashing.
            do_fallback_clear(cb);
            window_->frameReady();
            window_->requestUpdate();
            return;
        }

        const QSize sz = window_->swapChainImageSize();
        const int sc_w = sz.width();    // swapchain (presentation) dimensions
        const int sc_h = sz.height();
        const int N    = ssaa_factor_;  // 1 = off, 2 = 2x2, 3 = 3x3
        const int w    = sc_w * N;      // render-target (storage image) dimensions
        const int h    = sc_h * N;      // dispatch at this resolution

        // Reset this frame's query slot and write the start timestamp
        // before any GPU work. We index by frame_index_ % POOL_FRAMES
        // so the host-side readback (which runs 2 frames later) sees
        // exactly these slots.
        std::uint32_t this_slot = (frame_index_ % POOL_FRAMES) * 2;
        if (query_pool_ != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(cb, query_pool_, this_slot, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                query_pool_, this_slot);
        }

        // ── Temporal denoise: decide this frame's accumulation state ─────────
        // Build a key from everything that, if changed, should restart
        // the average: camera pose, projection, and the scene's dirty
        // revision. (Tracer-config / SSAA changes already force a full
        // pipeline rebuild, which recreates the renderer's state.) If
        // the key differs from last frame, the view changed → reset.
        const int accum_target = tracer_cfg_.accum_frames;
        float accum_blend = 1.0f;
        if (accum_target > 1 && scene_) {
            std::uint64_t key = 1469598103934665603ULL;
            auto mixb = [&key](const void* p, std::size_t n) {
                const auto* b = static_cast<const unsigned char*>(p);
                for (std::size_t i = 0; i < n; ++i) {
                    key ^= b[i]; key *= 1099511628211ULL;
                }
            };
            const auto& cam = scene_->camera();
            mixb(&cam.position, sizeof(cam.position));
            mixb(&cam.target,   sizeof(cam.target));
            mixb(&cam.up,       sizeof(cam.up));
            mixb(&cam.fov_deg,  sizeof(cam.fov_deg));
            int proj = static_cast<int>(cam.projection);
            mixb(&proj, sizeof(proj));
            std::uint64_t rev = scene_->revision();
            mixb(&rev, sizeof(rev));
            // Lights bypass the revision counter (edited via the non-const
            // lights() accessor), so fold them into the key directly.
            for (const auto& L : scene_->lights()) {
                mixb(&L.pos,       sizeof(L.pos));
                mixb(&L.color,     sizeof(L.color));
                mixb(&L.intensity, sizeof(L.intensity));
            }

            if (key != accum_key_) {
                accum_key_   = key;
                accum_count_ = 0;          // view changed → restart average
            }
            // Running mean: frame n (0-based) blends in at 1/(n+1), a
            // true average over n+1 frames. Once we've accumulated the
            // requested number of frames we FREEZE: blend drops to 0 so
            // no further (noisy) frames disturb the converged image, and
            // we stop requesting repaints (see end of startNextFrame) so
            // the GPU goes idle. Any view change resets accum_count_ and
            // wakes the loop back up.
            if (accum_count_ >= accum_target) {
                accum_blend     = 0.0f;     // frozen — keep converged result
                accum_converged_ = true;
            } else {
                accum_blend = 1.0f / static_cast<float>(accum_count_ + 1);
                accum_converged_ = false;
                ++accum_count_;
            }
        } else {
            // Accumulation off — every frame stands alone.
            accum_count_     = 0;
            accum_key_       = 0;
            accum_converged_ = false;
        }
        // Whether we keep the storage image's previous contents this
        // frame. We must preserve them whenever we're going to blend
        // (accum_blend < 1) AND the image actually holds a prior frame.
        const bool preserve = (accum_blend < 0.999f) && storage_has_content_;

        // ── (1) storage image → GENERAL for shader writes ────────────────────
        // When preserving (accumulation mid-average) we transition from
        // the GENERAL layout the image was left in, keeping its pixels
        // so the shader's imageLoad reads a valid previous frame. When
        // not preserving (reset / accumulation off) we use UNDEFINED,
        // which lets the driver discard — cheaper and avoids a stale
        // first frame after a resize.
        image_barrier(cb, storage_image_,
                      preserve ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_GENERAL,
                      preserve ? VK_ACCESS_SHADER_WRITE_BIT : 0,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // ── (2) bind + dispatch compute ──────────────────────────────────────
        // Dispatch at SSAA-scaled resolution. The compute shader knows
        // nothing about SSAA — it just renders one ray per pixel into
        // the storage image. The downsample happens in step (5) via
        // VK_FILTER_LINEAR during the blit.
        gpu::ShaderPush push = build_push();
        push.width       = w;
        push.height      = h;
        push.accum_blend = accum_blend;
        // Vary the stochastic seed each frame so accumulation averages
        // genuinely different jitter patterns. Using the accumulation
        // count means a reset (count→0) also resets the seed sequence,
        // keeping it deterministic per converged image.
        push.frame_seed  = static_cast<float>(accum_count_) * 0.6180339887f;
        storage_has_content_ = true;   // after this dispatch the image is valid

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout_, 0, 1, &desc_set_,
                                0, nullptr);
        vkCmdPushConstants(cb, pipe_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(gpu::ShaderPush), &push);
        std::uint32_t gx = static_cast<std::uint32_t>((w + 7) / 8);
        std::uint32_t gy = static_cast<std::uint32_t>((h + 7) / 8);
        vkCmdDispatch(cb, gx, gy, 1);

        // ── (3) storage image → TRANSFER_SRC for blit ────────────────────────
        image_barrier(cb, storage_image_,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);

        // ── (4) swapchain image → TRANSFER_DST ───────────────────────────────
        VkImage sc_image = window_->swapChainImage(window_->currentSwapChainImageIndex());
        image_barrier(cb, sc_image,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      0, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);

        // ── (5) vkCmdBlitImage: storage → swapchain ──────────────────────────
        // Source rect is full storage image (SSAA-scaled). Destination
        // is full swapchain. When ssaa_factor_ > 1 the filter is linear
        // so neighbouring sample texels mix into one swapchain pixel.
        //
        // IMPORTANT: a single bilinear blit only box-filters correctly
        // for EVEN supersample factors. At an even factor the output
        // pixel centre falls exactly between source texels, so the
        // bilinear tap averages a clean 2×2. At an ODD factor (e.g. 3×)
        // the output centre lands on the *centre* texel of the N×N
        // block, the bilinear weights of its neighbours go to zero, and
        // the blit effectively point-samples that one texel — no
        // averaging, so SSAA appears to do nothing. The SSAA control is
        // therefore restricted to even factors on the real-time path
        // (set_ssaa snaps odd values down); the offscreen GPU path uses
        // a true box downsample and can take any factor.
        VkImageBlit region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[1] = {w, h, 1};
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[1] = {sc_w, sc_h, 1};
        vkCmdBlitImage(cb,
                       storage_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       sc_image,       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region,
                       (N > 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);

        // ── (5b) storage image TRANSFER_SRC → GENERAL, preserving content ────
        // Return the storage image to GENERAL so next frame's
        // accumulation read (imageLoad) sees this frame's result. We
        // transition FROM TRANSFER_SRC_OPTIMAL (not UNDEFINED) so the
        // contents survive — this is what makes the temporal average
        // work without a second image.
        image_barrier(cb, storage_image_,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // ── (6) swapchain image → PRESENT_SRC_KHR ────────────────────────────
        image_barrier(cb, sc_image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        // End-of-frame timestamp. Paired with the begin-of-frame one
        // above, the delta is the GPU-side wall-clock for this frame's
        // dispatch + blit (excluding presentation/v-blank wait).
        if (query_pool_ != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                query_pool_, this_slot + 1);
        }

        ++frame_index_;

        window_->frameReady();
        // Always request the next frame. When converged (accum_blend
        // == 0) the dispatch still runs but the shader keeps the
        // existing pixels (mix(prev,new,0) == prev), so the image is
        // visually frozen while the loop stays alive to notice the next
        // camera/scene change. We intentionally do NOT gate this on
        // convergence: the camera-orbit handlers rely on the continuous
        // loop to pick up pose changes (they don't call requestUpdate
        // themselves), so suspending it could wedge the view.
        window_->requestUpdate();
    }

private:
    // ── pipeline build ───────────────────────────────────────────────────────

    bool rebuild_pipeline_from_scene() {
        build_status_.clear();

        // Emit GLSL from the live scene. If the scene has no objects
        // (initial empty state on first launch), fall back to the
        // precompiled `sphere_trace.spv` so the user at least sees the
        // reference scene rather than a blank window.
        std::vector<std::uint32_t> spv;
        if (scene_ && !scene_->objects().empty()) {
            // Pass the active TracerConfig so the real-time path
            // honours the same Render-tab settings as the offscreen
            // path. The tracer_cfg_ member is set by MainWindow via
            // VulkanViewport::set_tracer_config when sliders change.
            auto emit_or = gpu::GlslEmitter::emit(*scene_, tracer_cfg_);
            if (!emit_or) {
                build_status_ = "emit failed: " + emit_or.error();
                return false;
            }
            auto spv_or = gpu::compile_glsl_to_spv_managed(emit_or->source);
            if (!spv_or) {
                build_status_ = "glsl→spv: " + spv_or.error().substr(0, 120);
                return false;
            }
            spv = read_spv_file(spv_or->path());
            // Cache the raw payloads (move out of the emit_or result —
            // we own them until the next rebuild) so the descriptor-
            // writing path below can upload them as SSBOs. The fields
            // are kept as members rather than locals because some
            // future refactor (e.g. background pipeline rebuild) would
            // benefit from being able to upload data without re-emit.
            needs_mesh_buffer_  = !emit_or->mesh_voxels.empty();
            needs_tex_buffer_   = !emit_or->texture_pixels.empty();
            mesh_voxels_cache_  = std::move(emit_or->mesh_voxels);
            tex_pixels_cache_   = std::move(emit_or->texture_pixels);
            std::fprintf(stderr,
                "VulkanViewport: emitted GLSL OK — %d objs, "
                "mesh-voxels=%zu, texture-pixels=%zu, src=%zu chars\n",
                emit_or->object_count,
                mesh_voxels_cache_.size(),
                tex_pixels_cache_.size(),
                emit_or->source.size());
        } else {
            // Try a few well-known paths for the prebuilt SPV.
            const char* candidates[] = {
                "shaders/sphere_trace.spv",
                "build/shaders/sphere_trace.spv",
                "../shaders/sphere_trace.spv",
            };
            for (const char* p : candidates) {
                spv = read_spv_file(p);
                if (!spv.empty()) break;
            }
            if (spv.empty()) {
                build_status_ = "no scene + sphere_trace.spv not found";
                return false;
            }
            needs_mesh_buffer_ = false;
            needs_tex_buffer_  = false;
            mesh_voxels_cache_.clear();
            tex_pixels_cache_.clear();
        }

        // ── shader module ───────────────────────────────────────────────────
        VkShaderModuleCreateInfo smi{};
        smi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = spv.size() * sizeof(std::uint32_t);
        smi.pCode    = spv.data();
        if (vkCreateShaderModule(device_, &smi, nullptr, &shader_module_)
            != VK_SUCCESS)
        {
            build_status_ = "vkCreateShaderModule failed";
            return false;
        }

        // ── descriptor set layout ───────────────────────────────────────────
        // Binding 0 — output storage image (always present).
        // Binding 1 — mesh voxel SSBO (only when scene has MeshSDF nodes).
        // Binding 2 — texture pixel SSBO (only when scene has Texture
        //             pattern materials). These mirror the binding
        //             numbers expected by GlslEmitter's emitted shader,
        //             and match the offscreen VulkanCtx layout exactly so
        //             both code paths can use the same emitted SPV.
        //
        // We deliberately leave the cached emitter result alive in
        // mesh_voxels_cache_ / tex_pixels_cache_ so a pipeline rebuild
        // triggered by a camera-irrelevant material edit doesn't need to
        // re-emit + re-pack potentially-megabyte payloads. They're
        // refreshed only when the active scene actually changes.
        std::vector<VkDescriptorSetLayoutBinding> bs;
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bs.push_back(b);
        if (needs_mesh_buffer_) {
            b.binding        = 1;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bs.push_back(b);
        }
        if (needs_tex_buffer_) {
            b.binding        = 2;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bs.push_back(b);
        }

        VkDescriptorSetLayoutCreateInfo dsli{};
        dsli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsli.bindingCount = static_cast<std::uint32_t>(bs.size());
        dsli.pBindings    = bs.data();
        if (vkCreateDescriptorSetLayout(device_, &dsli, nullptr, &set_layout_)
            != VK_SUCCESS)
        {
            destroy_pipeline();
            build_status_ = "vkCreateDescriptorSetLayout failed";
            return false;
        }

        // ── pipeline layout (with push constants) ───────────────────────────
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(gpu::ShaderPush);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount         = 1;
        pli.pSetLayouts            = &set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(device_, &pli, nullptr, &pipe_layout_)
            != VK_SUCCESS)
        {
            destroy_pipeline();
            build_status_ = "vkCreatePipelineLayout failed";
            return false;
        }

        // ── compute pipeline ────────────────────────────────────────────────
        VkPipelineShaderStageCreateInfo ssi{};
        ssi.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssi.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ssi.module = shader_module_;
        ssi.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = ssi;
        cpci.layout = pipe_layout_;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci,
                                     nullptr, &pipeline_) != VK_SUCCESS)
        {
            destroy_pipeline();
            build_status_ = "vkCreateComputePipelines failed";
            return false;
        }

        // ── descriptor pool + set ───────────────────────────────────────────
        // Pool sized for whatever combination of bindings the scene
        // requires. The +1/+1 contributions go into a single
        // STORAGE_BUFFER pool slot — Vulkan pools group by descriptor
        // type, not by binding number.
        std::vector<VkDescriptorPoolSize> ps;
        ps.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1});
        std::uint32_t sb_count = (needs_mesh_buffer_ ? 1u : 0u)
                               + (needs_tex_buffer_  ? 1u : 0u);
        if (sb_count > 0)
            ps.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sb_count});

        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets       = 1;
        dpi.poolSizeCount = static_cast<std::uint32_t>(ps.size());
        dpi.pPoolSizes    = ps.data();
        if (vkCreateDescriptorPool(device_, &dpi, nullptr, &desc_pool_)
            != VK_SUCCESS)
        {
            destroy_pipeline();
            build_status_ = "vkCreateDescriptorPool failed";
            return false;
        }
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = desc_pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &set_layout_;
        if (vkAllocateDescriptorSets(device_, &dsai, &desc_set_)
            != VK_SUCCESS)
        {
            destroy_pipeline();
            build_status_ = "vkAllocateDescriptorSets failed";
            return false;
        }

        // ── upload mesh / texture data as host-visible SSBOs ────────────────
        // These mirror the offscreen VulkanCtx upload flow exactly. We
        // use HOST_VISIBLE | HOST_COHERENT memory so the upload is a
        // single map + memcpy + unmap — no staging buffer, no transfer
        // queue submission. For the working-set sizes we care about
        // (typical mesh voxel grid is ~64³ = 256 KB, textures are
        // 64×64 to 512×512 = 16 KB to 1 MB) the bandwidth difference
        // vs. DEVICE_LOCAL is negligible, and the simplicity is worth
        // far more than a few percent shader-side fetch speed.
        if (needs_mesh_buffer_) {
            if (!create_storage_buffer(
                    mesh_voxels_cache_.data(),
                    mesh_voxels_cache_.size() * sizeof(float),
                    /*binding=*/1,
                    mesh_buffer_, mesh_buffer_mem_))
            {
                destroy_pipeline();
                build_status_ = "mesh storage buffer creation failed";
                return false;
            }
        }
        if (needs_tex_buffer_) {
            if (!create_storage_buffer(
                    tex_pixels_cache_.data(),
                    tex_pixels_cache_.size(),     // bytes
                    /*binding=*/2,
                    tex_buffer_, tex_buffer_mem_))
            {
                destroy_pipeline();
                build_status_ = "texture storage buffer creation failed";
                return false;
            }
        }

        scene_hash_ = compute_scene_hash();
        return true;
    }

    void destroy_pipeline() {
        if (device_ == VK_NULL_HANDLE) return;
        if (desc_pool_     != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, desc_pool_, nullptr);     desc_pool_     = VK_NULL_HANDLE; desc_set_ = VK_NULL_HANDLE; }
        if (pipeline_      != VK_NULL_HANDLE) { vkDestroyPipeline(device_, pipeline_, nullptr);            pipeline_      = VK_NULL_HANDLE; }
        if (pipe_layout_   != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, pipe_layout_, nullptr);   pipe_layout_   = VK_NULL_HANDLE; }
        if (set_layout_    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr); set_layout_  = VK_NULL_HANDLE; }
        if (shader_module_ != VK_NULL_HANDLE) { vkDestroyShaderModule(device_, shader_module_, nullptr);   shader_module_ = VK_NULL_HANDLE; }
        // Mesh / texture storage buffers are owned by the pipeline
        // lifetime (they're written into the descriptor set, so they
        // can't outlive it). Free them here alongside the pool/pipeline
        // teardown. Memory frees follow the same paired order as the
        // VulkanCtx offscreen path uses.
        if (mesh_buffer_     != VK_NULL_HANDLE) { vkDestroyBuffer(device_, mesh_buffer_, nullptr); mesh_buffer_ = VK_NULL_HANDLE; }
        if (mesh_buffer_mem_ != VK_NULL_HANDLE) { vkFreeMemory(device_, mesh_buffer_mem_, nullptr); mesh_buffer_mem_ = VK_NULL_HANDLE; }
        if (tex_buffer_      != VK_NULL_HANDLE) { vkDestroyBuffer(device_, tex_buffer_, nullptr); tex_buffer_ = VK_NULL_HANDLE; }
        if (tex_buffer_mem_  != VK_NULL_HANDLE) { vkFreeMemory(device_, tex_buffer_mem_, nullptr); tex_buffer_mem_ = VK_NULL_HANDLE; }
    }

    // ── storage buffer helper ────────────────────────────────────────────────
    //
    // Allocates a HOST_VISIBLE | HOST_COHERENT VkBuffer of the given
    // size, memcpys `src` into it, and writes the descriptor set at
    // `binding` to point to it. Used for both the mesh-voxel and
    // texture-pixel SSBOs — the only thing that differs between them
    // is the binding number (and the source data). Output handles are
    // returned via out-params so the caller owns lifetime.
    //
    // On failure prints a one-line stderr diagnostic and returns
    // false; the caller is expected to call destroy_pipeline() to roll
    // back any partial state. Memory leaks are not a concern here
    // because every failure path goes through destroy_pipeline.
    bool create_storage_buffer(const void*        src,
                               VkDeviceSize       size_bytes,
                               std::uint32_t      binding,
                               VkBuffer&          out_buffer,
                               VkDeviceMemory&    out_memory)
    {
        if (size_bytes == 0) {
            std::fprintf(stderr,
                "VulkanViewport: create_storage_buffer called with 0 bytes "
                "(binding %u) — refusing\n", binding);
            return false;
        }
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size_bytes;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &bci, nullptr, &out_buffer) != VK_SUCCESS) {
            std::fprintf(stderr,
                "VulkanViewport: vkCreateBuffer failed (binding %u, "
                "size %llu)\n", binding,
                static_cast<unsigned long long>(size_bytes));
            return false;
        }
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(device_, out_buffer, &mr);
        std::uint32_t mtype = pick_memory_type(phys_, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mtype == UINT32_MAX) {
            std::fprintf(stderr,
                "VulkanViewport: no HOST_VISIBLE|HOST_COHERENT memory "
                "type available for binding %u\n", binding);
            return false;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = mtype;
        if (vkAllocateMemory(device_, &mai, nullptr, &out_memory) != VK_SUCCESS) {
            std::fprintf(stderr,
                "VulkanViewport: vkAllocateMemory failed (binding %u, "
                "size %llu)\n", binding,
                static_cast<unsigned long long>(mr.size));
            return false;
        }
        if (vkBindBufferMemory(device_, out_buffer, out_memory, 0)
            != VK_SUCCESS)
        {
            std::fprintf(stderr,
                "VulkanViewport: vkBindBufferMemory failed (binding %u)\n",
                binding);
            return false;
        }

        // One-shot upload. Map, memcpy, unmap — HOST_COHERENT means we
        // don't need a vkFlushMappedMemoryRanges roundtrip.
        void* mapped = nullptr;
        if (vkMapMemory(device_, out_memory, 0, size_bytes, 0, &mapped)
            != VK_SUCCESS)
        {
            std::fprintf(stderr,
                "VulkanViewport: vkMapMemory failed (binding %u)\n", binding);
            return false;
        }
        std::memcpy(mapped, src, size_bytes);
        vkUnmapMemory(device_, out_memory);

        // Write descriptor for the freshly-uploaded buffer.
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = out_buffer;
        dbi.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = desc_set_;
        w.dstBinding      = binding;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

        std::fprintf(stderr,
            "VulkanViewport: uploaded SSBO at binding %u (%llu bytes)\n",
            binding, static_cast<unsigned long long>(size_bytes));
        return true;
    }

    // ── storage image ────────────────────────────────────────────────────────

    void create_storage_image(int w, int h) {
        destroy_storage_image();
        if (device_ == VK_NULL_HANDLE) return;

        // A freshly created storage image holds no valid previous frame,
        // so the accumulation must restart — otherwise the first frame
        // after a resize / SSAA change would blend against garbage.
        storage_has_content_ = false;
        accum_count_         = 0;
        accum_key_           = 0;

        storage_w_ = w;
        storage_h_ = h;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent        = {static_cast<std::uint32_t>(w),
                             static_cast<std::uint32_t>(h), 1u};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device_, &ici, nullptr, &storage_image_)
            != VK_SUCCESS)
        {
            std::fprintf(stderr, "VulkanViewport: vkCreateImage failed\n");
            return;
        }

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(device_, storage_image_, &mr);
        std::uint32_t mtype = pick_memory_type(phys_, mr.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mtype == UINT32_MAX) {
            std::fprintf(stderr,
                "VulkanViewport: no DEVICE_LOCAL memory type for storage image\n");
            destroy_storage_image();
            return;
        }
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = mtype;
        if (vkAllocateMemory(device_, &mai, nullptr, &storage_mem_)
            != VK_SUCCESS)
        {
            std::fprintf(stderr, "VulkanViewport: vkAllocateMemory failed\n");
            destroy_storage_image();
            return;
        }
        vkBindImageMemory(device_, storage_image_, storage_mem_, 0);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = storage_image_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device_, &vci, nullptr, &storage_view_)
            != VK_SUCCESS)
        {
            std::fprintf(stderr, "VulkanViewport: vkCreateImageView failed\n");
            destroy_storage_image();
            return;
        }

        if (desc_set_ != VK_NULL_HANDLE)
            write_descriptor_image();
    }

    void destroy_storage_image() {
        if (device_ == VK_NULL_HANDLE) return;
        if (storage_view_  != VK_NULL_HANDLE) { vkDestroyImageView(device_, storage_view_, nullptr); storage_view_  = VK_NULL_HANDLE; }
        if (storage_image_ != VK_NULL_HANDLE) { vkDestroyImage(device_, storage_image_, nullptr);    storage_image_ = VK_NULL_HANDLE; }
        if (storage_mem_   != VK_NULL_HANDLE) { vkFreeMemory(device_, storage_mem_, nullptr);        storage_mem_   = VK_NULL_HANDLE; }
        storage_w_ = storage_h_ = 0;
    }

    void write_descriptor_image() {
        VkDescriptorImageInfo dii{};
        dii.imageView   = storage_view_;
        dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet wds{};
        wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet          = desc_set_;
        wds.dstBinding      = 0;
        wds.descriptorCount = 1;
        wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wds.pImageInfo      = &dii;
        vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    // Hash recipe that detects any change requiring GLSL re-emit. We
    // include not just `structural_hash` (geometry shape) but the
    // material fields too, because GlslEmitter bakes albedo /
    // pattern / roughness / metallic directly into the emitted source
    // — a `set_material()` call that only swaps the colour still
    // demands a full pipeline rebuild on the GPU side.
    //
    // The recipe is deliberately permissive: bit-cast every float to
    // uint32 and feed it through the same FNV-ish mix as the
    // structural hash. Cheap, and false-positive rebuilds (e.g. when
    // material values float-equal each other) are harmless apart from
    // the rebuild cost itself.
    std::size_t compute_scene_hash() const {
        if (!scene_) return 0;
        auto mix = [](std::size_t& h, std::size_t v) {
            h ^= v + 0x9E3779B9 + (h << 6) + (h >> 2);
        };
        auto hf = [](float f) -> std::size_t {
            std::uint32_t u; std::memcpy(&u, &f, sizeof u);
            return static_cast<std::size_t>(u);
        };
        std::size_t h = 0;
        for (const auto& [id, obj] : scene_->objects()) {
            if (!obj.visible) continue;
            mix(h, obj.geometry->structural_hash());
            const auto& m = obj.material;
            mix(h, hf(m.albedo[0]));  mix(h, hf(m.albedo[1]));  mix(h, hf(m.albedo[2]));
            mix(h, hf(m.albedo2[0])); mix(h, hf(m.albedo2[1])); mix(h, hf(m.albedo2[2]));
            mix(h, hf(m.pattern_scale));
            mix(h, hf(m.roughness));  mix(h, hf(m.metallic));
            mix(h, static_cast<std::size_t>(m.pattern));
        }
        // Lights are baked too (their world-space positions affect the
        // shadow term inlined into the emitted shader). Hashing
        // intensity and rgb together with position covers the typical
        // edit cases from the Lights tab.
        for (const auto& L : scene_->lights()) {
            mix(h, hf(L.pos[0])); mix(h, hf(L.pos[1])); mix(h, hf(L.pos[2]));
            mix(h, hf(L.color[0])); mix(h, hf(L.color[1])); mix(h, hf(L.color[2]));
            mix(h, hf(L.intensity));
        }
        // Tracer config — the GLSL emitter bakes shading model, shadow
        // toggle, AO toggle, shadow softness and AO strength into the
        // emitted source. Any change in these requires a pipeline
        // rebuild, so they must affect the hash. We pack the enums as
        // size_t and float members through `hf` for IEEE bit-equality.
        mix(h, static_cast<std::size_t>(tracer_cfg_.shading_model));
        mix(h, static_cast<std::size_t>(tracer_cfg_.enable_shadows));
        mix(h, static_cast<std::size_t>(tracer_cfg_.enable_ao));
        mix(h, hf(tracer_cfg_.shadow_softness));
        mix(h, hf(tracer_cfg_.ao_strength));
        mix(h, hf(tracer_cfg_.ao_step));
        mix(h, static_cast<std::size_t>(tracer_cfg_.ao_samples));
        // v4.1.0: tracer iteration limits / thresholds.
        mix(h, static_cast<std::size_t>(tracer_cfg_.max_steps));
        mix(h, hf(tracer_cfg_.max_dist));
        mix(h, hf(tracer_cfg_.epsilon));
        mix(h, static_cast<std::size_t>(tracer_cfg_.shadow_steps));
        for (float c : tracer_cfg_.sky_top)     mix(h, hf(c));
        for (float c : tracer_cfg_.sky_horizon) mix(h, hf(c));
        mix(h, static_cast<std::size_t>(tracer_cfg_.max_bounces));
        mix(h, static_cast<std::size_t>(tracer_cfg_.shadow_samples));
        mix(h, hf(tracer_cfg_.shadow_light_radius));
        return h;
    }

    gpu::ShaderPush build_push() {
        if (scene_ && !scene_->objects().empty())
            return gpu::build_push_from_scene(*scene_,
                window_->swapChainImageSize().width(),
                window_->swapChainImageSize().height());
        // No scene: use a reasonable default camera matching the
        // fallback sphere_trace.comp test scene.
        float cam[3]   = {0, 1.5f, 4};
        float tgt[3]   = {0, 0, 0};
        float light[3] = {5, 7, 5};
        return gpu::build_push_simple(cam, tgt, light,
            window_->swapChainImageSize().width(),
            window_->swapChainImageSize().height());
    }

    void do_fallback_clear(VkCommandBuffer cb) {
        // Clear the swapchain image to a recognisable colour so the
        // user immediately sees that the realtime widget is alive even
        // when the compute pipeline failed to build.
        VkImage sc_image = window_->swapChainImage(window_->currentSwapChainImageIndex());
        image_barrier(cb, sc_image,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      0, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue clear{};
        clear.float32[0] = 0.05f; clear.float32[1] = 0.10f;
        clear.float32[2] = 0.12f; clear.float32[3] = 1.0f;
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = 1;
        range.layerCount = 1;
        vkCmdClearColorImage(cb, sc_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clear, 1, &range);
        image_barrier(cb, sc_image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    // ── members ──────────────────────────────────────────────────────────────
    QVulkanWindow*    window_       = nullptr;
    SceneGraph*       scene_        = nullptr;
    VkDevice          device_       = VK_NULL_HANDLE;
    VkPhysicalDevice  phys_         = VK_NULL_HANDLE;
    std::uint32_t     graphics_qfam_ = 0;

    // Pipeline objects (device-lifetime).
    VkShaderModule        shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_    = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout_   = VK_NULL_HANDLE;
    VkPipeline            pipeline_      = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool_     = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set_      = VK_NULL_HANDLE;

    // Storage image (swapchain-resolution-lifetime).
    VkImage        storage_image_ = VK_NULL_HANDLE;
    VkDeviceMemory storage_mem_   = VK_NULL_HANDLE;
    VkImageView    storage_view_  = VK_NULL_HANDLE;
    int            storage_w_ = 0, storage_h_ = 0;

    // Mesh-voxel SSBO (pipeline-lifetime). One contiguous buffer
    // packed by GlslEmitter; layout matches what the emitted shader
    // expects at binding 1.
    VkBuffer       mesh_buffer_     = VK_NULL_HANDLE;
    VkDeviceMemory mesh_buffer_mem_ = VK_NULL_HANDLE;

    // Texture-pixel SSBO (pipeline-lifetime). Packed RGBA8 from
    // emit_or->texture_pixels, bound at binding 2.
    VkBuffer       tex_buffer_     = VK_NULL_HANDLE;
    VkDeviceMemory tex_buffer_mem_ = VK_NULL_HANDLE;

    // Owned copies of the emitted payloads, kept across pipeline
    // rebuilds. Two reasons to cache rather than re-emit every time:
    // (a) the data is identical for any rebuild that didn't change
    // mesh geometry or textures, and (b) we still hold the data after
    // emit_or has been destroyed (move semantics; see emit block).
    std::vector<float>        mesh_voxels_cache_;
    std::vector<std::uint8_t> tex_pixels_cache_;

    // Cached scene hash so we rebuild only on structural change.
    std::size_t  scene_hash_     = ~std::size_t(0);
    bool         needs_mesh_buffer_ = false;
    bool         needs_tex_buffer_  = false;
    std::string  build_status_;

    // Active tracer configuration baked into the emitted GLSL on the
    // next rebuild. Defaults match TracerConfig::default — CookTorrance
    // PBR, shadows on, AO on — so render quality is identical to the
    // CPU JIT path with no MainWindow involvement. Pushed in by
    // VulkanViewport::set_tracer_config when the user moves a slider
    // in the Render tab; included in the scene hash so a config change
    // forces a pipeline rebuild on the next frame.
public:
    void set_tracer_config(const TracerConfig& cfg) {
        tracer_cfg_ = cfg;
        // Don't rebuild immediately — let the per-frame scene-hash
        // check pick the change up. That keeps all rebuild bookkeeping
        // in one place and avoids races with frames already in flight.
    }
    const TracerConfig& tracer_config() const { return tracer_cfg_; }
    int  ssaa_factor() const { return ssaa_factor_; }

    // Most recent successfully-read GPU-side render time, in
    // nanoseconds. 0 if no timestamp has been resolved yet (first
    // few frames or pool absent on cheap devices). Used by the
    // status bar via VulkanViewport.
    double last_gpu_ns() const { return last_gpu_ns_; }

    // Pool layout: POOL_FRAMES rings of 2 slots each (start + end
    // timestamps). Total TIMESTAMP_POOL_SIZE slots. We index by
    // frame_index_ % POOL_FRAMES so even/odd frame slots interleave
    // and we never read a slot the GPU is still writing.
    static constexpr std::uint32_t POOL_FRAMES = 3;
    static constexpr std::uint32_t TIMESTAMP_POOL_SIZE = POOL_FRAMES * 2;
private:
    TracerConfig tracer_cfg_;
    int          ssaa_factor_ = 1;

    VkQueryPool  query_pool_   = VK_NULL_HANDLE;
    float        ns_per_tick_  = 1.0f;
    std::uint64_t frame_index_ = 0;
    double       last_gpu_ns_  = 0.0;

    // ── Temporal denoise accumulation state ──────────────────────────────────
    // accum_count_ counts how many consecutive frames have been blended
    // into the running average since the last reset. It resets to 0
    // whenever the camera or scene changes (detected by comparing a
    // hash of the relevant push-constant fields against accum_key_).
    // The per-frame blend factor handed to the shader is 1/(count+1),
    // capped so we never average more than tracer_cfg_.accum_frames.
    // storage_has_content_ tracks whether the storage image currently
    // holds a valid previous frame (false right after a resize/rebuild),
    // so the first post-reset frame uses an UNDEFINED→GENERAL barrier
    // (discard) and later frames preserve content.
    std::uint64_t accum_key_   = 0;
    int           accum_count_ = 0;
    bool          storage_has_content_ = false;
    // Set true once accum_count_ reaches the target — the image has
    // converged and the render loop idles (stops requesting repaints)
    // until the view changes. Reset on any camera/scene change.
    bool          accum_converged_ = false;
};

// QVulkanWindow subclass — gives us a hook to create the renderer and
// thread the SceneGraph* through to it. We also set the
// PersistentResources flag so VkDevice stays alive across visibility
// events (minimize, ALT-tab) — without this, releaseResources is
// called on minimize and our compute pipeline becomes invalid.
//
// This flag has also been reported (Qt forum thread on compute-based
// ray tracer renderers) to fix cases where startNextFrame stops being
// called after the first frame when the renderer dispatches compute +
// blit rather than going through defaultRenderPass — which is exactly
// our case.
//
// We also force the swapchain to UNORM (not sRGB) via
// setPreferredColorFormats. The compute shader (GlslEmitter output)
// emits already-display-encoded colour values, matching the CPU JIT
// path which applies `sqrt(color)` as the final step. If the swapchain
// is sRGB, vkCmdBlitImage applies an extra linear→sRGB encode on top,
// producing dim/desaturated colours (red → brown, yellow → olive — the
// exact symptoms reported on the GTX 1050 Ti). The UNORM swapchain
// path avoids that double encoding and renders identically to the
// offscreen CPU/GPU paths.
//
// Note: setPreferredColorFormats and setFlags MUST be called before
// the window is made visible (Qt docs warn the call has no effect
// later). Doing it in the constructor satisfies that requirement —
// QWidget::createWindowContainer doesn't show the window immediately.
//
// Mouse handlers (mousePress / mouseMove / wheel) translate drag and
// scroll into orbital camera updates on `scene_->camera()`, mirroring
// the offscreen Viewport's behaviour. The camera state
// (yaw/pitch/distance) is local — initialised from the current
// scene camera position the first time a drag begins.
//
// NOTE: this class is declared in gui/frep_vulkan_window.hpp so the
// public VulkanViewport class can hold a weak pointer to it without
// pulling Vulkan headers into the public API, AND so AutoMoc finds
// the Q_OBJECT macro (it scans headers, not .cpp files).

// ── FRepVulkanWindow ──────────────────────────────────────────────────────────
// Out-of-line definitions for the QVulkanWindow subclass declared in
// gui/frep_vulkan_window.hpp. The renderer (ComputeBlitRenderer) is
// also in this namespace (no longer hidden in anonymous) so the
// header's `ComputeBlitRenderer*` member declaration resolves.

FRepVulkanWindow::FRepVulkanWindow(SceneGraph* scene)
    : scene_(scene)
    , picker_(std::make_unique<ScenePicker>())
{
    setFlags(QVulkanWindow::PersistentResources);
    setPreferredColorFormats({
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
    });
    // Sample the renderer's GPU-time counter at ~10 Hz and emit a
    // signal MainWindow can wire up to its status bar. 100 ms is
    // enough to feel responsive but spares the status bar from being
    // re-rendered on every animation frame (which would itself cost
    // CPU time and skew the reading).
    sample_timer_.setInterval(100);
    sample_timer_.setSingleShot(false);
    QObject::connect(&sample_timer_, &QTimer::timeout,
                     this, &FRepVulkanWindow::sample_timing);
    sample_timer_.start();
}

FRepVulkanWindow::~FRepVulkanWindow() = default;

QVulkanWindowRenderer* FRepVulkanWindow::createRenderer() {
    // The renderer is constructed late (Qt creates it the first time
    // the window becomes visible). Push whatever tracer config and
    // SSAA factor we've received so far — typically the defaults,
    // but MainWindow may have pushed overrides during ctor wiring
    // before any frame was rendered.
    active_renderer_ = new ComputeBlitRenderer(this, scene_);
    active_renderer_->set_tracer_config(pending_cfg_);
    active_renderer_->set_ssaa(pending_ssaa_);
    return active_renderer_;
}

void FRepVulkanWindow::set_tracer_config(const TracerConfig& cfg) {
    pending_cfg_ = cfg;
    if (active_renderer_) active_renderer_->set_tracer_config(cfg);
}

void FRepVulkanWindow::set_ssaa(int n) {
    n = std::clamp(n, 1, 4);
    if (n == 3) n = 4;   // bilinear blit needs even factors (see renderer)
    pending_ssaa_ = n;
    if (active_renderer_) active_renderer_->set_ssaa(pending_ssaa_);
}

void FRepVulkanWindow::sample_timing() {
    if (!active_renderer_) return;
    double ns = active_renderer_->last_gpu_ns();
    if (ns <= 0.0) return;  // first few frames before any read-back
    int ms = static_cast<int>(ns / 1.0e6 + 0.5);
    Q_EMIT render_time_sampled(ms);
}

void FRepVulkanWindow::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        if (!orbit_init_) {
            const auto& p = scene_->camera().position;
            cam_dist_  = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
            if (cam_dist_ < 0.1f) cam_dist_ = cam_cfg_.default_distance;
            cam_yaw_   = std::atan2(p[0], p[2]);
            cam_pitch_ = std::asin(std::clamp(p[1] / cam_dist_, -1.0f, 1.0f));
            orbit_init_ = true;
        }
        dragging_        = true;
        last_pos_        = e->pos();
        press_pos_       = e->pos();
        press_modifiers_ = e->modifiers();
    }
}

void FRepVulkanWindow::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return;
    dragging_ = false;
    // Click-vs-drag heuristic: if the mouse barely moved between press
    // and release, treat it as a click and attempt a pick. A drag of
    // more than 4 pixels is taken as camera-orbit instead. The
    // threshold is generous to accommodate slight hand tremor while
    // still feeling responsive — same value as the offscreen path.
    if ((e->pos() - press_pos_).manhattanLength() <= 4) {
        bool additive = press_modifiers_ & Qt::ControlModifier;
        try_pick(e->pos(), additive);
    }
}

void FRepVulkanWindow::try_pick(const QPoint& pos, bool additive) {
    if (!scene_) return;
    int w = width();
    int h = height();
    if (w < 8 || h < 8) return;

    // Lazy rebuild — the picker re-JITs only when the scene's
    // structural hash drifts. Parameter edits (slider drags) don't
    // invalidate it.
    if (!picker_->valid_for(*scene_)) {
        auto rb = picker_->rebuild(*scene_);
        if (!rb) return;  // emit a silent miss rather than asserting
    }

    auto hit = picker_->pick_pixel(scene_->camera(),
                                   pos.x(), pos.y(), w, h);
    QString id = hit ? QString::fromStdString(*hit) : QString();
    // Always emit, even on miss: a plain click on empty background
    // clears the inspector selection (modulo Ctrl, which keeps it).
    Q_EMIT object_picked(id, additive);
}

void FRepVulkanWindow::mouseMoveEvent(QMouseEvent* e) {
    if (!dragging_) return;
    auto delta = e->pos() - last_pos_;
    last_pos_  = e->pos();
    cam_yaw_   -= delta.x() * cam_cfg_.mouse_sensitivity;
    cam_pitch_ -= delta.y() * cam_cfg_.mouse_sensitivity;
    cam_pitch_  = std::clamp(cam_pitch_, -cam_cfg_.max_pitch, cam_cfg_.max_pitch);
    apply_camera();
}

void FRepVulkanWindow::wheelEvent(QWheelEvent* e) {
    if (!orbit_init_) {
        const auto& p = scene_->camera().position;
        cam_dist_  = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        if (cam_dist_ < 0.1f) cam_dist_ = cam_cfg_.default_distance;
        cam_yaw_   = std::atan2(p[0], p[2]);
        cam_pitch_ = std::asin(std::clamp(p[1] / cam_dist_, -1.0f, 1.0f));
        orbit_init_ = true;
    }
    cam_dist_ *= (e->angleDelta().y() > 0)
        ? (1.0f / cam_cfg_.zoom_step) : cam_cfg_.zoom_step;
    cam_dist_  = std::clamp(cam_dist_, cam_cfg_.min_distance, cam_cfg_.max_distance);
    apply_camera();
}

void FRepVulkanWindow::apply_camera() {
    float r  = cam_dist_;
    float sx = r * std::cos(cam_pitch_) * std::sin(cam_yaw_);
    float sy = r * std::sin(cam_pitch_);
    float sz = r * std::cos(cam_pitch_) * std::cos(cam_yaw_);
    scene_->camera().position = {sx, sy, sz};
    scene_->camera().target   = {0.0f, 0.0f, 0.0f};
    scene_->camera().up       = {0.0f, 1.0f, 0.0f};
    // No requestUpdate() needed — startNextFrame is being called
    // continuously through the renderer's own requestUpdate loop.
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
bool vulkan_viewport_available() {
    const char* env = std::getenv("FREP_REALTIME_VIEWPORT");
    if (env) {
        if (std::strcmp(env, "0") == 0)     return false;
        if (std::strcmp(env, "force") == 0) return true;
    }
    return probe_real_vulkan_device();
}

// Helper used by both create() and create_iv(): does the actual
// QVulkanInstance + FRepVulkanWindow + container construction.
// Returns the container widget and (via out-param) the underlying
// FRepVulkanWindow* for callers that need to forward later config
// updates (e.g. tracer settings). Returns nullptr widget on failure
// — in which case g_last_status is updated to describe the reason
// and the caller can build a placeholder widget instead.
static QWidget* build_realtime_container(SceneGraph* scene, QWidget* parent,
                                         FRepVulkanWindow** out_window = nullptr)
{
    if (out_window) *out_window = nullptr;
    if (!vulkan_viewport_available()) {
        g_last_status = "real-time viewport not available — using offscreen path";
        return nullptr;
    }
    auto* vk_inst = new QVulkanInstance;
    vk_inst->setApiVersion(QVersionNumber(1, 3));
    if (!vk_inst->create()) {
        g_last_status = QString("QVulkanInstance::create failed: %1")
            .arg(vk_inst->errorCode());
        delete vk_inst;
        return nullptr;
    }
    auto* vk_window = new FRepVulkanWindow(scene);
    vk_window->setVulkanInstance(vk_inst);

    auto* container = QWidget::createWindowContainer(vk_window, parent);
    container->setMinimumSize(640, 480);
    container->setFocusPolicy(Qt::StrongFocus);
    g_last_status = "Real-time GPU viewport active";
    if (out_window) *out_window = vk_window;
    return container;
}

// Legacy QWidget-returning factory — preserved for callers that don't
// want the IViewport interface (none in-tree as of v4.0.2, but keeping
// the entry point avoids breakage for any user code that compiles
// against the public API). On failure it returns a QLabel placeholder
// just like before, so existing callsites don't need to handle nullptr.
QWidget* VulkanViewport::create(SceneGraph* scene, QWidget* parent) {
    if (QWidget* c = build_realtime_container(scene, parent))
        return c;
    auto* placeholder = new QLabel(g_last_status, parent);
    placeholder->setAlignment(Qt::AlignCenter);
    return placeholder;
}

// IViewport-returning factory — the path MainWindow uses. Returns
// nullptr on hardware/driver unavailability so the caller can fall
// back to OffscreenViewportAdapter. The widget() accessor on the
// returned object provides the QWidget MainWindow needs to add to
// its layout.
IViewport* VulkanViewport::create_iv(SceneGraph* scene, QWidget* parent) {
    FRepVulkanWindow* window = nullptr;
    QWidget* container = build_realtime_container(scene, parent, &window);
    if (!container) return nullptr;
    // VulkanViewport itself is just a wrapper around the container —
    // it parents on the same parent so Qt's object tree cleans both up.
    // We also keep a weak pointer to the FRepVulkanWindow so future
    // set_tracer_config() calls can forward into it.
    return new VulkanViewport(container, window, g_last_status);
}

VulkanViewport::VulkanViewport(QWidget* container,
                               FRepVulkanWindow* window,
                               QString status)
    : IViewport(container)   // parent the IViewport on the container so
                             // when the layout removes/destroys the
                             // widget, the IViewport also gets cleaned up
    , container_(container)
    , window_(window)
    , status_(std::move(status))
{
    // Forward the window's ~10Hz GPU-time sample into the IViewport's
    // render_completed signal so MainWindow's status bar sees real
    // numbers on the real-time path. Object picking is also wired
    // through: the window fires `object_picked(id, additive)` on a
    // click that didn't drag, and we forward to IViewport's
    // `object_picked(QString)` — MainWindow checks the modifier
    // separately via QGuiApplication::keyboardModifiers().
    if (window_) {
        connect(window_, &FRepVulkanWindow::render_time_sampled,
                this,    &IViewport::render_completed);
        // Also stash the sampled GPU frame time for the metrics HUD.
        connect(window_, &FRepVulkanWindow::render_time_sampled,
                this,    [this](int ms) { last_gpu_ms_ = ms; });
        connect(window_, &FRepVulkanWindow::object_picked,
                this,    [this](const QString& id, bool /*additive*/) {
                    // additive is consumed inside MainWindow's click
                    // handler (which polls QGuiApplication directly),
                    // so we don't need to plumb it through here.
                    Q_EMIT object_picked(id);
                });
    }
}

QWidget* VulkanViewport::widget() { return container_; }

void VulkanViewport::invalidate() {
    // The real-time renderer renders continuously (60+ FPS in the
    // requestUpdate loop) — there's no notion of a "stale" frame to
    // invalidate. This method exists so MainWindow can call
    // viewport_->invalidate() universally without special-casing the
    // active backend.
    //
    // Future: if we ever add a power-save mode that skips frames when
    // nothing changes, this would be where MainWindow signals "wake
    // up, the scene actually moved".
}

void VulkanViewport::set_tracer_config(const TracerConfig& cfg) {
    // Forward into the QVulkanWindow subclass. It may not yet have
    // built its renderer (Qt builds renderers lazily on first show),
    // in which case the window buffers the config in pending_cfg_
    // and applies it inside createRenderer(). Once the renderer
    // exists, this call propagates directly — the renderer's
    // per-frame scene-hash check then triggers a pipeline rebuild
    // on the next frame to actually re-emit GLSL with the new config.
    if (window_) window_->set_tracer_config(cfg);
    using CM = TracerConfig::CullMethod;
    switch (cfg.cull_method) {
        case CM::Auto:      cull_method_ = "Auto";      break;
        case CM::Lipschitz: cull_method_ = "Lipschitz"; break;
        case CM::Interval:  cull_method_ = "Interval";  break;
        case CM::Off:       cull_method_ = "Off";       break;
    }
}

QString VulkanViewport::metrics_text() const {
    // GPU frame time comes from a timestamp-query pair wrapping the compute
    // dispatch (measured in the renderer, sampled ~10 Hz). Plus the tile-cull
    // method in use.
    QString s = "GPU_GLSL (real-time)";
    if (last_gpu_ms_ > 0) {
        double fps = 1000.0 / last_gpu_ms_;
        s += QString("\n%1 ms  (%2 fps)")
                 .arg(last_gpu_ms_).arg(fps, 0, 'f', 0);
    }
    if (!cull_method_.isEmpty()) s += "\ncull: " + cull_method_;
    return s;
}

void VulkanViewport::set_ssaa(int n) {
    // Forward into the QVulkanWindow subclass. Same lazy-renderer
    // dance as set_tracer_config — the window caches the factor and
    // applies it inside createRenderer() or directly to the live
    // renderer if one exists. The renderer in turn forces a storage-
    // image rebuild at the new (sc_w*N × sc_h*N) dimensions.
    if (window_) window_->set_ssaa(n);
}

void VulkanViewport::set_camera_control_config(const CameraControlConfig& c) {
    if (window_) window_->set_camera_control_config(c);
}

QImage VulkanViewport::capture_image() {
    // True GPU framebuffer grab. QVulkanWindow::grab() blits the last
    // rendered swapchain image into a host-visible buffer (an internal
    // vkCmdCopyImage + readback) and returns it as a QImage — no
    // screen-coordinate capture, so overlapping windows and compositor
    // effects don't taint the result. Requires the PersistentResources
    // flag, which the window sets in its constructor. Returns a null
    // image if no frame has been rendered yet.
    if (!window_) return {};
    QImage img = window_->grab();
    if (img.isNull()) return img;

    // Channel-order fix. Our preferred swapchain format is
    // VK_FORMAT_B8G8R8A8_UNORM (BGRA in memory), but grab() hands the
    // bytes back tagged as an RGBA-order QImage, so the blue and red
    // channels come out swapped — saved files looked BGR. When the live
    // colour format is one of the B-first (BGRA) formats, swap R↔B back
    // into true RGB order. If the driver gave us an R-first format
    // instead, the bytes are already correct and we leave them alone.
    const VkFormat fmt = window_->colorFormat();
    const bool is_bgra =
        (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
    if (is_bgra)
        img = img.rgbSwapped();
    return img;
}

QString VulkanViewport::last_status() { return g_last_status; }

} // namespace frep::gui
