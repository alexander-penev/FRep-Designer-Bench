// core/gpu/rtx_pipeline.cpp — see rtx_pipeline.hpp.
//
// Dense Vulkan RT; the sandbox can't execute it (no device), so it's written
// carefully and validated on llvmpipe / hardware. Flow:
//   1. shader modules from SPIR-V
//   2. descriptor set layout: TLAS (binding 0) + storage image (binding 1)
//   3. RT pipeline: rgen + miss + hit group (intersection + closest-hit)
//   4. shader binding table (3 groups: raygen, miss, hit)
//   5. output image (rgba32f) + descriptor set
//   6. vkCmdTraceRaysKHR, then copy image → host-visible buffer → floats

#include "core/gpu/rtx_pipeline.hpp"
#include "core/gpu/rtx_query.hpp"
#include "core/gpu/rgba_unpack.hpp"
#include "core/gpu/vk_fence_wait.hpp"
#include "core/tracer/pixel_buffers.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstring>
#include <optional>
#include <functional>

namespace frep::gpu {

namespace {

template <class F> F api_cast(void* p) { return reinterpret_cast<F>(p); }

int find_mem(VkPhysicalDevice phys, std::uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

struct Buf { VkBuffer buf = nullptr; VkDeviceMemory mem = nullptr; };

std::expected<Buf, std::string>
mk_buf(const RtxCtx& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
       VkMemoryPropertyFlags props, bool device_addr) {
    Buf b;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage | (device_addr ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device(), &bci, nullptr, &b.buf) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: vkCreateBuffer failed");
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buf, &mr);
    int t = find_mem(ctx.physical_device(), mr.memoryTypeBits, props);
    if (t < 0) return std::unexpected("rtx_pipeline: no memory type");
    VkMemoryAllocateFlagsInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = (std::uint32_t)t;
    if (device_addr) mai.pNext = &fi;
    if (vkAllocateMemory(ctx.device(), &mai, nullptr, &b.mem) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: vkAllocateMemory failed");
    vkBindBufferMemory(ctx.device(), b.buf, b.mem, 0);
    return b;
}

// Readback buffer: prefer HOST_CACHED so the CPU read-back of the frame isn't
// from write-combined memory. NVIDIA maps a HOST_COHERENT-only host type to
// write-combined, which the CPU reads at ~300 MB/s — a 4 MB rgba32f frame then
// costs ~13 ms in the map+memcpy alone, dwarfing the ~0.7 ms trace. A CACHED
// host type reads at cache speed. Falls back to plain COHERENT if the device
// has no cached host type (the cached types NVIDIA exposes are also coherent,
// so no manual invalidate is needed on the fast path).
std::expected<Buf, std::string>
mk_readback_buf(const RtxCtx& ctx, VkDeviceSize size) {
    auto cached = mk_buf(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
    if (cached) return cached;
    return mk_buf(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
}

// Output rgba8 storage image + its readback buffer for one frame. The format
// matches the compute path: the frame is 8 bits per channel on both, so the RT
// path no longer moves four times the bytes for the same picture. When
// `cacheable` and a cache is supplied, these are reused across frames while the
// extent is unchanged (the two ~W*H*16-byte allocations otherwise recur every
// frame); a resize frees the stale pair and rebuilds. `cached==true` means the
// handles are owned by the cache and must NOT be torn down per frame.
struct FrameRT {
    VkImage        image        = VK_NULL_HANDLE;
    VkDeviceMemory image_mem    = VK_NULL_HANDLE;
    VkImageView    view         = VK_NULL_HANDLE;
    VkBuffer       readback     = VK_NULL_HANDLE;
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    void*          mapped       = nullptr;   // persistent map when cached
    bool           cached       = false;
};

std::expected<FrameRT, std::string>
acquire_frame_rt(const RtxCtx& ctx, RtxPipelineCache* cache,
                 int width, int height, bool cacheable, bool float_image) {
    auto dev = ctx.device();
    if (cache && cacheable && cache->frame_image &&
        cache->frame_w == width && cache->frame_h == height &&
        cache->frame_float == float_image) {
        FrameRT f;
        f.image        = (VkImage)cache->frame_image;
        f.image_mem    = (VkDeviceMemory)cache->frame_image_mem;
        f.view         = (VkImageView)cache->frame_image_view;
        f.readback     = (VkBuffer)cache->frame_readback;
        f.readback_mem = (VkDeviceMemory)cache->frame_readback_mem;
        f.mapped       = cache->frame_readback_mapped;
        f.cached       = true;
        return f;
    }
    // Free a stale-size cached frame before rebuilding.
    if (cache && cacheable && cache->frame_image) {
        if (cache->frame_image_view) vkDestroyImageView(dev, (VkImageView)cache->frame_image_view, nullptr);
        vkDestroyImage(dev, (VkImage)cache->frame_image, nullptr);
        vkFreeMemory(dev, (VkDeviceMemory)cache->frame_image_mem, nullptr);
        if (cache->frame_readback_mapped) vkUnmapMemory(dev, (VkDeviceMemory)cache->frame_readback_mem);
        vkDestroyBuffer(dev, (VkBuffer)cache->frame_readback, nullptr);
        vkFreeMemory(dev, (VkDeviceMemory)cache->frame_readback_mem, nullptr);
        cache->frame_image = cache->frame_image_mem = cache->frame_image_view = nullptr;
        cache->frame_readback = cache->frame_readback_mem = cache->frame_readback_mapped = nullptr;
    }

    // rgba8 unless the shaders were emitted for a float frame (the caller
    // asked for floats): then rgba32f, four times the bytes, no quantisation.
    const VkFormat fmt = float_image ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    const VkDeviceSize bpp = float_image ? 16 : 4;
    FrameRT f;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ici, nullptr, &f.image) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: vkCreateImage failed");
    VkMemoryRequirements imr{};
    vkGetImageMemoryRequirements(dev, f.image, &imr);
    int it = find_mem(ctx.physical_device(), imr.memoryTypeBits,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo imai{};
    imai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imai.allocationSize = imr.size; imai.memoryTypeIndex = (std::uint32_t)it;
    if (vkAllocateMemory(dev, &imai, nullptr, &f.image_mem) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: image vkAllocateMemory failed");
    vkBindImageMemory(dev, f.image, f.image_mem, 0);

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = f.image; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = fmt;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &f.view);

    // Sized for the image it receives (it was once sized for rgba32f long
    // after the image had stopped being one).
    auto rb = mk_readback_buf(ctx, (VkDeviceSize)width * height * bpp);
    if (!rb) return std::unexpected(rb.error());
    f.readback = rb->buf; f.readback_mem = rb->mem;

    if (cache && cacheable) {
        // Map once and keep it: the buffer is read on every frame and the
        // memory type is host-coherent, so the per-frame map/unmap only cost.
        vkMapMemory(dev, f.readback_mem, 0, VK_WHOLE_SIZE, 0, &f.mapped);
        cache->frame_image = f.image; cache->frame_image_mem = f.image_mem;
        cache->frame_image_view = f.view; cache->frame_readback = f.readback;
        cache->frame_readback_mem = f.readback_mem;
        cache->frame_readback_mapped = f.mapped;
        cache->frame_w = width; cache->frame_h = height;
        cache->frame_float = float_image;
        f.cached = true;   // stored in cache → don't tear down per frame
    }
    return f;
}

// Hand the mapped frame over in the form the caller asked for: the image's
// own bytes or floats when the request matches it, a conversion otherwise.
void read_frame(const void* p, int width, int height, bool float_image,
                bool want_rgba8, RtRenderResult& out) {
    const std::size_t n = (std::size_t)width * height * 4;   // channels
    auto t_alloc0 = std::chrono::steady_clock::now();
    // Recycled (pixel_buffers.hpp): sized without zeroing, overwritten below.
    if (want_rgba8) out.rgba8 = take_bytes(n);
    else            out.rgba  = take_floats(n);
    out.readback_alloc_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_alloc0).count();
    if (float_image) {
        if (want_rgba8) pack_rgba8(static_cast<const float*>(p), out.rgba8.data(), n);
        else            std::memcpy(out.rgba.data(), p, n * sizeof(float));
    } else {
        if (want_rgba8) std::memcpy(out.rgba8.data(), p, n);
        else unpack_rgba8(static_cast<const std::uint8_t*>(p), out.rgba.data(), n);
    }
}


// Upload `bytes` into a device-local buffer through a temporary staging copy.
//
// The shader binding table is read by the hardware on every trace, to find the
// shader for a hit. Leaving it in host-visible memory means those reads cross
// PCIe (and on NVIDIA come from a write-combined mapping) on every dispatch,
// which is a cost the compute path -- having no binding table at all -- never
// pays. The table only changes when the shader set does, and that is already
// cached, so uploading it once to VRAM costs nothing per frame.
std::expected<Buf, std::string>
mk_device_local_upload(const RtxCtx& ctx, const void* bytes, VkDeviceSize size,
                       VkBufferUsageFlags usage) {
    auto dst = mk_buf(ctx, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
    if (!dst) return dst;
    auto stg = mk_buf(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
    if (!stg) return std::unexpected(stg.error());
    void* p = nullptr;
    vkMapMemory(ctx.device(), stg->mem, 0, size, 0, &p);
    std::memcpy(p, bytes, (std::size_t)size);
    vkUnmapMemory(ctx.device(), stg->mem);

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = ctx.queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(ctx.device(), &cpi, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device(), &cbai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{}; region.size = size;
    vkCmdCopyBuffer(cmd, stg->buf, dst->buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.queue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.queue());
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    vkDestroyBuffer(ctx.device(), stg->buf, nullptr);
    vkFreeMemory(ctx.device(), stg->mem, nullptr);
    return dst;
}

VkDeviceAddress buf_addr(const RtxCtx& ctx, VkBuffer buf) {
    // core in 1.2 — call directly (see note in rtx_accel.cpp)
    VkBufferDeviceAddressInfo i{};
    i.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    i.buffer = buf;
    return vkGetBufferDeviceAddress(ctx.device(), &i);
}

VkShaderModule mk_module(VkDevice dev, const std::vector<std::uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(std::uint32_t);
    ci.pCode = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, nullptr, &m);
    return m;
}

// align up to `a`
inline VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) & ~(a - 1);
}

// FNV-1a hash of the four shader blobs + binding shape. Identifies a shader set
// so a pipeline cache knows when it can reuse its built pipeline + SBT.
inline std::uint64_t shader_set_key(const std::vector<std::uint32_t>& rgen,
                                    const std::vector<std::uint32_t>& rint,
                                    const std::vector<std::uint32_t>& rchit,
                                    const std::vector<std::uint32_t>& rmiss,
                                    unsigned binding_flags) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::vector<std::uint32_t>& v) {
        h ^= v.size(); h *= 1099511628211ull;
        for (std::uint32_t w : v) { h ^= w; h *= 1099511628211ull; }
    };
    mix(rgen); mix(rint); mix(rchit); mix(rmiss);
    h ^= binding_flags;
    h *= 1099511628211ull;
    return h;
}

// GPU timestamps around the trace and the copy, read when profiling
// (FREP_RTX_PROFILE=1). The fence time a frame otherwise reports is the
// submit-to-signal span, which holds the trace, the image-to-buffer copy
// across PCIe and the queue's own latency; only the first of those is the
// ray tracing, and at 512² it is the smaller part. Three queries: before the
// trace, after it, after the copy.
inline bool rtx_profile_on() {
    static const bool on = [] {
        const char* v = std::getenv("FREP_RTX_PROFILE");
        return v && v[0] == '1';
    }();
    return on;
}

}  // namespace

void RtxPipelineCache::release(const RtxCtx& ctx) {
    auto dev = ctx.device();
    if (pipeline)          vkDestroyPipeline(dev, (VkPipeline)pipeline, nullptr);
    if (pipeline_layout)   vkDestroyPipelineLayout(dev, (VkPipelineLayout)pipeline_layout, nullptr);
    if (descriptor_layout) vkDestroyDescriptorSetLayout(dev, (VkDescriptorSetLayout)descriptor_layout, nullptr);
    for (void*& m : shader_modules)
        if (m) { vkDestroyShaderModule(dev, (VkShaderModule)m, nullptr); m = nullptr; }
    for (void* m : group_rint_modules)
        if (m) vkDestroyShaderModule(dev, (VkShaderModule)m, nullptr);
    group_rint_modules.clear();
    for (void* m : group_rchit_modules)
        if (m) vkDestroyShaderModule(dev, (VkShaderModule)m, nullptr);
    group_rchit_modules.clear();
    if (sbt_buffer) vkDestroyBuffer(dev, (VkBuffer)sbt_buffer, nullptr);
    if (sbt_memory) vkFreeMemory(dev, (VkDeviceMemory)sbt_memory, nullptr);
    if (frame_image_view) vkDestroyImageView(dev, (VkImageView)frame_image_view, nullptr);
    if (frame_image)      vkDestroyImage(dev, (VkImage)frame_image, nullptr);
    if (frame_image_mem)  vkFreeMemory(dev, (VkDeviceMemory)frame_image_mem, nullptr);
    if (frame_readback_mapped) vkUnmapMemory(dev, (VkDeviceMemory)frame_readback_mem);
    if (frame_readback)   vkDestroyBuffer(dev, (VkBuffer)frame_readback, nullptr);
    if (frame_readback_mem) vkFreeMemory(dev, (VkDeviceMemory)frame_readback_mem, nullptr);
    if (frame_fence)     vkDestroyFence(dev, (VkFence)frame_fence, nullptr);
    if (frame_cmd_pool)  vkDestroyCommandPool(dev, (VkCommandPool)frame_cmd_pool, nullptr);
    if (frame_desc_pool) vkDestroyDescriptorPool(dev, (VkDescriptorPool)frame_desc_pool, nullptr);
    if (frame_query_pool) vkDestroyQueryPool(dev, (VkQueryPool)frame_query_pool, nullptr);
    frame_query_pool = nullptr;
    frame_fence = frame_cmd_pool = frame_cmd_buf = nullptr;
    frame_desc_pool = frame_desc_set = nullptr; frame_tlas = nullptr;
    frame_image = frame_image_mem = frame_image_view = nullptr;
    frame_readback = frame_readback_mem = frame_readback_mapped = nullptr;
    frame_w = frame_h = 0;
    pipeline = pipeline_layout = descriptor_layout = nullptr;
    sbt_buffer = sbt_memory = nullptr;
    sbt_base = 0; handle_size = 0; key = 0; valid = false;
    key_src.clear(); sbt_base_align = sbt_handle_align = 0;
}

// Shared implementation. `cache` is optional: when non-null, the pipeline +
// SBT + shader modules + layouts are taken from it on a hit (and built into it
// on a miss, surviving teardown); when null, everything is built fresh and
// torn down at the end (the original one-shot behavior).
static std::expected<RtRenderResult, std::string>
rtx_trace_impl(const RtxCtx& ctx, const RtAccel& accel,
               RtxPipelineCache* cache,
               const std::vector<std::uint32_t>& rgen,
               const std::vector<std::uint32_t>& rint,
               const std::vector<std::uint32_t>& rchit,
               const std::vector<std::uint32_t>& rmiss,
               const RtPushConstants& pc, int width, int height,
               const std::vector<std::uint32_t>& texture_pixels,
               const std::vector<float>& mesh_voxels, bool want_rgba8,
               bool float_image) {
    auto dev = ctx.device();
    const bool has_tex  = !texture_pixels.empty();
    const bool has_mesh = !mesh_voxels.empty();
    // More than one box in the single-BLAS build means shell cells
    // (rtx_cells.hpp): the intersection shader was emitted to read its cell
    // from the box list, so that list is bound at binding 4.
    const bool has_cells = accel.aabb_count() > 1 && accel.aabb_buffer();
    const unsigned binding_flags =
        (has_tex ? 2u : 0u) | (has_mesh ? 1u : 0u) | (has_cells ? 4u : 0u);
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t_pipe0 = clk::now();

    // Does the cache hold a pipeline for this exact shader set? If so, skip the
    // build (shader modules + RT pipeline + SBT) and reuse it; only the camera/
    // tile push-constants and the per-frame descriptors change between frames.
    // Build and per-frame work share a clean boundary: the descriptor pool/set
    // (bound to the TLAS + output image) is rebuilt each frame regardless, so
    // the cached objects (pipeline, layouts, modules, SBT) are independent.
    const std::vector<std::pair<const void*, std::size_t>> key_src = {
        {rgen.data(), rgen.size()}, {rint.data(), rint.size()},
        {rchit.data(), rchit.size()}, {rmiss.data(), rmiss.size()},
        {nullptr, (std::size_t)binding_flags}};
    const std::uint64_t want_key =
        (cache && cache->valid && cache->key_src == key_src)
            ? cache->key   // same blobs as the frame before: no rehash
            : shader_set_key(rgen, rint, rchit, rmiss, binding_flags);
    const bool cache_hit = cache && cache->valid && cache->key == want_key;
    if (cache && cache->valid && cache->key != want_key)
        cache->release(ctx);  // different shaders → drop the stale build

    // Objects shared between the build and per-frame sections. On a hit these
    // come from the cache; on a miss they're built below and (if a cache was
    // supplied) stored into it instead of being torn down.
    VkShaderModule m_rgen = VK_NULL_HANDLE, m_rint = VK_NULL_HANDLE,
                   m_rchit = VK_NULL_HANDLE, m_rmiss = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::optional<Buf> sbt;          // only set on a fresh build
    VkBuffer sbt_buf_handle = VK_NULL_HANDLE;
    VkDeviceAddress sbt_base = 0;
    std::uint32_t handle_size = 0;
    VkDeviceSize region = 0;

    auto create_pipelines = api_cast<PFN_vkCreateRayTracingPipelinesKHR>(
                                ctx.api().createRayTracingPipelines);
    auto get_handles = api_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
                           ctx.api().getRayTracingShaderGroupHandles);
    auto cmd_trace = api_cast<PFN_vkCmdTraceRaysKHR>(ctx.api().cmdTraceRays);

    if (cache_hit) {
        // Reuse the cached build; nothing to create here.
        m_rgen  = (VkShaderModule)cache->shader_modules[0];
        m_rint  = (VkShaderModule)cache->shader_modules[1];
        m_rchit = (VkShaderModule)cache->shader_modules[2];
        m_rmiss = (VkShaderModule)cache->shader_modules[3];
        dsl       = (VkDescriptorSetLayout)cache->descriptor_layout;
        playout   = (VkPipelineLayout)cache->pipeline_layout;
        pipeline  = (VkPipeline)cache->pipeline;
        sbt_buf_handle = (VkBuffer)cache->sbt_buffer;
        sbt_base  = cache->sbt_base;
        handle_size = cache->handle_size;
        // region == base-aligned stride, from the alignments stored at build.
        VkDeviceSize stride = align_up(handle_size, cache->sbt_handle_align);
        region = align_up(stride, cache->sbt_base_align);
    } else {

    // ── Shader modules ──────────────────────────────────────────────────────
    m_rgen  = mk_module(dev, rgen);
    m_rint  = mk_module(dev, rint);
    m_rchit = mk_module(dev, rchit);
    m_rmiss = mk_module(dev, rmiss);

    // ── Descriptor set layout: TLAS + storage image ─────────────────────────
    VkDescriptorSetLayoutBinding b0{};  // TLAS
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b0.descriptorCount = 1;
    // Closest-hit reads the TLAS as well on the instanced path: its shadow ray
    // is traced from there rather than marched over the field.
    b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    VkDescriptorSetLayoutBinding b1{};  // output image
    b1.binding = 1; b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b1.descriptorCount = 1; b1.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    VkDescriptorSetLayoutBinding b2{};  // texture storage buffer (closest-hit)
    b2.binding = 2; b2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b2.descriptorCount = 1;
    b2.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                    VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                    VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    VkDescriptorSetLayoutBinding b3{};  // mesh voxel buffer (intersection + chit)
    b3.binding = 3; b3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b3.descriptorCount = 1;
    b3.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                    VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                    VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    VkDescriptorSetLayoutBinding b4{};  // shell-cell box list (intersection)
    b4.binding = 4; b4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b4.descriptorCount = 1;
    b4.stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    std::vector<VkDescriptorSetLayoutBinding> binds = {b0, b1};
    if (has_tex)   binds.push_back(b2);
    if (has_mesh)  binds.push_back(b3);
    if (has_cells) binds.push_back(b4);
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (std::uint32_t)binds.size(); dslci.pBindings = binds.data();
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                     VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                     VK_SHADER_STAGE_MISS_BIT_KHR;
    pcr.size = sizeof(RtPushConstants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(dev, &plci, nullptr, &playout);

    // ── Stages + groups ─────────────────────────────────────────────────────
    VkPipelineShaderStageCreateInfo stages[4]{};
    auto st = [&](int i, VkShaderStageFlagBits s, VkShaderModule m) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = s; stages[i].module = m; stages[i].pName = "main";
    };
    st(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,       m_rgen);
    st(1, VK_SHADER_STAGE_MISS_BIT_KHR,         m_rmiss);
    st(2, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  m_rchit);
    st(3, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, m_rint);

    VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
    for (auto& g : groups) {
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    // group 0: raygen (general)
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    // group 1: miss (general)
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    // group 2: hit group (procedural: intersection + closest-hit)
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    groups[2].closestHitShader = 2;
    groups[2].intersectionShader = 3;

    VkRayTracingPipelineCreateInfoKHR rtci{};
    rtci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtci.stageCount = 4; rtci.pStages = stages;
    rtci.groupCount = 3; rtci.pGroups = groups;
    rtci.maxPipelineRayRecursionDepth = 1;
    rtci.layout = playout;
    if (create_pipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtci,
                         nullptr, &pipeline) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: vkCreateRayTracingPipelinesKHR failed");

    // ── Shader binding table ────────────────────────────────────────────────
    // Query the handle/alignment sizes from the device.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops{};
    rtprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtprops;
    vkGetPhysicalDeviceProperties2(ctx.physical_device(), &props2);

    handle_size = rtprops.shaderGroupHandleSize;
    std::uint32_t base_align  = rtprops.shaderGroupBaseAlignment;
    std::uint32_t handle_align = rtprops.shaderGroupHandleAlignment;
    VkDeviceSize stride = align_up(handle_size, handle_align);

    // Three groups → three SBT records; we put each region (raygen/miss/hit)
    // base-aligned. For one record per region, region size == stride rounded
    // to base alignment.
    std::uint32_t group_count = 3;
    std::vector<std::uint8_t> handles(handle_size * group_count);
    if (get_handles(dev, pipeline, 0, group_count,
                    handles.size(), handles.data()) != VK_SUCCESS)
        return std::unexpected("rtx_pipeline: get shader group handles failed");

    region = align_up(stride, base_align);
    VkDeviceSize sbt_size = region * 3;
    std::vector<std::uint8_t> sbt_host((std::size_t)sbt_size, 0);
    for (std::uint32_t g = 0; g < 3; ++g)
        std::memcpy(sbt_host.data() + region * g,
                    handles.data() + handle_size * g, handle_size);
    auto sbt_built = mk_device_local_upload(
        ctx, sbt_host.data(), sbt_size,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR);
    if (!sbt_built) return std::unexpected(sbt_built.error());
    sbt = *sbt_built;
    sbt_buf_handle = sbt_built->buf;
    sbt_base = buf_addr(ctx, sbt_built->buf);

    // If a cache was supplied, hand it the freshly built objects so they
    // survive teardown and the next frame is a hit.
    if (cache) {
        cache->pipeline          = pipeline;
        cache->pipeline_layout   = playout;
        cache->descriptor_layout = dsl;
        cache->shader_modules[0] = m_rgen;
        cache->shader_modules[1] = m_rint;
        cache->shader_modules[2] = m_rchit;
        cache->shader_modules[3] = m_rmiss;
        cache->sbt_buffer        = sbt_built->buf;
        cache->sbt_memory        = sbt_built->mem;
        cache->sbt_base          = sbt_base;
        cache->handle_size       = handle_size;
        cache->sbt_base_align    = base_align;
        cache->sbt_handle_align  = handle_align;
        cache->key               = want_key;
        cache->key_src           = key_src;
        cache->valid             = true;
    }
    }  // end of cache-miss build branch

    VkStridedDeviceAddressRegionKHR rgen_r{ sbt_base + region * 0, region, region };
    VkStridedDeviceAddressRegionKHR miss_r{ sbt_base + region * 1, region, region };
    VkStridedDeviceAddressRegionKHR hit_r { sbt_base + region * 2, region, region };
    VkStridedDeviceAddressRegionKHR call_r{ 0, 0, 0 };

    // ── Output image (rgba32f, storage) + readback buffer ───────────────────
    // Cached across frames (reused while the extent is unchanged) for analytic
    // scenes; tex/mesh scenes keep the per-frame path. `frame_cached` gates the
    // per-frame teardown below.
    const bool frame_cacheable = !has_tex && !has_mesh;
    auto frt = acquire_frame_rt(ctx, cache, width, height, frame_cacheable, float_image);
    if (!frt) return std::unexpected(frt.error());
    VkImage        image      = frt->image;
    VkDeviceMemory image_mem  = frt->image_mem;
    VkImageView    image_view = frt->view;
    const bool     frame_cached = frt->cached;

    // ── Texture / mesh storage buffers (bindings 2/3), if present ───────────
    std::optional<Buf> tex_buf;
    if (has_tex) {
        VkDeviceSize tbytes = texture_pixels.size() * sizeof(std::uint32_t);
        auto tb = mk_buf(ctx, tbytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
        if (!tb) return std::unexpected(tb.error());
        void* p = nullptr;
        vkMapMemory(dev, tb->mem, 0, tbytes, 0, &p);
        std::memcpy(p, texture_pixels.data(), (size_t)tbytes);
        vkUnmapMemory(dev, tb->mem);
        tex_buf = *tb;
    }
    std::optional<Buf> mesh_buf;
    if (has_mesh) {
        VkDeviceSize mbytes = mesh_voxels.size() * sizeof(float);
        auto mb = mk_buf(ctx, mbytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
        if (!mb) return std::unexpected(mb.error());
        void* p = nullptr;
        vkMapMemory(dev, mb->mem, 0, mbytes, 0, &p);
        std::memcpy(p, mesh_voxels.data(), (size_t)mbytes);
        vkUnmapMemory(dev, mb->mem);
        mesh_buf = *mb;
    }

    // ── Descriptor pool + set ───────────────────────────────────────────────
    // The descriptor set binds the TLAS and the output image; with both cached
    // it describes the same thing every frame, so build it once.
    VkAccelerationStructureKHR tlas_now = accel.tlas();
    const bool desc_cached = cache && frame_cached && cache->frame_desc_set &&
                             cache->frame_tlas == (const void*)tlas_now;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  dset = VK_NULL_HANDLE;
    if (desc_cached) {
        pool = (VkDescriptorPool)cache->frame_desc_pool;
        dset = (VkDescriptorSet)cache->frame_desc_set;
    } else {
        if (cache && cache->frame_desc_pool)
            vkDestroyDescriptorPool(dev, (VkDescriptorPool)cache->frame_desc_pool, nullptr);
        // One storage-buffer entry sized for every buffer binding the layout
        // may have (texture, mesh voxels, shell cells); an entry larger than
        // what the set uses is allowed.
        VkDescriptorPoolSize ps[3]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
        ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[2].descriptorCount = 3;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes = ps;
        vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(dev, &dsai, &dset);
        if (cache && frame_cached) {
            cache->frame_desc_pool = pool; cache->frame_desc_set = dset;
            cache->frame_tlas = (const void*)tlas_now;
        }
    }

    VkAccelerationStructureKHR tlas = accel.tlas();
    VkWriteDescriptorSetAccelerationStructureKHR was{};
    was.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    was.accelerationStructureCount = 1; was.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.pNext = &was; w0.dstSet = dset; w0.dstBinding = 0;
    w0.descriptorCount = 1; w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    VkDescriptorImageInfo dii{};
    dii.imageView = image_view; dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w1{};
    w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = dset; w1.dstBinding = 1;
    w1.descriptorCount = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.pImageInfo = &dii;
    VkDescriptorBufferInfo dbi{};
    VkWriteDescriptorSet w2{};
    std::vector<VkWriteDescriptorSet> writes = {w0, w1};
    if (has_tex) {
        dbi.buffer = tex_buf->buf; dbi.offset = 0; dbi.range = VK_WHOLE_SIZE;
        w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w2.dstSet = dset; w2.dstBinding = 2;
        w2.descriptorCount = 1; w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w2.pBufferInfo = &dbi;
        writes.push_back(w2);
    }
    VkDescriptorBufferInfo dbi_m{};
    VkWriteDescriptorSet w3{};
    if (has_mesh) {
        dbi_m.buffer = mesh_buf->buf; dbi_m.offset = 0; dbi_m.range = VK_WHOLE_SIZE;
        w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w3.dstSet = dset; w3.dstBinding = 3;
        w3.descriptorCount = 1; w3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w3.pBufferInfo = &dbi_m;
        writes.push_back(w3);
    }
    VkDescriptorBufferInfo dbi_c{};
    VkWriteDescriptorSet w4{};
    if (has_cells) {
        dbi_c.buffer = accel.aabb_buffer(); dbi_c.offset = 0; dbi_c.range = VK_WHOLE_SIZE;
        w4.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w4.dstSet = dset; w4.dstBinding = 4;
        w4.descriptorCount = 1; w4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w4.pBufferInfo = &dbi_c;
        writes.push_back(w4);
    }
    if (!desc_cached)
        vkUpdateDescriptorSets(dev, (std::uint32_t)writes.size(), writes.data(), 0, nullptr);

    // Timestamp queries (profiling only, cached frames only).
    VkQueryPool qpool = VK_NULL_HANDLE;
    if (rtx_profile_on() && cache && frame_cached) {
        if (!cache->frame_query_pool) {
            VkQueryPoolCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = 3;
            VkQueryPool qp = VK_NULL_HANDLE;
            if (vkCreateQueryPool(dev, &qci, nullptr, &qp) == VK_SUCCESS) {
                cache->frame_query_pool = qp;
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
                cache->timestamp_period_ns = props.limits.timestampPeriod;
            }
        }
        qpool = (VkQueryPool)cache->frame_query_pool;
    }

    // ── Command buffer: layout transition, trace, copy to buffer ────────────
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const bool cmd_cached = cache && frame_cached && cache->frame_cmd_buf;
    if (cmd_cached) {
        cpool = (VkCommandPool)cache->frame_cmd_pool;
        cmd   = (VkCommandBuffer)cache->frame_cmd_buf;
        vkResetCommandPool(dev, cpool, 0);   // recorded commands only; keeps the pool
    } else {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = ctx.queue_family();
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(dev, &cpci, nullptr, &cpool);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cbai, &cmd);
        if (cache && frame_cached) { cache->frame_cmd_pool = cpool; cache->frame_cmd_buf = cmd; }
    }

    // readback buffer (from the cached frame resources)
    VkDeviceSize img_bytes = (VkDeviceSize)width * height * (float_image ? 16 : 4);
    VkBuffer       rb_buf = frt->readback;
    VkDeviceMemory rb_mem = frt->readback_mem;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if (qpool) vkCmdResetQueryPool(cmd, qpool, 0, 3);

    // UNDEFINED → GENERAL for storage write
    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image = image;
    toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask = 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            playout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, playout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                       VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                       VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, sizeof(RtPushConstants), &pc);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    cmd_trace(cmd, &rgen_r, &miss_r, &hit_r, &call_r,
              (std::uint32_t)width, (std::uint32_t)height, 1);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);

    // GENERAL → TRANSFER_SRC for copy
    VkImageMemoryBarrier toSrc = toGeneral;
    toSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSrc);

    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           rb_buf, 1, &copy);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 2);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    if (cache && frame_cached && cache->frame_fence) {
        fence = (VkFence)cache->frame_fence;
        vkResetFences(dev, 1, &fence);
    } else {
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(dev, &fci, nullptr, &fence);
        if (cache && frame_cached) cache->frame_fence = fence;
    }
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    wait_fence(dev, fence);   // spins before blocking, see vk_fence_wait.hpp
    auto t_trace1 = clk::now();

    // ── Readback ────────────────────────────────────────────────────────────
    RtRenderResult out;
    out.width = width; out.height = height;
    if (qpool) {
        std::uint64_t ts[3] = {0, 0, 0};
        if (vkGetQueryPoolResults(dev, qpool, 0, 3, sizeof(ts), ts, sizeof(ts[0]),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const double k = cache->timestamp_period_ns * 1e-6;   // ticks → ms
            out.gpu_trace_ms = (double)(ts[1] - ts[0]) * k;
            out.gpu_copy_ms  = (double)(ts[2] - ts[1]) * k;
        }
    }
    auto t_rb0 = clk::now();
    {
        // Cached frames keep their readback buffer mapped; the per-frame
        // (texture/mesh) path still maps and unmaps its own. The frame is
        // handed over as the rgba8 the image holds, or widened to float.
        void* p = frt->mapped;
        const bool transient = (p == nullptr);
        if (transient) vkMapMemory(dev, rb_mem, 0, img_bytes, 0, &p);
        read_frame(p, width, height, float_image, want_rgba8, out);
        if (transient) vkUnmapMemory(dev, rb_mem);
    }
    auto t_rb1 = clk::now();
    // pipeline = entry → just before submit; trace = submit→fence; readback = map.
    out.pipeline_ms = ms(t_pipe0, t_trace0);
    out.trace_ms    = ms(t_trace0, t_trace1);
    out.readback_ms = ms(t_rb0, t_rb1);

    // ── Teardown ────────────────────────────────────────────────────────────
    if (!(cache && frame_cached)) {
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, cpool, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
    }
    // The output image + readback buffer live in the cache when frame_cached;
    // only tear them down for the per-frame (tex/mesh or no-cache) case.
    if (!frame_cached) {
        vkDestroyImageView(dev, image_view, nullptr);
        vkDestroyImage(dev, image, nullptr);
        vkFreeMemory(dev, image_mem, nullptr);
        vkDestroyBuffer(dev, rb_buf, nullptr); vkFreeMemory(dev, rb_mem, nullptr);
    }
    if (tex_buf) { vkDestroyBuffer(dev, tex_buf->buf, nullptr);
                   vkFreeMemory(dev, tex_buf->mem, nullptr); }
    if (mesh_buf) { vkDestroyBuffer(dev, mesh_buf->buf, nullptr);
                    vkFreeMemory(dev, mesh_buf->mem, nullptr); }
    // The pipeline, layouts, shader modules and SBT are owned by the cache when
    // one was supplied (they live across frames); only tear them down in the
    // one-shot case (cache == nullptr). The cache's release() frees them later.
    if (!cache) {
        vkDestroyBuffer(dev, sbt->buf, nullptr); vkFreeMemory(dev, sbt->mem, nullptr);
        vkDestroyPipeline(dev, pipeline, nullptr);
        vkDestroyPipelineLayout(dev, playout, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyShaderModule(dev, m_rgen, nullptr);
        vkDestroyShaderModule(dev, m_rint, nullptr);
        vkDestroyShaderModule(dev, m_rchit, nullptr);
        vkDestroyShaderModule(dev, m_rmiss, nullptr);
    }

    return out;
}

std::expected<RtRenderResult, std::string>
rtx_trace(const RtxCtx& ctx, const RtAccel& accel,
          const std::vector<std::uint32_t>& rgen,
          const std::vector<std::uint32_t>& rint,
          const std::vector<std::uint32_t>& rchit,
          const std::vector<std::uint32_t>& rmiss,
          const RtPushConstants& pc, int width, int height,
          const std::vector<std::uint32_t>& texture_pixels,
          const std::vector<float>& mesh_voxels) {
    return rtx_trace_impl(ctx, accel, nullptr, rgen, rint, rchit, rmiss,
                          pc, width, height, texture_pixels, mesh_voxels, false, false);
}

std::expected<RtRenderResult, std::string>
rtx_trace_cached(const RtxCtx& ctx, const RtAccel& accel,
                 RtxPipelineCache& cache,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::uint32_t>& rint,
                 const std::vector<std::uint32_t>& rchit,
                 const std::vector<std::uint32_t>& rmiss,
                 const RtPushConstants& pc, int width, int height,
                 const std::vector<std::uint32_t>& texture_pixels,
                 const std::vector<float>& mesh_voxels, bool want_rgba8,
                 bool float_image) {
    return rtx_trace_impl(ctx, accel, &cache, rgen, rint, rchit, rmiss,
                          pc, width, height, texture_pixels, mesh_voxels, want_rgba8,
                          float_image);
}

static std::expected<RtRenderResult, std::string>
rtx_trace_groups_impl(const RtxCtx& ctx, const RtAccel& accel,
                 RtxPipelineCache* cache,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                 const std::vector<std::vector<std::uint32_t>>& rchit_per_group,
                 const std::vector<std::uint32_t>& rmiss,
                 const std::vector<std::uint32_t>& rmiss_shadow,
                 const RtPushConstants& pc, int width, int height,
                 bool want_rgba8, bool float_image) {
    auto dev = ctx.device();
    using clk = std::chrono::steady_clock;
    auto msf = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t_pipe0 = clk::now();

    const std::uint32_t ng = (std::uint32_t)rint_per_group.size();
    if (ng == 0) return std::unexpected("rtx_trace_groups: no groups");

    auto create_pipelines = api_cast<PFN_vkCreateRayTracingPipelinesKHR>(
                                ctx.api().createRayTracingPipelines);
    auto get_handles = api_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
                           ctx.api().getRayTracingShaderGroupHandles);
    auto cmd_trace = api_cast<PFN_vkCmdTraceRaysKHR>(ctx.api().cmdTraceRays);

    // Cache key over the whole group shader set (rgen, rchit, rmiss + all rints).
    // Same blobs as the frame before (by identity) → the stored key, no rehash.
    std::vector<std::pair<const void*, std::size_t>> key_src;
    key_src.reserve(4 + rint_per_group.size() + rchit_per_group.size());
    key_src.emplace_back(rgen.data(), rgen.size());
    key_src.emplace_back(rmiss.data(), rmiss.size());
    key_src.emplace_back(rmiss_shadow.data(), rmiss_shadow.size());
    key_src.emplace_back(nullptr, (std::size_t)ng);
    for (const auto& ri : rint_per_group) key_src.emplace_back(ri.data(), ri.size());
    for (const auto& rc : rchit_per_group) key_src.emplace_back(rc.data(), rc.size());
    std::uint64_t want_key = 1469598103934665603ull;
    if (cache && cache->valid && cache->key_src == key_src) {
        want_key = cache->key;
    } else {
        auto mix = [&](const std::vector<std::uint32_t>& v) {
            want_key ^= v.size(); want_key *= 1099511628211ull;
            for (std::uint32_t w : v) { want_key ^= w; want_key *= 1099511628211ull; }
        };
        mix(rgen); mix(rmiss); mix(rmiss_shadow);
        want_key ^= ng; want_key *= 1099511628211ull;
        for (const auto& ri : rint_per_group) mix(ri);
        want_key ^= rchit_per_group.size(); want_key *= 1099511628211ull;
        for (const auto& rc : rchit_per_group) mix(rc);
    }
    const bool cache_hit = cache && cache->valid && cache->key == want_key;
    if (cache && cache->valid && cache->key != want_key) cache->release(ctx);

    // Device SBT alignment props: read once at build time and kept in the
    // cache; a hit takes them from there instead of querying the device.
    std::uint32_t balign = 0, halign = 0;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops{};   // filled on a build
    if (cache_hit) {
        balign = cache->sbt_base_align;
        halign = cache->sbt_handle_align;
    } else {
        rtprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &rtprops;
        vkGetPhysicalDeviceProperties2(ctx.physical_device(), &props2);
        balign = rtprops.shaderGroupBaseAlignment;
        halign = rtprops.shaderGroupHandleAlignment;
    }

    // Objects shared between cache hit and fresh build.
    // One closest-hit module per hit group on the instanced path (each emitted
    // from its own prototype's field, so none of them grows with the scene);
    // one shared module reused for every group on the per-group path.
    const std::uint32_t nch = (std::uint32_t)rchit_per_group.size();
    if (nch != 1 && nch != ng)
        return std::unexpected("rtx_trace_groups: need one closest-hit shader, "
                               "or one per group");
    const bool shadow_miss = !rmiss_shadow.empty();
    const std::uint32_t nmiss = shadow_miss ? 2u : 1u;

    VkShaderModule m_rgen = VK_NULL_HANDLE, m_rmiss = VK_NULL_HANDLE;
    VkShaderModule m_rmiss_sh = VK_NULL_HANDLE;
    std::vector<VkShaderModule> m_rint(ng, VK_NULL_HANDLE);
    std::vector<VkShaderModule> m_rchit(nch, VK_NULL_HANDLE);
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::optional<Buf> sbt;               // set only on a fresh build
    VkDeviceAddress base = 0;
    std::uint32_t hsize = 0;
    VkDeviceSize hstride = 0, rgen_sz = 0, miss_sz = 0, hit_sz = 0;

    if (cache_hit) {
        m_rgen    = (VkShaderModule)cache->shader_modules[0];
        m_rmiss   = (VkShaderModule)cache->shader_modules[3];
        m_rmiss_sh = (VkShaderModule)cache->shader_modules[1];
        for (std::uint32_t i = 0; i < ng; ++i)
            m_rint[i] = (VkShaderModule)cache->group_rint_modules[i];
        for (std::uint32_t i = 0; i < nch; ++i)
            m_rchit[i] = (VkShaderModule)cache->group_rchit_modules[i];
        dsl      = (VkDescriptorSetLayout)cache->descriptor_layout;
        playout  = (VkPipelineLayout)cache->pipeline_layout;
        pipeline = (VkPipeline)cache->pipeline;
        base     = cache->sbt_base;
        hsize    = cache->handle_size;
        hstride  = align_up(hsize, halign);
        rgen_sz  = align_up(hstride, balign);
        miss_sz  = align_up(hstride * nmiss, balign);
        hit_sz   = align_up(hstride * ng, balign);
    } else {
    // ── Shader modules: raygen, miss, closest-hit (shared) + N intersection ─
    m_rgen  = mk_module(dev, rgen);
    m_rmiss = mk_module(dev, rmiss);
    if (shadow_miss) m_rmiss_sh = mk_module(dev, rmiss_shadow);
    for (std::uint32_t i = 0; i < ng; ++i) m_rint[i] = mk_module(dev, rint_per_group[i]);
    for (std::uint32_t i = 0; i < nch; ++i) m_rchit[i] = mk_module(dev, rchit_per_group[i]);

    // Descriptor layout (same as single-BLAS: TLAS + storage image).
    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b0.descriptorCount = 1;
    // Closest-hit reads the TLAS as well on the instanced path: its shadow ray
    // is traced from there rather than marched over the field.
    b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1; b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b1.descriptorCount = 1; b1.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    VkDescriptorSetLayoutBinding binds[] = {b0, b1};
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2; dslci.pBindings = binds;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                     VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    pcr.size = sizeof(RtPushConstants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(dev, &plci, nullptr, &playout);

    // Stages: [0]=rgen, [1..nmiss]=miss (primary, then the shadow ray's),
    // then one closest-hit per distinct shader, then one intersection per group.
    const std::uint32_t st_miss  = 1;
    const std::uint32_t st_chit  = st_miss + nmiss;
    const std::uint32_t st_isect = st_chit + nch;
    std::vector<VkPipelineShaderStageCreateInfo> stages(st_isect + ng);
    auto st = [&](std::uint32_t idx, VkShaderStageFlagBits s, VkShaderModule m) {
        stages[idx] = {};
        stages[idx].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[idx].stage = s; stages[idx].module = m; stages[idx].pName = "main";
    };
    st(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, m_rgen);
    st(st_miss, VK_SHADER_STAGE_MISS_BIT_KHR, m_rmiss);
    if (shadow_miss) st(st_miss + 1, VK_SHADER_STAGE_MISS_BIT_KHR, m_rmiss_sh);
    for (std::uint32_t i = 0; i < nch; ++i)
        st(st_chit + i, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, m_rchit[i]);
    for (std::uint32_t i = 0; i < ng; ++i)
        st(st_isect + i, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, m_rint[i]);

    // Groups: [0]=raygen, [1..nmiss]=miss, then one procedural hit group per
    // group = that group's closest-hit (shared when there is only one) plus its
    // own intersection shader.
    const std::uint32_t gp_hit = 1 + nmiss;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(gp_hit + ng);
    for (auto& g : groups) {
        g = {};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    for (std::uint32_t i = 0; i < nmiss; ++i) {
        groups[1 + i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[1 + i].generalShader = st_miss + i;
    }
    for (std::uint32_t i = 0; i < ng; ++i) {
        groups[gp_hit + i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        groups[gp_hit + i].closestHitShader = st_chit + (nch == 1 ? 0 : i);
        groups[gp_hit + i].intersectionShader = st_isect + i;
    }

    VkRayTracingPipelineCreateInfoKHR rtci{};
    rtci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtci.stageCount = (std::uint32_t)stages.size(); rtci.pStages = stages.data();
    rtci.groupCount = (std::uint32_t)groups.size(); rtci.pGroups = groups.data();
    // Depth 2 when closest-hit traces its own occlusion ray; 1 otherwise, so
    // the per-group path asks the driver for no more than it did before.
    rtci.maxPipelineRayRecursionDepth = shadow_miss ? 2 : 1;
    rtci.layout = playout;
    if (create_pipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtci,
                         nullptr, &pipeline) != VK_SUCCESS)
        return std::unexpected("rtx_trace_groups: create pipeline failed");

    // ── SBT: raygen(1) + miss(1) + hit(N) records ───────────────────────────
    hsize   = rtprops.shaderGroupHandleSize;
    hstride = align_up(hsize, halign);
    std::uint32_t total_groups = gp_hit + ng;
    std::vector<std::uint8_t> handles(hsize * total_groups);
    if (get_handles(dev, pipeline, 0, total_groups, handles.size(), handles.data())
            != VK_SUCCESS)
        return std::unexpected("rtx_trace_groups: get group handles failed");

    // Region layout: [raygen region][miss region][hit region]. Each region base-
    // aligned; the hit region holds ng records of hstride each.
    rgen_sz = align_up(hstride, balign);
    miss_sz = align_up(hstride * nmiss, balign);
    hit_sz  = align_up(hstride * ng, balign);
    VkDeviceSize sbt_size = rgen_sz + miss_sz + hit_sz;
    std::vector<std::uint8_t> sbt_host((std::size_t)sbt_size, 0);
    std::memcpy(sbt_host.data(), handles.data() + hsize * 0, hsize);            // raygen
    for (std::uint32_t i = 0; i < nmiss; ++i)                                   // miss
        std::memcpy(sbt_host.data() + rgen_sz + hstride * i,
                    handles.data() + hsize * (1 + i), hsize);
    for (std::uint32_t i = 0; i < ng; ++i)                                      // hits
        std::memcpy(sbt_host.data() + rgen_sz + miss_sz + hstride * i,
                    handles.data() + hsize * (gp_hit + i), hsize);
    auto sbt_built = mk_device_local_upload(
        ctx, sbt_host.data(), sbt_size,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR);
    if (!sbt_built) return std::unexpected(sbt_built.error());
    sbt  = *sbt_built;
    base = buf_addr(ctx, sbt_built->buf);

    // Hand the freshly built objects to the cache (if any) so they survive
    // teardown and the next frame with the same shaders is a hit.
    if (cache) {
        cache->pipeline          = pipeline;
        cache->pipeline_layout   = playout;
        cache->descriptor_layout = dsl;
        cache->shader_modules[0] = m_rgen;
        cache->shader_modules[1] = m_rmiss_sh;
        cache->shader_modules[2] = nullptr;
        cache->shader_modules[3] = m_rmiss;
        cache->group_rint_modules.assign(ng, nullptr);
        for (std::uint32_t i = 0; i < ng; ++i)
            cache->group_rint_modules[i] = m_rint[i];
        cache->group_rchit_modules.assign(nch, nullptr);
        for (std::uint32_t i = 0; i < nch; ++i)
            cache->group_rchit_modules[i] = m_rchit[i];
        cache->sbt_buffer  = sbt_built->buf;
        cache->sbt_memory  = sbt_built->mem;
        cache->sbt_base    = base;
        cache->handle_size = hsize;
        cache->sbt_base_align   = balign;
        cache->sbt_handle_align = halign;
        cache->key         = want_key;
        cache->key_src     = key_src;
        cache->valid       = true;
    }
    }  // end fresh-build (cache-miss) branch

    VkStridedDeviceAddressRegionKHR rgen_r{ base, hstride, rgen_sz };
    VkStridedDeviceAddressRegionKHR miss_r{ base + rgen_sz, hstride, miss_sz };
    VkStridedDeviceAddressRegionKHR hit_r { base + rgen_sz + miss_sz, hstride, hit_sz };
    VkStridedDeviceAddressRegionKHR call_r{ 0, 0, 0 };

    // ── Output image (rgba32f storage) + readback buffer ────────────────────
    // Cached across frames (the group path is always analytic — no tex/mesh).
    auto frt = acquire_frame_rt(ctx, cache, width, height, /*cacheable=*/true, float_image);
    if (!frt) return std::unexpected(frt.error());
    VkImage        image        = frt->image;
    VkDeviceMemory image_mem    = frt->image_mem;
    VkImageView    image_view   = frt->view;
    const bool     frame_cached = frt->cached;

    VkAccelerationStructureKHR tlas_now = accel.tlas();
    const bool desc_cached = cache && frame_cached && cache->frame_desc_set &&
                             cache->frame_tlas == (const void*)tlas_now;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  dset = VK_NULL_HANDLE;
    if (desc_cached) {
        pool = (VkDescriptorPool)cache->frame_desc_pool;
        dset = (VkDescriptorSet)cache->frame_desc_set;
    } else {
        if (cache && cache->frame_desc_pool)
            vkDestroyDescriptorPool(dev, (VkDescriptorPool)cache->frame_desc_pool, nullptr);
        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
        vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(dev, &dsai, &dset);
        if (cache && frame_cached) {
            cache->frame_desc_pool = pool; cache->frame_desc_set = dset;
            cache->frame_tlas = (const void*)tlas_now;
        }
    }

    VkAccelerationStructureKHR tlas = accel.tlas();
    VkWriteDescriptorSetAccelerationStructureKHR was{};
    was.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    was.accelerationStructureCount = 1; was.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.pNext = &was; w0.dstSet = dset; w0.dstBinding = 0;
    w0.descriptorCount = 1; w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    VkDescriptorImageInfo dii{};
    dii.imageView = image_view; dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w1{};
    w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = dset; w1.dstBinding = 1;
    w1.descriptorCount = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.pImageInfo = &dii;
    VkWriteDescriptorSet writes[] = {w0, w1};
    if (!desc_cached) vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (cache && frame_cached && cache->frame_cmd_buf) {
        cpool = (VkCommandPool)cache->frame_cmd_pool;
        cmd   = (VkCommandBuffer)cache->frame_cmd_buf;
        vkResetCommandPool(dev, cpool, 0);
    } else {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = ctx.queue_family();
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(dev, &cpci, nullptr, &cpool);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cbai, &cmd);
        if (cache && frame_cached) { cache->frame_cmd_pool = cpool; cache->frame_cmd_buf = cmd; }
    }

    VkDeviceSize img_bytes = (VkDeviceSize)width * height * (float_image ? 16 : 4);
    VkBuffer       rb_buf = frt->readback;
    VkDeviceMemory rb_mem = frt->readback_mem;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image = image;
    toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask = 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            playout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, playout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                       VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, sizeof(RtPushConstants), &pc);
    cmd_trace(cmd, &rgen_r, &miss_r, &hit_r, &call_r,
              (std::uint32_t)width, (std::uint32_t)height, 1);

    VkImageMemoryBarrier toSrc = toGeneral;
    toSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSrc);
    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           rb_buf, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    if (cache && frame_cached && cache->frame_fence) {
        fence = (VkFence)cache->frame_fence;
        vkResetFences(dev, 1, &fence);
    } else {
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(dev, &fci, nullptr, &fence);
        if (cache && frame_cached) cache->frame_fence = fence;
    }
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    wait_fence(dev, fence);   // spins before blocking, see vk_fence_wait.hpp
    auto t_trace1 = clk::now();

    RtRenderResult out;
    out.width = width; out.height = height;
    auto t_rb0 = clk::now();
    {
        // Cached frames keep their readback buffer mapped; the per-frame
        // (texture/mesh) path still maps and unmaps its own. The frame is
        // handed over as the rgba8 the image holds, or widened to float.
        void* p = frt->mapped;
        const bool transient = (p == nullptr);
        if (transient) vkMapMemory(dev, rb_mem, 0, img_bytes, 0, &p);
        read_frame(p, width, height, float_image, want_rgba8, out);
        if (transient) vkUnmapMemory(dev, rb_mem);
    }
    auto t_rb1 = clk::now();
    out.pipeline_ms = msf(t_pipe0, t_trace0);
    out.trace_ms    = msf(t_trace0, t_trace1);
    out.readback_ms = msf(t_rb0, t_rb1);

    if (!(cache && frame_cached)) {
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, cpool, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
    }
    if (!frame_cached) {   // image + readback live in the cache on a hit
        vkDestroyImageView(dev, image_view, nullptr);
        vkDestroyImage(dev, image, nullptr);
        vkFreeMemory(dev, image_mem, nullptr);
        vkDestroyBuffer(dev, rb_buf, nullptr); vkFreeMemory(dev, rb_mem, nullptr);
    }
    // Pipeline/layouts/modules/SBT are owned by the cache when supplied (they
    // live across frames); tear them down only in the one-shot case. On a cache
    // hit `sbt` is empty, so this also avoids dereferencing an unset optional.
    if (!cache) {
        vkDestroyBuffer(dev, sbt->buf, nullptr); vkFreeMemory(dev, sbt->mem, nullptr);
        vkDestroyPipeline(dev, pipeline, nullptr);
        vkDestroyPipelineLayout(dev, playout, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyShaderModule(dev, m_rgen, nullptr);
        vkDestroyShaderModule(dev, m_rmiss, nullptr);
        if (m_rmiss_sh) vkDestroyShaderModule(dev, m_rmiss_sh, nullptr);
        for (auto m : m_rchit) vkDestroyShaderModule(dev, m, nullptr);
        for (auto m : m_rint) vkDestroyShaderModule(dev, m, nullptr);
    }

    return out;
}

std::expected<RtRenderResult, std::string>
rtx_trace_groups(const RtxCtx& ctx, const RtAccel& accel,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                 const std::vector<std::vector<std::uint32_t>>& rchit_per_group,
                 const std::vector<std::uint32_t>& rmiss,
                 const std::vector<std::uint32_t>& rmiss_shadow,
                 const RtPushConstants& pc, int width, int height) {
    return rtx_trace_groups_impl(ctx, accel, nullptr, rgen, rint_per_group,
                                 rchit_per_group, rmiss, rmiss_shadow,
                                 pc, width, height, false, false);
}

std::expected<RtRenderResult, std::string>
rtx_trace_groups_cached(const RtxCtx& ctx, const RtAccel& accel,
                        RtxPipelineCache& cache,
                        const std::vector<std::uint32_t>& rgen,
                        const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                        const std::vector<std::vector<std::uint32_t>>& rchit_per_group,
                        const std::vector<std::uint32_t>& rmiss,
                        const std::vector<std::uint32_t>& rmiss_shadow,
                        const RtPushConstants& pc, int width, int height,
                        bool want_rgba8, bool float_image) {
    return rtx_trace_groups_impl(ctx, accel, &cache, rgen, rint_per_group,
                                 rchit_per_group, rmiss, rmiss_shadow,
                                 pc, width, height, want_rgba8, float_image);
}

// ── Ray query: the compute-pipeline trace against the same structure ─────────
//
// Declared in rtx_query.hpp. The per-frame section is the SBT path's with a
// compute dispatch where the trace was; the cache holds a compute pipeline in
// the slots the RT pipeline would occupy, keyed by the compute blob alone.
std::expected<RtRenderResult, std::string>
rtx_query_trace_cached(const RtxCtx& ctx, const RtAccel& accel,
                       RtxPipelineCache& cache_ref,
                       const std::vector<std::uint32_t>& comp,
                       const RtPushConstants& pc, int width, int height,
                       bool want_rgba8, bool float_image) {
    RtxPipelineCache* cache = &cache_ref;
    auto dev = ctx.device();
    if (!ctx.has_ray_query())
        return std::unexpected("rtx_query: VK_KHR_ray_query not enabled on device");
    const bool has_cells = accel.aabb_count() > 1 && accel.aabb_buffer();
    const unsigned binding_flags = 8u | (has_cells ? 4u : 0u);   // 8: compute mode
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t_pipe0 = clk::now();

    const std::vector<std::pair<const void*, std::size_t>> key_src = {
        {comp.data(), comp.size()}, {nullptr, (std::size_t)binding_flags}};
    const std::uint64_t want_key =
        (cache->valid && cache->key_src == key_src)
            ? cache->key
            : shader_set_key(comp, {}, {}, {}, binding_flags);
    const bool cache_hit = cache->valid && cache->key == want_key;
    if (cache->valid && cache->key != want_key) cache->release(ctx);

    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (cache_hit) {
        module   = (VkShaderModule)cache->shader_modules[0];
        dsl      = (VkDescriptorSetLayout)cache->descriptor_layout;
        playout  = (VkPipelineLayout)cache->pipeline_layout;
        pipeline = (VkPipeline)cache->pipeline;
    } else {
        module = mk_module(dev, comp);
        VkDescriptorSetLayoutBinding binds[3]{};
        binds[0].binding = 0; binds[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        binds[0].descriptorCount = 1; binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds[1].binding = 1; binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binds[1].descriptorCount = 1; binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds[2].binding = 2; binds[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[2].descriptorCount = 1; binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = has_cells ? 3u : 2u; dslci.pBindings = binds;
        vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(RtPushConstants);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(dev, &plci, nullptr, &playout);

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module;
        cpci.stage.pName = "main";
        cpci.layout = playout;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline)
            != VK_SUCCESS)
            return std::unexpected("rtx_query: vkCreateComputePipelines failed");

        cache->pipeline          = pipeline;
        cache->pipeline_layout   = playout;
        cache->descriptor_layout = dsl;
        cache->shader_modules[0] = module;
        cache->key               = want_key;
        cache->key_src           = key_src;
        cache->valid             = true;
    }

    // Frame resources, cached across frames (the SBT path's helper).
    auto frt = acquire_frame_rt(ctx, cache, width, height, /*cacheable=*/true, float_image);
    if (!frt) return std::unexpected(frt.error());
    VkImage        image      = frt->image;
    VkImageView    image_view = frt->view;

    // Descriptor set: TLAS + image (+ cells), rebuilt when the structure changes.
    VkAccelerationStructureKHR tlas = accel.tlas();
    const bool desc_cached = cache->frame_desc_set && cache->frame_tlas == (const void*)tlas;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  dset = VK_NULL_HANDLE;
    if (desc_cached) {
        pool = (VkDescriptorPool)cache->frame_desc_pool;
        dset = (VkDescriptorSet)cache->frame_desc_set;
    } else {
        if (cache->frame_desc_pool)
            vkDestroyDescriptorPool(dev, (VkDescriptorPool)cache->frame_desc_pool, nullptr);
        VkDescriptorPoolSize ps[3]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
        ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[2].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1; dpci.poolSizeCount = 3; dpci.pPoolSizes = ps;
        vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(dev, &dsai, &dset);
        cache->frame_desc_pool = pool; cache->frame_desc_set = dset;
        cache->frame_tlas = (const void*)tlas;

        VkWriteDescriptorSetAccelerationStructureKHR was{};
        was.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        was.accelerationStructureCount = 1; was.pAccelerationStructures = &tlas;
        VkWriteDescriptorSet w0{};
        w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0.pNext = &was; w0.dstSet = dset; w0.dstBinding = 0;
        w0.descriptorCount = 1; w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        VkDescriptorImageInfo dii{};
        dii.imageView = image_view; dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w1{};
        w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1.dstSet = dset; w1.dstBinding = 1;
        w1.descriptorCount = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1.pImageInfo = &dii;
        VkDescriptorBufferInfo dbi{};
        VkWriteDescriptorSet w2{};
        std::vector<VkWriteDescriptorSet> writes = {w0, w1};
        if (has_cells) {
            dbi.buffer = accel.aabb_buffer(); dbi.offset = 0; dbi.range = VK_WHOLE_SIZE;
            w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w2.dstSet = dset; w2.dstBinding = 2;
            w2.descriptorCount = 1; w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w2.pBufferInfo = &dbi;
            writes.push_back(w2);
        }
        vkUpdateDescriptorSets(dev, (std::uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Timestamps (profiling only), as the SBT path takes them.
    VkQueryPool qpool = VK_NULL_HANDLE;
    if (rtx_profile_on()) {
        if (!cache->frame_query_pool) {
            VkQueryPoolCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = 3;
            VkQueryPool qp = VK_NULL_HANDLE;
            if (vkCreateQueryPool(dev, &qci, nullptr, &qp) == VK_SUCCESS) {
                cache->frame_query_pool = qp;
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
                cache->timestamp_period_ns = props.limits.timestampPeriod;
            }
        }
        qpool = (VkQueryPool)cache->frame_query_pool;
    }

    // Command buffer, reused.
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (cache->frame_cmd_buf) {
        cpool = (VkCommandPool)cache->frame_cmd_pool;
        cmd   = (VkCommandBuffer)cache->frame_cmd_buf;
        vkResetCommandPool(dev, cpool, 0);
    } else {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = ctx.queue_family();
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(dev, &cpci, nullptr, &cpool);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cbai, &cmd);
        cache->frame_cmd_pool = cpool; cache->frame_cmd_buf = cmd;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if (qpool) vkCmdResetQueryPool(cmd, qpool, 0, 3);

    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image = image;
    toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask = 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtPushConstants), &pc);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    vkCmdDispatch(cmd, ((std::uint32_t)width + 7) / 8, ((std::uint32_t)height + 7) / 8, 1);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);

    VkImageMemoryBarrier toSrc = toGeneral;
    toSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSrc);
    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           frt->readback, 1, &copy);
    if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 2);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    if (cache->frame_fence) {
        fence = (VkFence)cache->frame_fence;
        vkResetFences(dev, 1, &fence);
    } else {
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(dev, &fci, nullptr, &fence);
        cache->frame_fence = fence;
    }
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    wait_fence(dev, fence);
    auto t_trace1 = clk::now();

    RtRenderResult out;
    out.width = width; out.height = height;
    if (qpool) {
        std::uint64_t ts[3] = {0, 0, 0};
        if (vkGetQueryPoolResults(dev, qpool, 0, 3, sizeof(ts), ts, sizeof(ts[0]),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const double k = cache->timestamp_period_ns * 1e-6;
            out.gpu_trace_ms = (double)(ts[1] - ts[0]) * k;
            out.gpu_copy_ms  = (double)(ts[2] - ts[1]) * k;
        }
    }
    auto t_rb0 = clk::now();
    read_frame(frt->mapped, width, height, float_image, want_rgba8, out);
    auto t_rb1 = clk::now();
    out.pipeline_ms = ms(t_pipe0, t_trace0);
    out.trace_ms    = ms(t_trace0, t_trace1);
    out.readback_ms = ms(t_rb0, t_rb1);
    return out;
}

}  // namespace frep::gpu
