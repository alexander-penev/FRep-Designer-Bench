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

// Output rgba32f storage image + its readback buffer for one frame. When
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
    bool           cached       = false;
};

std::expected<FrameRT, std::string>
acquire_frame_rt(const RtxCtx& ctx, RtxPipelineCache* cache,
                 int width, int height, bool cacheable) {
    auto dev = ctx.device();
    if (cache && cacheable && cache->frame_image &&
        cache->frame_w == width && cache->frame_h == height) {
        FrameRT f;
        f.image        = (VkImage)cache->frame_image;
        f.image_mem    = (VkDeviceMemory)cache->frame_image_mem;
        f.view         = (VkImageView)cache->frame_image_view;
        f.readback     = (VkBuffer)cache->frame_readback;
        f.readback_mem = (VkDeviceMemory)cache->frame_readback_mem;
        f.cached       = true;
        return f;
    }
    // Free a stale-size cached frame before rebuilding.
    if (cache && cacheable && cache->frame_image) {
        if (cache->frame_image_view) vkDestroyImageView(dev, (VkImageView)cache->frame_image_view, nullptr);
        vkDestroyImage(dev, (VkImage)cache->frame_image, nullptr);
        vkFreeMemory(dev, (VkDeviceMemory)cache->frame_image_mem, nullptr);
        vkDestroyBuffer(dev, (VkBuffer)cache->frame_readback, nullptr);
        vkFreeMemory(dev, (VkDeviceMemory)cache->frame_readback_mem, nullptr);
        cache->frame_image = cache->frame_image_mem = cache->frame_image_view = nullptr;
        cache->frame_readback = cache->frame_readback_mem = nullptr;
    }

    FrameRT f;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
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
    ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &f.view);

    auto rb = mk_readback_buf(ctx, (VkDeviceSize)width * height * 4 * sizeof(float));
    if (!rb) return std::unexpected(rb.error());
    f.readback = rb->buf; f.readback_mem = rb->mem;

    if (cache && cacheable) {
        cache->frame_image = f.image; cache->frame_image_mem = f.image_mem;
        cache->frame_image_view = f.view; cache->frame_readback = f.readback;
        cache->frame_readback_mem = f.readback_mem;
        cache->frame_w = width; cache->frame_h = height;
        f.cached = true;   // stored in cache → don't tear down per frame
    }
    return f;
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
                                    bool has_tex, bool has_mesh) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::vector<std::uint32_t>& v) {
        h ^= v.size(); h *= 1099511628211ull;
        for (std::uint32_t w : v) { h ^= w; h *= 1099511628211ull; }
    };
    mix(rgen); mix(rint); mix(rchit); mix(rmiss);
    h ^= (has_tex ? 2u : 0u) | (has_mesh ? 1u : 0u);
    h *= 1099511628211ull;
    return h;
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
    if (sbt_buffer) vkDestroyBuffer(dev, (VkBuffer)sbt_buffer, nullptr);
    if (sbt_memory) vkFreeMemory(dev, (VkDeviceMemory)sbt_memory, nullptr);
    if (frame_image_view) vkDestroyImageView(dev, (VkImageView)frame_image_view, nullptr);
    if (frame_image)      vkDestroyImage(dev, (VkImage)frame_image, nullptr);
    if (frame_image_mem)  vkFreeMemory(dev, (VkDeviceMemory)frame_image_mem, nullptr);
    if (frame_readback)   vkDestroyBuffer(dev, (VkBuffer)frame_readback, nullptr);
    if (frame_readback_mem) vkFreeMemory(dev, (VkDeviceMemory)frame_readback_mem, nullptr);
    frame_image = frame_image_mem = frame_image_view = nullptr;
    frame_readback = frame_readback_mem = nullptr;
    frame_w = frame_h = 0;
    pipeline = pipeline_layout = descriptor_layout = nullptr;
    sbt_buffer = sbt_memory = nullptr;
    sbt_base = 0; handle_size = 0; key = 0; valid = false;
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
               const std::vector<float>& mesh_voxels) {
    auto dev = ctx.device();
    const bool has_tex  = !texture_pixels.empty();
    const bool has_mesh = !mesh_voxels.empty();
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
    const std::uint64_t want_key =
        shader_set_key(rgen, rint, rchit, rmiss, has_tex, has_mesh);
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
        // region == base-aligned stride; recompute from device props (cheap).
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp{};
        rtp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2; p2.pNext = &rtp;
        vkGetPhysicalDeviceProperties2(ctx.physical_device(), &p2);
        VkDeviceSize stride = align_up(rtp.shaderGroupHandleSize,
                                       rtp.shaderGroupHandleAlignment);
        region = align_up(stride, rtp.shaderGroupBaseAlignment);
    } else {

    // ── Shader modules ──────────────────────────────────────────────────────
    m_rgen  = mk_module(dev, rgen);
    m_rint  = mk_module(dev, rint);
    m_rchit = mk_module(dev, rchit);
    m_rmiss = mk_module(dev, rmiss);

    // ── Descriptor set layout: TLAS + storage image ─────────────────────────
    VkDescriptorSetLayoutBinding b0{};  // TLAS
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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
    std::vector<VkDescriptorSetLayoutBinding> binds = {b0, b1};
    if (has_tex)  binds.push_back(b2);
    if (has_mesh) binds.push_back(b3);
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
    auto sbt_built = mk_buf(ctx, sbt_size,
                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    if (!sbt_built) return std::unexpected(sbt_built.error());
    {
        std::uint8_t* p = nullptr;
        vkMapMemory(dev, sbt_built->mem, 0, sbt_size, 0, (void**)&p);
        std::memset(p, 0, sbt_size);
        for (std::uint32_t g = 0; g < 3; ++g)
            std::memcpy(p + region * g, handles.data() + handle_size * g, handle_size);
        vkUnmapMemory(dev, sbt_built->mem);
    }
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
        cache->key               = want_key;
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
    auto frt = acquire_frame_rt(ctx, cache, width, height, frame_cacheable);
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
    VkDescriptorPoolSize ps[4]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
    ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[2].descriptorCount = 1;
    ps[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[3].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2u + (has_tex ? 1u : 0u) + (has_mesh ? 1u : 0u);
    dpci.pPoolSizes = ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &dset);

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
    vkUpdateDescriptorSets(dev, (std::uint32_t)writes.size(), writes.data(), 0, nullptr);

    // ── Command buffer: layout transition, trace, copy to buffer ────────────
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = ctx.queue_family();
    VkCommandPool cpool = VK_NULL_HANDLE;
    vkCreateCommandPool(dev, &cpci, nullptr, &cpool);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    // readback buffer (from the cached frame resources)
    VkDeviceSize img_bytes = (VkDeviceSize)width * height * 4 * sizeof(float);
    VkBuffer       rb_buf = frt->readback;
    VkDeviceMemory rb_mem = frt->readback_mem;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

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
    cmd_trace(cmd, &rgen_r, &miss_r, &hit_r, &call_r,
              (std::uint32_t)width, (std::uint32_t)height, 1);

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

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(dev, &fci, nullptr, &fence);
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull);
    auto t_trace1 = clk::now();

    // ── Readback ────────────────────────────────────────────────────────────
    RtRenderResult out;
    out.width = width; out.height = height;
    out.rgba.resize((std::size_t)width * height * 4);
    auto t_rb0 = clk::now();
    {
        void* p = nullptr;
        vkMapMemory(dev, rb_mem, 0, img_bytes, 0, &p);
        std::memcpy(out.rgba.data(), p, (size_t)img_bytes);
        vkUnmapMemory(dev, rb_mem);
    }
    auto t_rb1 = clk::now();
    // pipeline = entry → just before submit; trace = submit→fence; readback = map.
    out.pipeline_ms = ms(t_pipe0, t_trace0);
    out.trace_ms    = ms(t_trace0, t_trace1);
    out.readback_ms = ms(t_rb0, t_rb1);

    // ── Teardown ────────────────────────────────────────────────────────────
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
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
                          pc, width, height, texture_pixels, mesh_voxels);
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
                 const std::vector<float>& mesh_voxels) {
    return rtx_trace_impl(ctx, accel, &cache, rgen, rint, rchit, rmiss,
                          pc, width, height, texture_pixels, mesh_voxels);
}

static std::expected<RtRenderResult, std::string>
rtx_trace_groups_impl(const RtxCtx& ctx, const RtAccel& accel,
                 RtxPipelineCache* cache,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                 const std::vector<std::uint32_t>& rchit,
                 const std::vector<std::uint32_t>& rmiss,
                 const RtPushConstants& pc, int width, int height) {
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
    std::uint64_t want_key = 1469598103934665603ull;
    {
        auto mix = [&](const std::vector<std::uint32_t>& v) {
            want_key ^= v.size(); want_key *= 1099511628211ull;
            for (std::uint32_t w : v) { want_key ^= w; want_key *= 1099511628211ull; }
        };
        mix(rgen); mix(rchit); mix(rmiss);
        want_key ^= ng; want_key *= 1099511628211ull;
        for (const auto& ri : rint_per_group) mix(ri);
    }
    const bool cache_hit = cache && cache->valid && cache->key == want_key;
    if (cache && cache->valid && cache->key != want_key) cache->release(ctx);

    // Device SBT alignment props — both the hit and miss branches need them to
    // compute the region layout.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops{};
    rtprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtprops;
    vkGetPhysicalDeviceProperties2(ctx.physical_device(), &props2);
    const std::uint32_t balign = rtprops.shaderGroupBaseAlignment;
    const std::uint32_t halign = rtprops.shaderGroupHandleAlignment;

    // Objects shared between cache hit and fresh build.
    VkShaderModule m_rgen = VK_NULL_HANDLE, m_rmiss = VK_NULL_HANDLE, m_rchit = VK_NULL_HANDLE;
    std::vector<VkShaderModule> m_rint(ng, VK_NULL_HANDLE);
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::optional<Buf> sbt;               // set only on a fresh build
    VkDeviceAddress base = 0;
    std::uint32_t hsize = 0;
    VkDeviceSize hstride = 0, rgen_sz = 0, miss_sz = 0, hit_sz = 0;

    if (cache_hit) {
        m_rgen  = (VkShaderModule)cache->shader_modules[0];
        m_rchit = (VkShaderModule)cache->shader_modules[2];
        m_rmiss = (VkShaderModule)cache->shader_modules[3];
        for (std::uint32_t i = 0; i < ng; ++i)
            m_rint[i] = (VkShaderModule)cache->group_rint_modules[i];
        dsl      = (VkDescriptorSetLayout)cache->descriptor_layout;
        playout  = (VkPipelineLayout)cache->pipeline_layout;
        pipeline = (VkPipeline)cache->pipeline;
        base     = cache->sbt_base;
        hsize    = cache->handle_size;
        hstride  = align_up(hsize, halign);
        rgen_sz  = align_up(hstride, balign);
        miss_sz  = align_up(hstride, balign);
        hit_sz   = align_up(hstride * ng, balign);
    } else {
    // ── Shader modules: raygen, miss, closest-hit (shared) + N intersection ─
    m_rgen  = mk_module(dev, rgen);
    m_rmiss = mk_module(dev, rmiss);
    m_rchit = mk_module(dev, rchit);
    for (std::uint32_t i = 0; i < ng; ++i) m_rint[i] = mk_module(dev, rint_per_group[i]);

    // Descriptor layout (same as single-BLAS: TLAS + storage image).
    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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

    // Stages: [0]=rgen [1]=rmiss [2]=rchit [3..]=intersection per group.
    std::vector<VkPipelineShaderStageCreateInfo> stages(3 + ng);
    auto st = [&](std::uint32_t idx, VkShaderStageFlagBits s, VkShaderModule m) {
        stages[idx] = {};
        stages[idx].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[idx].stage = s; stages[idx].module = m; stages[idx].pName = "main";
    };
    st(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,      m_rgen);
    st(1, VK_SHADER_STAGE_MISS_BIT_KHR,        m_rmiss);
    st(2, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, m_rchit);
    for (std::uint32_t i = 0; i < ng; ++i)
        st(3 + i, VK_SHADER_STAGE_INTERSECTION_BIT_KHR, m_rint[i]);

    // Groups: [0]=raygen [1]=miss [2..]=one procedural hit group per group,
    // each = shared closest-hit (stage 2) + that group's intersection (3+i).
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(2 + ng);
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
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    for (std::uint32_t i = 0; i < ng; ++i) {
        groups[2 + i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        groups[2 + i].closestHitShader = 2;
        groups[2 + i].intersectionShader = 3 + i;
    }

    VkRayTracingPipelineCreateInfoKHR rtci{};
    rtci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtci.stageCount = (std::uint32_t)stages.size(); rtci.pStages = stages.data();
    rtci.groupCount = (std::uint32_t)groups.size(); rtci.pGroups = groups.data();
    rtci.maxPipelineRayRecursionDepth = 1;
    rtci.layout = playout;
    if (create_pipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtci,
                         nullptr, &pipeline) != VK_SUCCESS)
        return std::unexpected("rtx_trace_groups: create pipeline failed");

    // ── SBT: raygen(1) + miss(1) + hit(N) records ───────────────────────────
    hsize   = rtprops.shaderGroupHandleSize;
    hstride = align_up(hsize, halign);
    std::uint32_t total_groups = 2 + ng;
    std::vector<std::uint8_t> handles(hsize * total_groups);
    if (get_handles(dev, pipeline, 0, total_groups, handles.size(), handles.data())
            != VK_SUCCESS)
        return std::unexpected("rtx_trace_groups: get group handles failed");

    // Region layout: [raygen region][miss region][hit region]. Each region base-
    // aligned; the hit region holds ng records of hstride each.
    rgen_sz = align_up(hstride, balign);
    miss_sz = align_up(hstride, balign);
    hit_sz  = align_up(hstride * ng, balign);
    VkDeviceSize sbt_size = rgen_sz + miss_sz + hit_sz;
    auto sbt_built = mk_buf(ctx, sbt_size,
                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    if (!sbt_built) return std::unexpected(sbt_built.error());
    {
        std::uint8_t* p = nullptr;
        vkMapMemory(dev, sbt_built->mem, 0, sbt_size, 0, (void**)&p);
        std::memset(p, 0, sbt_size);
        std::memcpy(p, handles.data() + hsize * 0, hsize);            // raygen
        std::memcpy(p + rgen_sz, handles.data() + hsize * 1, hsize);  // miss
        for (std::uint32_t i = 0; i < ng; ++i)                        // hits
            std::memcpy(p + rgen_sz + miss_sz + hstride * i,
                        handles.data() + hsize * (2 + i), hsize);
        vkUnmapMemory(dev, sbt_built->mem);
    }
    sbt  = *sbt_built;
    base = buf_addr(ctx, sbt_built->buf);

    // Hand the freshly built objects to the cache (if any) so they survive
    // teardown and the next frame with the same shaders is a hit.
    if (cache) {
        cache->pipeline          = pipeline;
        cache->pipeline_layout   = playout;
        cache->descriptor_layout = dsl;
        cache->shader_modules[0] = m_rgen;
        cache->shader_modules[1] = nullptr;
        cache->shader_modules[2] = m_rchit;
        cache->shader_modules[3] = m_rmiss;
        cache->group_rint_modules.assign(ng, nullptr);
        for (std::uint32_t i = 0; i < ng; ++i)
            cache->group_rint_modules[i] = m_rint[i];
        cache->sbt_buffer  = sbt_built->buf;
        cache->sbt_memory  = sbt_built->mem;
        cache->sbt_base    = base;
        cache->handle_size = hsize;
        cache->key         = want_key;
        cache->valid       = true;
    }
    }  // end fresh-build (cache-miss) branch

    VkStridedDeviceAddressRegionKHR rgen_r{ base, hstride, rgen_sz };
    VkStridedDeviceAddressRegionKHR miss_r{ base + rgen_sz, hstride, miss_sz };
    VkStridedDeviceAddressRegionKHR hit_r { base + rgen_sz + miss_sz, hstride, hit_sz };
    VkStridedDeviceAddressRegionKHR call_r{ 0, 0, 0 };

    // ── Output image (rgba32f storage) + readback buffer ────────────────────
    // Cached across frames (the group path is always analytic — no tex/mesh).
    auto frt = acquire_frame_rt(ctx, cache, width, height, /*cacheable=*/true);
    if (!frt) return std::unexpected(frt.error());
    VkImage        image        = frt->image;
    VkDeviceMemory image_mem    = frt->image_mem;
    VkImageView    image_view   = frt->view;
    const bool     frame_cached = frt->cached;

    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &dset);

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
    vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = ctx.queue_family();
    VkCommandPool cpool = VK_NULL_HANDLE;
    vkCreateCommandPool(dev, &cpci, nullptr, &cpool);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    VkDeviceSize img_bytes = (VkDeviceSize)width * height * 4 * sizeof(float);
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
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(dev, &fci, nullptr, &fence);
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull);
    auto t_trace1 = clk::now();

    RtRenderResult out;
    out.width = width; out.height = height;
    out.rgba.resize((std::size_t)width * height * 4);
    auto t_rb0 = clk::now();
    {
        void* p = nullptr;
        vkMapMemory(dev, rb_mem, 0, img_bytes, 0, &p);
        std::memcpy(out.rgba.data(), p, (size_t)img_bytes);
        vkUnmapMemory(dev, rb_mem);
    }
    auto t_rb1 = clk::now();
    out.pipeline_ms = msf(t_pipe0, t_trace0);
    out.trace_ms    = msf(t_trace0, t_trace1);
    out.readback_ms = msf(t_rb0, t_rb1);

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
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
        vkDestroyShaderModule(dev, m_rchit, nullptr);
        for (auto m : m_rint) vkDestroyShaderModule(dev, m, nullptr);
    }

    return out;
}

std::expected<RtRenderResult, std::string>
rtx_trace_groups(const RtxCtx& ctx, const RtAccel& accel,
                 const std::vector<std::uint32_t>& rgen,
                 const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                 const std::vector<std::uint32_t>& rchit,
                 const std::vector<std::uint32_t>& rmiss,
                 const RtPushConstants& pc, int width, int height) {
    return rtx_trace_groups_impl(ctx, accel, nullptr, rgen, rint_per_group,
                                 rchit, rmiss, pc, width, height);
}

std::expected<RtRenderResult, std::string>
rtx_trace_groups_cached(const RtxCtx& ctx, const RtAccel& accel,
                        RtxPipelineCache& cache,
                        const std::vector<std::uint32_t>& rgen,
                        const std::vector<std::vector<std::uint32_t>>& rint_per_group,
                        const std::vector<std::uint32_t>& rchit,
                        const std::vector<std::uint32_t>& rmiss,
                        const RtPushConstants& pc, int width, int height) {
    return rtx_trace_groups_impl(ctx, accel, &cache, rgen, rint_per_group,
                                 rchit, rmiss, pc, width, height);
}

}  // namespace frep::gpu
