// core/gpu/rtx_accel.cpp — see rtx_accel.hpp.
//
// This is dense Vulkan RT code that the sandbox cannot execute (no Vulkan
// device here), so it is written defensively and validated on llvmpipe /
// hardware. The flow for each acceleration structure is the standard KHR
// pattern: describe geometry → query build sizes → allocate AS buffer +
// scratch → create AS handle → record a build command → submit and wait.

#include "core/gpu/rtx_accel.hpp"

#include <vulkan/vulkan.h>

#include <cstring>
#include <functional>
#include <utility>

namespace frep::gpu {

namespace {

// Cast helpers for the entry points RtxCtx loaded as void*.
template <class F> F api_cast(void* p) { return reinterpret_cast<F>(p); }

int find_mem_type(VkPhysicalDevice phys, std::uint32_t bits,
                  VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

// Allocate a buffer + memory. When `device_address` is set, the buffer is
// created with the SHADER_DEVICE_ADDRESS usage/flag so its GPU address can be
// taken (required for AS inputs and the AS backing store).
std::expected<void, std::string>
make_buffer(const RtxCtx& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
            VkMemoryPropertyFlags mem_props, bool device_address,
            VkBuffer& out_buf, VkDeviceMemory& out_mem) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = usage | (device_address
                ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device(), &bci, nullptr, &out_buf) != VK_SUCCESS)
        return std::unexpected("rtx_accel: vkCreateBuffer failed");

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(ctx.device(), out_buf, &mr);
    int t = find_mem_type(ctx.physical_device(), mr.memoryTypeBits, mem_props);
    if (t < 0) return std::unexpected("rtx_accel: no suitable memory type");

    VkMemoryAllocateFlagsInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = (std::uint32_t)t;
    if (device_address) mai.pNext = &fi;
    if (vkAllocateMemory(ctx.device(), &mai, nullptr, &out_mem) != VK_SUCCESS)
        return std::unexpected("rtx_accel: vkAllocateMemory failed");
    vkBindBufferMemory(ctx.device(), out_buf, out_mem, 0);
    return {};
}

VkDeviceAddress buffer_address(const RtxCtx& ctx, VkBuffer buf) {
    // vkGetBufferDeviceAddress is core in Vulkan 1.2, so call it directly
    // rather than through a vkGetDeviceProcAddr-loaded pointer: on some
    // software loaders (llvmpipe) the proc-addr lookup for a promoted-to-core
    // function returns a non-null but unusable trampoline that faults when
    // called. The link-time symbol is the reliable one here.
    VkBufferDeviceAddressInfo bdai{};
    bdai.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bdai.buffer = buf;
    return vkGetBufferDeviceAddress(ctx.device(), &bdai);
}

// Upload host bytes into a freshly created host-visible buffer.
std::expected<void, std::string>
upload(const RtxCtx& ctx, const void* data, VkDeviceSize size,
       VkBufferUsageFlags usage, bool device_address,
       VkBuffer& buf, VkDeviceMemory& mem) {
    auto r = make_buffer(ctx, size, usage,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         device_address, buf, mem);
    if (!r) return r;
    void* p = nullptr;
    vkMapMemory(ctx.device(), mem, 0, size, 0, &p);
    std::memcpy(p, data, (size_t)size);
    vkUnmapMemory(ctx.device(), mem);
    return {};
}

// Run a one-shot command buffer that records `rec` and wait for it.
std::expected<void, std::string>
one_shot(const RtxCtx& ctx, const std::function<void(VkCommandBuffer)>& rec) {
    VkCommandPoolCreateInfo pci{};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = ctx.queue_family();
    pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool) != VK_SUCCESS)
        return std::unexpected("rtx_accel: vkCreateCommandPool failed");

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = pool;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device(), &cbi, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    rec(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(ctx.device(), &fci, nullptr, &fence);
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, ~0ull);

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return {};
}

}  // namespace

RtAccel::~RtAccel() { destroy(); }
RtAccel::RtAccel(RtAccel&& o) noexcept { *this = std::move(o); }

RtAccel& RtAccel::operator=(RtAccel&& o) noexcept {
    if (this != &o) {
        destroy();
        ctx_ = o.ctx_;
        aabb_buf_ = o.aabb_buf_; aabb_mem_ = o.aabb_mem_;
        blas_buf_ = o.blas_buf_; blas_mem_ = o.blas_mem_;
        tlas_buf_ = o.tlas_buf_; tlas_mem_ = o.tlas_mem_;
        inst_buf_ = o.inst_buf_; inst_mem_ = o.inst_mem_;
        scratch_buf_ = o.scratch_buf_; scratch_mem_ = o.scratch_mem_;
        blas_ = o.blas_; tlas_ = o.tlas_; tlas_addr_ = o.tlas_addr_;
        inst_count_ = o.inst_count_;
        g_aabb_buf_ = std::move(o.g_aabb_buf_); g_aabb_mem_ = std::move(o.g_aabb_mem_);
        g_blas_buf_ = std::move(o.g_blas_buf_); g_blas_mem_ = std::move(o.g_blas_mem_);
        g_scratch_buf_ = std::move(o.g_scratch_buf_);
        g_scratch_mem_ = std::move(o.g_scratch_mem_);
        g_blas_ = std::move(o.g_blas_);
        o.ctx_ = nullptr; o.blas_ = nullptr; o.tlas_ = nullptr;
        o.aabb_buf_ = o.blas_buf_ = o.tlas_buf_ = o.inst_buf_ = o.scratch_buf_ = nullptr;
        o.aabb_mem_ = o.blas_mem_ = o.tlas_mem_ = o.inst_mem_ = o.scratch_mem_ = nullptr;
    }
    return *this;
}

void RtAccel::destroy() {
    if (!ctx_) return;
    auto dev = ctx_->device();
    auto destroy_as = api_cast<PFN_vkDestroyAccelerationStructureKHR>(
                          ctx_->api().destroyAccelerationStructure);
    if (tlas_) destroy_as(dev, tlas_, nullptr);
    if (blas_) destroy_as(dev, blas_, nullptr);
    auto db = [&](VkBuffer b, VkDeviceMemory m) {
        if (b) vkDestroyBuffer(dev, b, nullptr);
        if (m) vkFreeMemory(dev, m, nullptr);
    };
    db(scratch_buf_, scratch_mem_); db(inst_buf_, inst_mem_);
    db(tlas_buf_, tlas_mem_); db(blas_buf_, blas_mem_); db(aabb_buf_, aabb_mem_);
    // Multi-BLAS resources.
    for (auto as : g_blas_) if (as) destroy_as(dev, as, nullptr);
    for (std::size_t i = 0; i < g_blas_buf_.size(); ++i) {
        db(g_aabb_buf_[i], g_aabb_mem_[i]);
        db(g_blas_buf_[i], g_blas_mem_[i]);
        db(g_scratch_buf_[i], g_scratch_mem_[i]);
    }
    g_blas_.clear(); g_aabb_buf_.clear(); g_blas_buf_.clear(); g_scratch_buf_.clear();
    g_aabb_mem_.clear(); g_blas_mem_.clear(); g_scratch_mem_.clear();
    ctx_ = nullptr; tlas_ = blas_ = nullptr;
}

std::expected<RtAccel, std::string>
RtAccel::build_whole_scene(const RtxCtx& ctx, const FRepNode::AABB& b) {
    RtAabb box;
    box.min[0] = b.min_x; box.min[1] = b.min_y; box.min[2] = b.min_z;
    box.max[0] = b.max_x; box.max[1] = b.max_y; box.max[2] = b.max_z;
    return build(ctx, {box});
}

std::expected<RtAccel, std::string>
RtAccel::build(const RtxCtx& ctx, const std::vector<RtAabb>& boxes) {
    if (boxes.empty())
        return std::unexpected("rtx_accel: need at least one AABB");

    RtAccel a;
    a.ctx_ = &ctx;
    auto dev = ctx.device();

    auto get_sizes = api_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                         ctx.api().getAccelerationStructureBuildSizes);
    auto create_as = api_cast<PFN_vkCreateAccelerationStructureKHR>(
                         ctx.api().createAccelerationStructure);
    auto cmd_build = api_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                         ctx.api().cmdBuildAccelerationStructures);
    auto as_addr   = api_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                         ctx.api().getAccelerationStructureDeviceAddress);

    // ── BLAS: an AABBs geometry over `boxes` ────────────────────────────────
    // VkAabbPositionsKHR is exactly {minX,minY,minZ,maxX,maxY,maxZ} floats, so
    // our RtAabb maps 1:1.
    {
        VkDeviceSize aabb_bytes = boxes.size() * sizeof(VkAabbPositionsKHR);
        auto up = upload(ctx, boxes.data(), aabb_bytes,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         /*device_address=*/true, a.aabb_buf_, a.aabb_mem_);
        if (!up) return std::unexpected(up.error());

        VkAccelerationStructureGeometryKHR geo{};
        geo.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geo.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geo.geometry.aabbs.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geo.geometry.aabbs.data.deviceAddress = buffer_address(ctx, a.aabb_buf_);
        geo.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries   = &geo;

        std::uint32_t prim_count = (std::uint32_t)boxes.size();
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                  &bgi, &prim_count, &sizes);

        auto rb = make_buffer(ctx, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.blas_buf_, a.blas_mem_);
        if (!rb) return std::unexpected(rb.error());
        auto rs = make_buffer(ctx, sizes.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.scratch_buf_, a.scratch_mem_);
        if (!rs) return std::unexpected(rs.error());

        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = a.blas_buf_;
        ci.size   = sizes.accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (create_as(dev, &ci, nullptr, &a.blas_) != VK_SUCCESS)
            return std::unexpected("rtx_accel: create BLAS failed");

        bgi.dstAccelerationStructure  = a.blas_;
        bgi.scratchData.deviceAddress = buffer_address(ctx, a.scratch_buf_);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = prim_count;
        const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;

        auto r = one_shot(ctx, [&](VkCommandBuffer cmd) {
            cmd_build(cmd, 1, &bgi, &pr);
        });
        if (!r) return std::unexpected(r.error());
    }

    // ── TLAS: one instance referencing the BLAS (identity transform) ────────
    {
        VkAccelerationStructureDeviceAddressInfoKHR ai{};
        ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        ai.accelerationStructure = a.blas_;
        VkDeviceAddress blas_addr = as_addr(dev, &ai);

        VkAccelerationStructureInstanceKHR inst{};
        // identity 3x4 transform
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.instanceCustomIndex = 0;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas_addr;

        auto up = upload(ctx, &inst, sizeof(inst),
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         true, a.inst_buf_, a.inst_mem_);
        if (!up) return std::unexpected(up.error());

        VkAccelerationStructureGeometryKHR geo{};
        geo.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo.geometry.instances.arrayOfPointers = VK_FALSE;
        geo.geometry.instances.data.deviceAddress = buffer_address(ctx, a.inst_buf_);

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries   = &geo;

        std::uint32_t one = 1;
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                  &bgi, &one, &sizes);

        auto rb = make_buffer(ctx, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.tlas_buf_, a.tlas_mem_);
        if (!rb) return std::unexpected(rb.error());
        // reuse a fresh scratch (TLAS scratch need may differ); allocate its own
        VkBuffer tscratch = nullptr; VkDeviceMemory tscratch_mem = nullptr;
        auto rs = make_buffer(ctx, sizes.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              tscratch, tscratch_mem);
        if (!rs) return std::unexpected(rs.error());

        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = a.tlas_buf_;
        ci.size   = sizes.accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if (create_as(dev, &ci, nullptr, &a.tlas_) != VK_SUCCESS) {
            vkDestroyBuffer(dev, tscratch, nullptr);
            vkFreeMemory(dev, tscratch_mem, nullptr);
            return std::unexpected("rtx_accel: create TLAS failed");
        }

        bgi.dstAccelerationStructure  = a.tlas_;
        bgi.scratchData.deviceAddress = buffer_address(ctx, tscratch);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = 1;
        const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;

        auto r = one_shot(ctx, [&](VkCommandBuffer cmd) {
            cmd_build(cmd, 1, &bgi, &pr);
        });
        vkDestroyBuffer(dev, tscratch, nullptr);
        vkFreeMemory(dev, tscratch_mem, nullptr);
        if (!r) return std::unexpected(r.error());

        VkAccelerationStructureDeviceAddressInfoKHR tai{};
        tai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        tai.accelerationStructure = a.tlas_;
        a.tlas_addr_ = as_addr(dev, &tai);
    }

    return a;
}

std::expected<RtAccel, std::string>
RtAccel::build_groups(const RtxCtx& ctx, const std::vector<RtAabb>& group_boxes) {
    if (group_boxes.empty())
        return std::unexpected("rtx_accel: build_groups needs >=1 box");

    auto dev = ctx.device();
    auto get_sizes = api_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                         ctx.api().getAccelerationStructureBuildSizes);
    auto create_as = api_cast<PFN_vkCreateAccelerationStructureKHR>(
                         ctx.api().createAccelerationStructure);
    auto cmd_build = api_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                         ctx.api().cmdBuildAccelerationStructures);
    auto as_addr   = api_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                         ctx.api().getAccelerationStructureDeviceAddress);

    RtAccel a;
    a.ctx_ = &ctx;
    const std::uint32_t n = (std::uint32_t)group_boxes.size();
    a.inst_count_ = n;
    a.g_aabb_buf_.resize(n, nullptr);  a.g_aabb_mem_.resize(n, nullptr);
    a.g_blas_buf_.resize(n, nullptr);  a.g_blas_mem_.resize(n, nullptr);
    a.g_scratch_buf_.resize(n, nullptr); a.g_scratch_mem_.resize(n, nullptr);
    a.g_blas_.resize(n, nullptr);

    // ── One BLAS per group box ──────────────────────────────────────────────
    for (std::uint32_t i = 0; i < n; ++i) {
        VkAabbPositionsKHR aabb{ group_boxes[i].min[0], group_boxes[i].min[1],
                                 group_boxes[i].min[2], group_boxes[i].max[0],
                                 group_boxes[i].max[1], group_boxes[i].max[2] };
        auto up = upload(ctx, &aabb, sizeof(aabb),
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         true, a.g_aabb_buf_[i], a.g_aabb_mem_[i]);
        if (!up) return std::unexpected(up.error());

        VkAccelerationStructureGeometryKHR geo{};
        geo.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geo.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geo.geometry.aabbs.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geo.geometry.aabbs.data.deviceAddress = buffer_address(ctx, a.g_aabb_buf_[i]);
        geo.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries   = &geo;

        std::uint32_t prim = 1;
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                  &bgi, &prim, &sizes);

        auto rb = make_buffer(ctx, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.g_blas_buf_[i], a.g_blas_mem_[i]);
        if (!rb) return std::unexpected(rb.error());
        auto rs = make_buffer(ctx, sizes.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.g_scratch_buf_[i], a.g_scratch_mem_[i]);
        if (!rs) return std::unexpected(rs.error());

        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = a.g_blas_buf_[i];
        ci.size   = sizes.accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (create_as(dev, &ci, nullptr, &a.g_blas_[i]) != VK_SUCCESS)
            return std::unexpected("rtx_accel: create group BLAS failed");

        bgi.dstAccelerationStructure  = a.g_blas_[i];
        bgi.scratchData.deviceAddress = buffer_address(ctx, a.g_scratch_buf_[i]);
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = 1;
        const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
        auto r = one_shot(ctx, [&](VkCommandBuffer cmd) {
            cmd_build(cmd, 1, &bgi, &pr);
        });
        if (!r) return std::unexpected(r.error());
    }

    // ── TLAS over N instances, instance i → BLAS i, SBT offset i ────────────
    {
        std::vector<VkAccelerationStructureInstanceKHR> insts(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            VkAccelerationStructureDeviceAddressInfoKHR ai{};
            ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            ai.accelerationStructure = a.g_blas_[i];
            VkDeviceAddress baddr = as_addr(dev, &ai);

            auto& inst = insts[i];
            inst = {};
            inst.transform.matrix[0][0] = 1.0f;
            inst.transform.matrix[1][1] = 1.0f;
            inst.transform.matrix[2][2] = 1.0f;
            inst.instanceCustomIndex = i;
            inst.mask = 0xFF;
            // Each group gets its own hit record → its own intersection shader.
            inst.instanceShaderBindingTableRecordOffset = i;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = baddr;
        }

        auto up = upload(ctx, insts.data(),
                         insts.size() * sizeof(VkAccelerationStructureInstanceKHR),
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         true, a.inst_buf_, a.inst_mem_);
        if (!up) return std::unexpected(up.error());

        VkAccelerationStructureGeometryKHR geo{};
        geo.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo.geometry.instances.arrayOfPointers = VK_FALSE;
        geo.geometry.instances.data.deviceAddress = buffer_address(ctx, a.inst_buf_);

        VkAccelerationStructureBuildGeometryInfoKHR bgi{};
        bgi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bgi.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bgi.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bgi.geometryCount = 1;
        bgi.pGeometries   = &geo;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                  &bgi, &n, &sizes);

        auto rb = make_buffer(ctx, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              a.tlas_buf_, a.tlas_mem_);
        if (!rb) return std::unexpected(rb.error());
        VkBuffer tscratch = nullptr; VkDeviceMemory tscratch_mem = nullptr;
        auto rs = make_buffer(ctx, sizes.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true,
                              tscratch, tscratch_mem);
        if (!rs) return std::unexpected(rs.error());

        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = a.tlas_buf_;
        ci.size   = sizes.accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if (create_as(dev, &ci, nullptr, &a.tlas_) != VK_SUCCESS) {
            vkDestroyBuffer(dev, tscratch, nullptr);
            vkFreeMemory(dev, tscratch_mem, nullptr);
            return std::unexpected("rtx_accel: create groups TLAS failed");
        }

        bgi.dstAccelerationStructure  = a.tlas_;
        bgi.scratchData.deviceAddress = buffer_address(ctx, tscratch);
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = n;
        const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
        auto r = one_shot(ctx, [&](VkCommandBuffer cmd) {
            cmd_build(cmd, 1, &bgi, &pr);
        });
        vkDestroyBuffer(dev, tscratch, nullptr);
        vkFreeMemory(dev, tscratch_mem, nullptr);
        if (!r) return std::unexpected(r.error());

        VkAccelerationStructureDeviceAddressInfoKHR tai{};
        tai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        tai.accelerationStructure = a.tlas_;
        a.tlas_addr_ = as_addr(dev, &tai);
    }

    return a;
}

}  // namespace frep::gpu
