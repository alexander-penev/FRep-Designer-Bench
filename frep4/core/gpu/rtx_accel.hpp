// core/gpu/rtx_accel.hpp
//
// Acceleration-structure build for the GpuRtx path.
//
// The scene is split into several AABBs (one per CSG-independent group); each
// becomes a BLAS (an "AABBs" geometry, the kind whose hits are resolved by a
// custom intersection shader — exactly what's needed for SDF sphere tracing),
// and a TLAS references them. The intersection shader sphere-traces the group's
// scene_sdf inside its box, so parity with the other paths holds while the RT
// cores cull groups a ray misses. A whole-scene single-box build is also
// available (build_whole_scene) for the trivial one-group case.

#pragma once

#include "core/gpu/rtx_ctx.hpp"
#include "core/frep/node.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

// Vulkan handles kept opaque in the header.
struct VkBuffer_T;              using VkBuffer              = VkBuffer_T*;
struct VkDeviceMemory_T;        using VkDeviceMemory        = VkDeviceMemory_T*;
struct VkAccelerationStructureKHR_T;
using VkAccelerationStructureKHR = VkAccelerationStructureKHR_T*;

namespace frep::gpu {

// One axis-aligned box fed to the BLAS (min/max corners in world space).
struct RtAabb {
    float min[3];
    float max[3];
};

// Owns the BLAS + TLAS and their backing buffers. Built once per scene; the
// The RT pipeline binds the TLAS as a descriptor and traces against it.
class RtAccel {
public:
    RtAccel() = default;
    ~RtAccel();
    RtAccel(const RtAccel&)            = delete;
    RtAccel& operator=(const RtAccel&) = delete;
    RtAccel(RtAccel&&) noexcept;
    RtAccel& operator=(RtAccel&&) noexcept;

    // Build a BLAS from the given AABBs (≥1) and a single-instance TLAS over
    // it. `ctx` must outlive the RtAccel. Returns an error string on any
    // Vulkan failure.
    static std::expected<RtAccel, std::string>
    build(const RtxCtx& ctx, const std::vector<RtAabb>& boxes);

    // Convenience: one box from a scene-space AABB.
    static std::expected<RtAccel, std::string>
    build_whole_scene(const RtxCtx& ctx, const FRepNode::AABB& scene_box);

    // Build one BLAS per group box and a TLAS with one instance per
    // group, where instance i gets instanceShaderBindingTableRecordOffset = i,
    // so the SBT can give each group its own intersection shader. `boxes[i]` is
    // group i's world-space AABB. This is what lets the RT cores cull groups a
    // ray misses. The single-instance build() above is the n=1 special case.
    static std::expected<RtAccel, std::string>
    build_groups(const RtxCtx& ctx, const std::vector<RtAabb>& group_boxes);

    // Number of TLAS instances (= groups for build_groups, 1 otherwise).
    std::uint32_t instance_count() const { return inst_count_; }

    VkAccelerationStructureKHR tlas() const { return tlas_; }
    std::uint64_t tlas_device_address() const { return tlas_addr_; }

private:
    void destroy();

    const RtxCtx*   ctx_  = nullptr;

    VkBuffer        aabb_buf_     = nullptr;  VkDeviceMemory aabb_mem_  = nullptr;
    VkBuffer        blas_buf_     = nullptr;  VkDeviceMemory blas_mem_  = nullptr;
    VkBuffer        tlas_buf_     = nullptr;  VkDeviceMemory tlas_mem_  = nullptr;
    VkBuffer        inst_buf_     = nullptr;  VkDeviceMemory inst_mem_  = nullptr;
    VkBuffer        scratch_buf_  = nullptr;  VkDeviceMemory scratch_mem_ = nullptr;

    VkAccelerationStructureKHR blas_ = nullptr;
    VkAccelerationStructureKHR tlas_ = nullptr;
    std::uint64_t   tlas_addr_ = 0;
    std::uint32_t   inst_count_ = 1;

    // Multi-BLAS: one entry per group. Parallel to the single-BLAS
    // members above; build_groups fills these instead of blas_/aabb_buf_.
    std::vector<VkBuffer>        g_aabb_buf_, g_blas_buf_, g_scratch_buf_;
    std::vector<VkDeviceMemory>  g_aabb_mem_, g_blas_mem_, g_scratch_mem_;
    std::vector<VkAccelerationStructureKHR> g_blas_;
};

}  // namespace frep::gpu
