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

// One axis-aligned box fed to the BLAS (min/max corners in world space, or in
// the prototype's object space when the instance carries a transform).
struct RtAabb {
    float min[3];
    float max[3];
};

// One TLAS placement: which BLAS to reference and where to put it. `m` is the
// object->world affine map in VkTransformMatrixKHR's row-major 3x4 layout
// ([R|t] per row), so the traversal unit transforms the ray into the
// prototype's object space before handing it to the intersection shader. The
// BLAS index doubles as the SBT hit-record offset, so all placements of one
// prototype run the same intersection shader.
struct RtInstanceXform {
    std::uint32_t blas = 0;
    float m[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
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

    // Build one BLAS per *prototype* box and a TLAS whose instances place those
    // prototypes with arbitrary affine transforms. This is build_groups
    // generalised: there, boxes and instances are in bijection and every
    // transform is the identity; here many placements may share one BLAS (and
    // therefore one intersection shader and one SBT record), so a scene of a
    // thousand copies of one shape costs one shader instead of a thousand.
    //
    // `proto_boxes[i]` is prototype i's bounds in *its own object space*, and
    // `placements[j].m` maps that space to world.
    static std::expected<RtAccel, std::string>
    build_instanced(const RtxCtx& ctx, const std::vector<RtAabb>& proto_boxes,
                    const std::vector<RtInstanceXform>& placements);

    // Number of TLAS instances (= groups for build_groups, 1 otherwise).
    std::uint32_t instance_count() const { return inst_count_; }

    // The single-BLAS build's box list, as the shader sees it: `aabb_count()`
    // boxes of six floats {min.xyz, max.xyz}, in a device-local storage
    // buffer. With more than one box the field was split into shell cells
    // (rtx_cells.hpp) and the intersection shader reads its cell from here by
    // gl_PrimitiveID; the group and instanced builds keep their boxes per
    // BLAS and report zero.
    VkBuffer      aabb_buffer() const { return aabb_buf_; }
    std::uint32_t aabb_count()  const { return aabb_count_; }

    // Device memory the acceleration structure occupies, in bytes. Reported
    // because memory is one of the axes the backends are compared on, and it is
    // where instancing shows its clearest win: N copies of one shape need one
    // bottom-level structure, not N. Broken out because the parts scale
    // differently -- `blas` is per PROTOTYPE while `instances` is per placement,
    // so the instanced build trades a large term for a small one.
    struct Memory {
        std::uint64_t blas = 0;       // bottom-level structures + their AABB inputs
        std::uint64_t tlas = 0;       // top-level structure
        std::uint64_t instances = 0;  // the TLAS instance array
        std::uint64_t scratch = 0;    // build scratch (freeable after the build)
        std::uint64_t total() const { return blas + tlas + instances + scratch; }
    };
    const Memory& memory() const { return mem_; }
    std::uint32_t blas_count() const {
        return g_blas_.empty() ? (blas_ ? 1u : 0u) : (std::uint32_t)g_blas_.size();
    }

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
    std::uint32_t   aabb_count_ = 0;
    Memory          mem_{};

    // Multi-BLAS: one entry per group. Parallel to the single-BLAS
    // members above; build_groups fills these instead of blas_/aabb_buf_.
    std::vector<VkBuffer>        g_aabb_buf_, g_blas_buf_, g_scratch_buf_;
    std::vector<VkDeviceMemory>  g_aabb_mem_, g_blas_mem_, g_scratch_mem_;
    std::vector<VkAccelerationStructureKHR> g_blas_;
};

}  // namespace frep::gpu
