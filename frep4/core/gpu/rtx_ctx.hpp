// core/gpu/rtx_ctx.hpp
//
// Vulkan ray-tracing context for the GpuRtx path.
//
// Parallel to VulkanCtx (the compute-pipeline context), but builds a device
// with the ray-tracing extensions enabled and loads the KHR ray-tracing entry
// points (which are not in the core Vulkan loader and must be fetched with
// vkGetDeviceProcAddr). This stands up the device + entry points;
// acceleration structures, RT pipeline, SBT and dispatch arrive in 1b–1d.
//
// Works on real RT hardware (RTX 2080) and on a CPU emulation that advertises
// the extensions (llvmpipe), so the whole pipeline can be developed and tested
// without RT cores — just slower.

#pragma once

#include <cstdint>
#include <expected>
#include <string>

// Forward-declare Vulkan handles we expose without pulling <vulkan/vulkan.h>
// into every includer. The .cpp includes the real header.
struct VkInstance_T;       using VkInstance       = VkInstance_T*;
struct VkPhysicalDevice_T; using VkPhysicalDevice = VkPhysicalDevice_T*;
struct VkDevice_T;         using VkDevice         = VkDevice_T*;
struct VkQueue_T;          using VkQueue          = VkQueue_T*;

namespace frep::gpu {

class RtxCtx {
public:
    RtxCtx() = default;
    ~RtxCtx();
    RtxCtx(const RtxCtx&)            = delete;
    RtxCtx& operator=(const RtxCtx&) = delete;

    // Build a ray-tracing device. Picks a physical device that advertises
    // VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure (preferring
    // a non-CPU one, but accepting a software rasterizer like llvmpipe),
    // creates the logical device with those extensions + their dependencies
    // enabled, and loads the KHR RT entry points.
    //
    // Returns an error string if no RT-capable device exists or device
    // creation fails. On success the context owns the device until destroyed.
    static std::expected<RtxCtx, std::string> create();

    // Movable (owns Vulkan handles).
    RtxCtx(RtxCtx&& o) noexcept;
    RtxCtx& operator=(RtxCtx&& o) noexcept;

    bool        is_software() const { return is_software_; }
    const std::string& device_name() const { return device_name_; }

    // True when VK_KHR_ray_query is enabled on this device (the rayQueryEXT
    // inline-in-compute path). Additive to the RT-pipeline path: enabled only
    // when the device advertises the extension + the rayQuery feature, so a
    // device without it still creates fine and only the SBT path is available.
    bool        has_ray_query() const { return has_ray_query_; }

    // Raw handles for the later phases (1b builds AS on these).
    VkDevice         device()          const { return device_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkQueue          queue()           const { return queue_; }
    std::uint32_t    queue_family()    const { return queue_family_; }

    // KHR ray-tracing entry points, loaded with vkGetDeviceProcAddr in
    // create() (they are not in the core loader). Opaque void* here so the
    // header stays Vulkan-free; rtx_ctx.cpp and later AS/pipeline code cast
    // them to the right PFN_ type. All non-null on a successful create().
    struct RtApi {
        void* createAccelerationStructure   = nullptr;
        void* destroyAccelerationStructure  = nullptr;
        void* getAccelerationStructureBuildSizes = nullptr;
        void* cmdBuildAccelerationStructures = nullptr;
        void* getAccelerationStructureDeviceAddress = nullptr;
        void* getBufferDeviceAddress        = nullptr;
        void* createRayTracingPipelines     = nullptr;
        void* getRayTracingShaderGroupHandles = nullptr;
        void* cmdTraceRays                  = nullptr;
        bool  complete() const {
            return createAccelerationStructure && destroyAccelerationStructure &&
                   getAccelerationStructureBuildSizes &&
                   cmdBuildAccelerationStructures &&
                   getAccelerationStructureDeviceAddress &&
                   getBufferDeviceAddress && createRayTracingPipelines &&
                   getRayTracingShaderGroupHandles && cmdTraceRays;
        }
    };
    const RtApi& api() const { return api_; }

private:
    void destroy();

    VkInstance       instance_        = nullptr;
    VkPhysicalDevice physical_device_ = nullptr;
    VkDevice         device_          = nullptr;
    VkQueue          queue_           = nullptr;
    std::uint32_t    queue_family_    = 0;
    bool             is_software_     = false;
    bool             has_ray_query_   = false;
    std::string      device_name_;
    RtApi            api_;
};

}  // namespace frep::gpu
