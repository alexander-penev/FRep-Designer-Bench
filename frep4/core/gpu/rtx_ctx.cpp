// core/gpu/rtx_ctx.cpp — see rtx_ctx.hpp. RT device + entry points.

#include "core/gpu/rtx_ctx.hpp"

#include <vulkan/vulkan.h>

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace frep::gpu {

namespace {

constexpr const char* kRayTracingPipeline    = "VK_KHR_ray_tracing_pipeline";
constexpr const char* kAccelerationStructure = "VK_KHR_acceleration_structure";
constexpr const char* kDeviceAddress         = "VK_KHR_buffer_device_address";
constexpr const char* kDeferredHostOps       = "VK_KHR_deferred_host_operations";
constexpr const char* kRayQuery              = "VK_KHR_ray_query";
// VK_KHR_acceleration_structure also requires VK_EXT_descriptor_indexing and
// VK_KHR_spirv_1_4 in older setups; on Vulkan 1.2 these are core, so we target
// apiVersion 1.2 and only request the four extensions above.

bool device_has(VkPhysicalDevice pd, const char* name) {
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    for (const auto& x : e)
        if (std::strcmp(x.extensionName, name) == 0) return true;
    return false;
}

bool rt_capable(VkPhysicalDevice pd) {
    return device_has(pd, kRayTracingPipeline) &&
           device_has(pd, kAccelerationStructure);
}

}  // namespace

RtxCtx::~RtxCtx() { destroy(); }

RtxCtx::RtxCtx(RtxCtx&& o) noexcept { *this = std::move(o); }

RtxCtx& RtxCtx::operator=(RtxCtx&& o) noexcept {
    if (this != &o) {
        destroy();
        instance_        = o.instance_;
        physical_device_ = o.physical_device_;
        device_          = o.device_;
        queue_           = o.queue_;
        queue_family_    = o.queue_family_;
        is_software_     = o.is_software_;
        has_ray_query_   = o.has_ray_query_;
        device_name_     = std::move(o.device_name_);
        api_             = o.api_;
        o.instance_ = nullptr; o.device_ = nullptr; o.physical_device_ = nullptr;
        o.queue_ = nullptr;
    }
    return *this;
}

void RtxCtx::destroy() {
    if (device_)   { vkDestroyDevice(device_, nullptr);   device_ = nullptr; }
    if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = nullptr; }
}

std::expected<RtxCtx, std::string> RtxCtx::create() {
    RtxCtx ctx;

    VkApplicationInfo ai{};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "frep4-rtx";
    ai.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;

    // Optional validation layer for diagnosing RT issues: set FREP_RTX_VALIDATE=1.
    const char* val_layer = "VK_LAYER_KHRONOS_validation";
    if (const char* v = std::getenv("FREP_RTX_VALIDATE"); v && v[0] == '1') {
        ici.enabledLayerCount   = 1;
        ici.ppEnabledLayerNames = &val_layer;
    }
    if (vkCreateInstance(&ici, nullptr, &ctx.instance_) != VK_SUCCESS)
        return std::unexpected("rtx: vkCreateInstance failed (no Vulkan driver)");

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(ctx.instance_, &n, nullptr);
    if (n == 0) return std::unexpected("rtx: no Vulkan physical devices");
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(ctx.instance_, &n, phys.data());

    // Prefer a non-CPU RT-capable device; accept a CPU one (llvmpipe) as
    // fallback so the pipeline can run without RT cores.
    VkPhysicalDevice chosen = nullptr;
    bool chosen_is_cpu = false;
    for (auto pd : phys) {
        if (!rt_capable(pd)) continue;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        bool is_cpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);
        if (!chosen || (chosen_is_cpu && !is_cpu)) {
            chosen = pd; chosen_is_cpu = is_cpu;
            if (!is_cpu) break;  // hardware RT — take it
        }
    }
    if (!chosen)
        return std::unexpected("rtx: no device advertises the RT extensions");

    ctx.physical_device_ = chosen;
    ctx.is_software_     = chosen_is_cpu;
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(chosen, &props);
        ctx.device_name_ = props.deviceName;
    }

    // Queue family: first with compute (RT dispatch runs on compute/graphics).
    std::uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(chosen, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(chosen, &qn, qf.data());
    bool found_q = false;
    for (std::uint32_t i = 0; i < qn; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queue_family_ = i; found_q = true; break;
        }
    if (!found_q) return std::unexpected("rtx: no compute-capable queue family");

    // Enable RT feature chain. Query what the device actually supports first
    // (via vkGetPhysicalDeviceFeatures2) and only request features it reports —
    // a software emulation may advertise the extensions yet not every feature
    // bit, and requesting an unsupported feature fails device creation.
    // VK_KHR_ray_query is optional: enabled only when the device advertises the
    // extension *and* reports rayQuery=true. It lets a plain compute shader run
    // rayQueryEXT inline (hardware BVH broad phase, march in-shader at full
    // occupancy) instead of the SBT ray-tracing pipeline's intersection shader.
    const bool rq_ext = device_has(chosen, kRayQuery);

    VkPhysicalDeviceRayQueryFeaturesKHR rqf_q{};
    rqf_q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf_q{};
    rtf_q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    if (rq_ext) rtf_q.pNext = &rqf_q;   // query rayQuery only if the ext exists
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf_q{};
    asf_q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asf_q.pNext = &rtf_q;
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_q{};
    bda_q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bda_q.pNext = &asf_q;
    VkPhysicalDeviceFeatures2 feat_q{};
    feat_q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat_q.pNext = &bda_q;
    vkGetPhysicalDeviceFeatures2(chosen, &feat_q);

    if (!rtf_q.rayTracingPipeline)
        return std::unexpected("rtx: device advertises the extension but "
                               "rayTracingPipeline feature is false");
    if (!asf_q.accelerationStructure)
        return std::unexpected("rtx: accelerationStructure feature is false");
    if (!bda_q.bufferDeviceAddress)
        return std::unexpected("rtx: bufferDeviceAddress feature is false");

    VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bda.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{};
    asf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asf.accelerationStructure = VK_TRUE;
    asf.pNext = &bda;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf{};
    rtf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtf.rayTracingPipeline = VK_TRUE;
    rtf.pNext = &asf;

    // Only enable ray query when the device actually supports the feature; then
    // splice its feature struct into the chain head and add the extension.
    ctx.has_ray_query_ = rq_ext && rqf_q.rayQuery;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf{};
    rqf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqf.rayQuery = VK_TRUE;

    std::vector<const char*> exts = {
        kRayTracingPipeline, kAccelerationStructure,
        kDeviceAddress, kDeferredHostOps,
    };
    const void* chain_head = &rtf;
    if (ctx.has_ray_query_) {
        exts.push_back(kRayQuery);
        rqf.pNext  = &rtf;   // prepend ray-query features to the chain
        chain_head = &rqf;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqi{};
    dqi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqi.queueFamilyIndex = ctx.queue_family_;
    dqi.queueCount       = 1;
    dqi.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = chain_head;  // feature chain (ray-query head if on)
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &dqi;
    dci.enabledExtensionCount   = (std::uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();

    VkResult dr = vkCreateDevice(chosen, &dci, nullptr, &ctx.device_);
    if (dr != VK_SUCCESS)
        return std::unexpected("rtx: vkCreateDevice failed (code " +
                               std::to_string((int)dr) +
                               ") — RT feature chain may be unsupported");

    vkGetDeviceQueue(ctx.device_, ctx.queue_family_, 0, &ctx.queue_);

    // Load the KHR ray-tracing entry points (not in the core loader).
    auto load = [&](const char* name) -> void* {
        return (void*)vkGetDeviceProcAddr(ctx.device_, name);
    };
    ctx.api_.createAccelerationStructure   = load("vkCreateAccelerationStructureKHR");
    ctx.api_.destroyAccelerationStructure  = load("vkDestroyAccelerationStructureKHR");
    ctx.api_.getAccelerationStructureBuildSizes = load("vkGetAccelerationStructureBuildSizesKHR");
    ctx.api_.cmdBuildAccelerationStructures = load("vkCmdBuildAccelerationStructuresKHR");
    ctx.api_.getAccelerationStructureDeviceAddress = load("vkGetAccelerationStructureDeviceAddressKHR");
    ctx.api_.getBufferDeviceAddress        = load("vkGetBufferDeviceAddressKHR");
    if (!ctx.api_.getBufferDeviceAddress)  // core in 1.2, try the unsuffixed name
        ctx.api_.getBufferDeviceAddress    = load("vkGetBufferDeviceAddress");
    ctx.api_.createRayTracingPipelines     = load("vkCreateRayTracingPipelinesKHR");
    ctx.api_.getRayTracingShaderGroupHandles = load("vkGetRayTracingShaderGroupHandlesKHR");
    ctx.api_.cmdTraceRays                  = load("vkCmdTraceRaysKHR");
    if (!ctx.api_.complete())
        return std::unexpected("rtx: failed to load one or more KHR RT entry "
                               "points (driver exposes the extension but not "
                               "all functions)");

    return ctx;
}

}  // namespace frep::gpu
