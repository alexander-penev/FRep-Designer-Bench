// core/gpu/rtx_caps.cpp — see rtx_caps.hpp.

#include "core/gpu/rtx_caps.hpp"

#include <vulkan/vulkan.h>

#include <cstring>

namespace frep::gpu {

// Extension names as plain strings so we don't depend on the RT SDK headers
// defining the VK_KHR_*_EXTENSION_NAME macros.
static constexpr const char* kRayTracingPipeline   = "VK_KHR_ray_tracing_pipeline";
static constexpr const char* kAccelerationStructure = "VK_KHR_acceleration_structure";

std::string RtxCaps::describe() const {
    switch (backend) {
        case RtxBackend::None:
            return "no Vulkan device — GpuRtx unavailable";
        case RtxBackend::Hardware:
            return "hardware ray tracing on \"" + device_name +
                   "\" (RT cores, VK_KHR_ray_tracing_pipeline)";
        case RtxBackend::Software:
            if (has_ray_tracing_pipeline && has_acceleration_structure)
                return "software ray tracing on \"" + device_name +
                       "\" (Vulkan RT pipeline emulated on CPU, e.g. llvmpipe "
                       "— no RT cores)";
            return "no RT cores on \"" + device_name +
                   "\" — GpuRtx will use the software BVH fallback";
    }
    return "unknown";
}

static RtxCaps detect_impl(std::string* log) {
    RtxCaps caps;
    auto note = [&](const std::string& s) { if (log) { *log += s; *log += '\n'; } };

    VkApplicationInfo ai{};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "frep4-rtx-probe";
    ai.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult ir = vkCreateInstance(&ici, nullptr, &inst);
    if (ir != VK_SUCCESS) {
        note("vkCreateInstance failed (code " + std::to_string((int)ir) +
             ") — no Vulkan loader/driver");
        return caps;
    }

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    note("physical devices: " + std::to_string(n));
    if (n == 0) { vkDestroyInstance(inst, nullptr); return caps; }

    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(inst, &n, phys.data());

    // Scan every device; prefer one with hardware RT. A multi-GPU host may
    // list the RT card second (e.g. iGPU first), and a headless box may list a
    // software rasterizer alongside the real card.
    // Selection precedence:
    //   1. a real GPU (discrete/integrated/virtual) advertising both RT
    //      extensions  → Hardware (real RT cores)
    //   2. otherwise, any device advertising RT (e.g. a CPU rasterizer like
    //      llvmpipe/lavapipe that emulates the extensions) → Software, since
    //      the RT pipeline runs but on the CPU, not RT cores
    //   3. otherwise, the first device at all → Software fallback (our own
    //      BVH walk, no Vulkan RT pipeline)
    // A device advertising RT does not by itself mean RT cores: software
    // rasterizers report VK_PHYSICAL_DEVICE_TYPE_CPU and emulate everything.
    int best = -1;
    bool have_hw = false;
    for (std::uint32_t d = 0; d < n; ++d) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys[d], &props);

        std::uint32_t ext_n = 0;
        vkEnumerateDeviceExtensionProperties(phys[d], nullptr, &ext_n, nullptr);
        std::vector<VkExtensionProperties> exts(ext_n);
        vkEnumerateDeviceExtensionProperties(phys[d], nullptr, &ext_n, exts.data());

        bool rtp = false, accel = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, kRayTracingPipeline) == 0)   rtp = true;
            if (std::strcmp(e.extensionName, kAccelerationStructure) == 0) accel = true;
        }
        bool is_cpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);
        bool rt_ext = rtp && accel;

        note("  device[" + std::to_string(d) + "] \"" +
             std::string(props.deviceName) + "\"  type=" +
             (is_cpu ? "CPU(software)" :
              props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete" :
              props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated" :
              props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "virtual" : "other") +
             "  exts=" + std::to_string(ext_n) +
             "  ray_tracing_pipeline=" + (rtp ? "yes" : "no") +
             "  acceleration_structure=" + (accel ? "yes" : "no"));

        if (rt_ext && !is_cpu) {
            // Real GPU with RT cores — best case, take it and stop.
            caps.device_name                = props.deviceName;
            caps.has_ray_tracing_pipeline   = true;
            caps.has_acceleration_structure = true;
            caps.backend                    = RtxBackend::Hardware;
            have_hw = true;
            best = (int)d;
            break;
        }
        if (best < 0 || (rt_ext && !have_hw)) {
            // First device seen, or a software RT-capable device that's better
            // than a plain fallback (it can at least run the Vulkan RT
            // pipeline, just on the CPU).
            caps.device_name               = props.deviceName;
            caps.has_ray_tracing_pipeline  = rtp;
            caps.has_acceleration_structure = accel;
            caps.backend                   = RtxBackend::Software;
            best = (int)d;
        }
    }

    vkDestroyInstance(inst, nullptr);
    return caps;
}

RtxCaps detect_rtx_caps() {
    return detect_impl(nullptr);
}

RtxCaps detect_rtx_caps_verbose(std::string& log) {
    return detect_impl(&log);
}

}  // namespace frep::gpu
