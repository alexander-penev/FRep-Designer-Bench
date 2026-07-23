// tools/rtx_probe.cpp
//
// Standalone diagnostic for the GpuRtx path. Prints exactly what Vulkan
// reports about ray-tracing support on this machine: device count, each
// device's name, and whether it exposes VK_KHR_ray_tracing_pipeline and
// VK_KHR_acceleration_structure. Use it when a card that should have RT cores
// (e.g. RTX 2080) is not being detected as hardware, to see where detection
// diverges (loader/driver, device ordering, extension enumeration).

#include "core/gpu/rtx_caps.hpp"
#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"
#include "core/frep/node.hpp"

#include <cstdio>
#include <string>

int main() {
    std::string log;
    auto caps = frep::gpu::detect_rtx_caps_verbose(log);

    std::printf("=== Vulkan RT probe ===\n%s\n", log.c_str());
    std::printf("result: %s\n", caps.describe().c_str());
    std::printf("  backend          : %s\n",
                caps.backend == frep::gpu::RtxBackend::Hardware ? "Hardware" :
                caps.backend == frep::gpu::RtxBackend::Software ? "Software" : "None");
    std::printf("  usable           : %s\n", caps.usable() ? "yes" : "no");
    std::printf("  using hardware RT: %s\n", caps.hardware() ? "yes" : "no");
    std::printf("  vulkan_rt_pipeline: %s\n",
                caps.vulkan_rt_pipeline() ? "yes" : "no");

    // Actually try to create the RT device (device + extensions +
    // feature chain). This is the real test that the RT pipeline can stand up
    // on this machine — detection only reads extension lists.
    std::printf("\n=== RT device creation ===\n");
    auto ctx = frep::gpu::RtxCtx::create();
    if (ctx) {
        std::printf("  RT device created OK on \"%s\" (%s)\n",
                    ctx->device_name().c_str(),
                    ctx->is_software() ? "software/CPU" : "hardware");

        // Build a BLAS+TLAS over a unit-cube scene box.
        std::printf("\n=== acceleration structure build ===\n");
        frep::FRepNode::AABB box{-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
        auto accel = frep::gpu::RtAccel::build_whole_scene(*ctx, box);
        if (accel) {
            std::printf("  BLAS+TLAS built OK; TLAS device address = 0x%llx\n",
                        (unsigned long long)accel->tlas_device_address());
        } else {
            std::printf("  AS build failed: %s\n", accel.error().c_str());
        }
    } else {
        std::printf("  RT device creation failed: %s\n", ctx.error().c_str());
    }
    return 0;
}
