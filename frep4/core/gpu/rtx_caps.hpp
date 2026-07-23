// core/gpu/rtx_caps.hpp
//
// Ray-tracing capability detection for the GpuRtx path.
//
// The RTX path renders the *same* F-Rep scene as the other three: a hardware
// BVH does broad-phase (which object's AABB does this ray enter?), and a
// custom intersection shader sphere-traces the exact SDF inside — so the
// surface is still implicit, and parity with cpu_ir/gpu_ir/gpu_glsl holds.
//
// This header only answers "can this machine do hardware ray tracing?" by
// probing for the VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure
// device extensions. It deliberately does NOT pull in any ray-tracing SDK
// headers: detection is done by extension-name string match against the
// device's reported extensions, so the code compiles on a stock Vulkan SDK
// (and in the sandbox, which has only llvmpipe) without the RT headers.
//
// Hardware reality: RT cores start at Turing (RTX 20xx). A Pascal card
// (GTX 1050 Ti) reports neither extension, so hardware RT is unavailable and
// the executor uses the software BVH-walk fallback instead (see
// rtx_executor.hpp). The RTX 2080 reports both and runs the real RT pipeline.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace frep::gpu {

// What kind of ray-tracing backend a machine can offer for the GpuRtx path.
enum class RtxBackend {
    None,        // no Vulkan device at all
    Hardware,    // VK_KHR_ray_tracing_pipeline present → real RT cores
    Software,    // Vulkan present but no RT extension → software BVH fallback
};

struct RtxCaps {
    RtxBackend  backend = RtxBackend::None;
    std::string device_name;                 // GPU reported name, if any
    bool        has_ray_tracing_pipeline = false;
    bool        has_acceleration_structure = false;

    bool hardware() const { return backend == RtxBackend::Hardware; }
    // A Vulkan RT pipeline can run (hardware RT cores, OR a CPU emulation like
    // llvmpipe that advertises the extensions). Either way the same RT shader
    // code path is exercised — useful for validating the pipeline without RT
    // cores, just slower.
    bool vulkan_rt_pipeline() const {
        return has_ray_tracing_pipeline && has_acceleration_structure;
    }
    // The path can run (one way or another) as long as there is a device:
    // hardware RT, software-emulated RT pipeline, or our own BVH-walk fallback.
    bool usable()   const { return backend != RtxBackend::None; }

    // Human-readable one-liner for logs / warnings.
    std::string describe() const;
};

// Probe the first Vulkan physical device for ray-tracing support. Cheap;
// creates a throwaway instance, enumerates device extensions, destroys it.
// Safe to call when no Vulkan loader/device is present (returns backend=None).
RtxCaps detect_rtx_caps();

// Same probe but fills `log` with a human-readable trace of what was found
// (device count, names, whether each RT extension is present). Use this when
// a machine that should have RT cores reports otherwise, to see where the
// detection diverges (driver/loader, extension enumeration, etc.).
RtxCaps detect_rtx_caps_verbose(std::string& log);

}  // namespace frep::gpu
