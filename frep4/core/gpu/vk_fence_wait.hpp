#pragma once
// core/gpu/vk_fence_wait.hpp
//
// How a frame waits for its fence decides how long the frame is. The plain
// vkWaitForFences puts the thread to sleep in the kernel; the core it was on
// drops into an idle state (C6 here, 133 µs to wake) and, because the thread
// is then idle for most of every frame, the frequency governor also lowers
// that core's clock, so the host work around the GPU runs slow as well. A
// 512² dispatch on this machine is 0.1–0.4 ms of GPU time, and the same frame
// measured 1.1–1.5 ms with the blocking wait (core at 1.2–2.0 GHz) against
// 0.55 ms spinning (core at 3.0 GHz), steady over a five-second loop. The
// CUDA path never had the problem: its synchronous copy spins.
//
// wait_fence spins on vkGetFenceStatus for a budget of a few milliseconds and
// only then falls back to the blocking wait, so a short frame never pays the
// wake-up and a long one does not burn a core for its whole duration. The
// budget can follow the previous frame (spin_budget_ms): a frame that took
// longer than the cap is expected to again, and blocks at once.
// FREP_VK_WAIT=block restores the plain blocking wait for an A/B.
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace frep::gpu {

inline constexpr double kFenceSpinCapMs = 4.0;

// Spin budget for the next frame from the last frame's submit-to-signal time:
// twice that, but never under 0.25 ms (the first frames and noise) and never
// over the cap; a frame that exceeded the cap gets no spin at all.
inline double spin_budget_ms(double last_wait_ms) {
    if (last_wait_ms > kFenceSpinCapMs) return 0.0;
    double b = 2.0 * last_wait_ms;
    if (b < 0.25) b = 0.25;
    if (b > kFenceSpinCapMs) b = kFenceSpinCapMs;
    return b;
}

inline VkResult wait_fence(VkDevice dev, VkFence fence, double spin_ms = kFenceSpinCapMs) {
    static const bool spin = [] {
        const char* s = std::getenv("FREP_VK_WAIT");
        return !(s && std::strcmp(s, "block") == 0);
    }();
    if (spin && spin_ms > 0) {
        auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            VkResult r = vkGetFenceStatus(dev, fence);
            if (r != VK_NOT_READY) return r;      // signalled, or a device error
            if (std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() > spin_ms)
                break;
        }
    }
    return vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ULL);
}

} // namespace frep::gpu
