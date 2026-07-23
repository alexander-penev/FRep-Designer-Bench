// tools/diag_handwritten.cpp — diagnose the legacy sphere_trace.comp render.
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace frep;

static std::string find_spv() {
    namespace fs = std::filesystem;
    for (const char* p : {"build/shaders/sphere_trace.spv",
                          "shaders/sphere_trace.spv",
                          "../build/shaders/sphere_trace.spv"})
        if (fs::exists(p)) return p;
    return {};
}

int main() {
    if (!gpu::VulkanCtx::available()) { std::printf("no Vulkan\n"); return 0; }
    std::string spv = find_spv();
    if (spv.empty()) { std::printf("spv not found\n"); return 1; }
    std::printf("spv: %s\n", spv.c_str());

    float cam[3] = {0, 1, 4}, tgt[3] = {0, 0, 0}, light[3] = {4, 6, 4};
    int W = 80, H = 60;
    gpu::ShaderPush p = gpu::build_push_simple(cam, tgt, light, W, H);
    std::printf("push: cam_pos=(%.2f,%.2f,%.2f) fov_scale=%.3f sphere_radius=%.3f\n",
                p.cam_pos[0], p.cam_pos[1], p.cam_pos[2], p.fov_scale, p.sphere_radius);
    std::printf("      cam_fwd=(%.3f,%.3f,%.3f)\n", p.cam_fwd[0], p.cam_fwd[1], p.cam_fwd[2]);
    std::printf("      cam_right=(%.3f,%.3f,%.3f)\n", p.cam_right[0], p.cam_right[1], p.cam_right[2]);
    std::printf("      cam_up=(%.3f,%.3f,%.3f)\n", p.cam_up[0], p.cam_up[1], p.cam_up[2]);
    std::printf("      light0=(%.2f,%.2f,%.2f) w=%.2f  light_count=%.1f\n",
                p.lights[0][0], p.lights[0][1], p.lights[0][2], p.lights[0][3], p.light_count);
    std::printf("      sizeof(ShaderPush)=%zu bytes\n", sizeof(gpu::ShaderPush));

    auto ctx = gpu::VulkanCtx::create(spv);
    if (!ctx) { std::printf("create failed: %s\n", ctx.error().c_str()); return 1; }
    std::printf("device: %s\n", (**ctx).stats().device_name.c_str());

    std::vector<std::uint8_t> px;
    auto rr = (**ctx).render(p, px);
    if (!rr) { std::printf("render failed: %s\n", rr.error().c_str()); return 1; }

    // Histogram of the red channel + sample a few pixels.
    int rl = 0, rm = 0, rh = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i * 4 + 0];
        if (r < 60) ++rl; else if (r > 180) ++rh; else ++rm;
    }
    std::printf("red histogram: low(<60)=%d  mid=%d  high(>180)=%d  (of %d)\n", rl, rm, rh, W * H);
    auto dump = [&](const char* tag, int x, int y) {
        std::size_t i = ((std::size_t)y * W + x) * 4;
        std::printf("  %-8s px(%d,%d) = rgba(%d,%d,%d,%d)\n", tag, x, y,
                    px[i], px[i + 1], px[i + 2], px[i + 3]);
    };
    dump("center", W / 2, H / 2);
    dump("top",    W / 2, 2);
    dump("bottom", W / 2, H - 3);
    return 0;
}
