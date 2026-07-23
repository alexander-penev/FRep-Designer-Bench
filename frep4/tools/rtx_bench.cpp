// tools/rtx_bench.cpp — GpuRtx broad-phase scaling benchmark.
//
// Sweeps a sphere-grid scene over increasing object counts and times two
// renderers on each:
//   compute (cpu_ir): O(N) flat-union scene_sdf per march step
//   gpu_rtx multi-BLAS: one BLAS per CSG group; RT cores cull groups a ray
//     misses, so each intersection shader evaluates only one sphere
//
// The headline output is the scaling curve: as N grows, compute time should
// rise ~linearly while the RT broad-phase rises far slower. Reports the
// trace-only RT time (setup is amortizable) next to the CPU time.
//
// Usage: frep_rtx_bench [--width W] [--height H] [--counts 1,4,16,64]

#include "core/exec/bench_scenes.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/multipath.hpp"
#include "core/gpu/rtx_caps.hpp"
#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"
#include "core/gpu/rtx_shaders.hpp"
#include "core/gpu/rtx_pipeline.hpp"
#include "core/gpu/rtx_csg_groups.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/compiler/codegen.hpp"
#include "core/power/energy_meter.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <optional>

using namespace frep;

namespace {

// Compile one RT stage SPIR-V from GLSL text (returns words or empty on error).
std::vector<std::uint32_t> compile_stage(const std::string& src, const char* stage,
                                         std::string& err) {
    auto spv = gpu::compile_rt_stage_to_spv(src, stage);
    if (!spv) { err = spv.error(); return {}; }
    std::ifstream f(*spv, std::ios::binary | std::ios::ate);
    if (!f) { err = "cannot read " + *spv; return {}; }
    auto n = f.tellg(); f.seekg(0);
    std::vector<std::uint32_t> out((std::size_t)n / sizeof(std::uint32_t));
    f.read(reinterpret_cast<char*>(out.data()), n);
    std::remove(spv->c_str());
    return out;
}

FRepNode::Ptr scene_root(const SceneGraph& s) {
    for (auto& [id, o] : s.objects()) return o.geometry;
    return nullptr;
}

gpu::RtAabb to_aabb(const FRepNode::AABB& b, float m = 0.05f) {
    return gpu::RtAabb{ {b.min_x - m, b.min_y - m, b.min_z - m},
                        {b.max_x + m, b.max_y + m, b.max_z + m} };
}

}  // namespace

int main(int argc, char** argv) {
    int W = 256, H = 256;
    bool energy = false;
    std::vector<int> counts = {1, 4, 16, 64};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int d){ return i + 1 < argc ? std::atoi(argv[++i]) : d; };
        if (a == "--help" || a == "-h") {
            std::printf(
                "usage: %s [options]\n"
                "  --width N         frame width  (default 256)\n"
                "  --height N        frame height (default 256)\n"
                "  --counts A,B,...  CSG group counts to sweep (default 1,4,16,64)\n"
                "  --energy          measure CPU (RAPL) + GPU (NVML) energy, "
                "report Mpix/kWh\n",
                argv[0]);
            return 0;
        }
        else if (a == "--width") W = next(W);
        else if (a == "--height") H = next(H);
        else if (a == "--energy") energy = true;
        else if (a == "--counts") {
            counts.clear();
            std::string s = argv[++i]; size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                counts.push_back(std::atoi(s.substr(p, c - p).c_str()));
                if (c == std::string::npos) break; p = c + 1;
            }
        }
    }

    auto caps = gpu::detect_rtx_caps();
    std::printf("rtx_bench %dx%d  (%d Mpix/frame)\n[gpu_rtx backend] %s\n\n",
                W, H, (int)((double)W * H / 1e6 + 0.5), caps.describe().c_str());

    // Optional energy counters (RAPL for CPU, NVML for GPU). Probed once; if a
    // counter isn't available the column is simply omitted — never invented.
    std::unique_ptr<power::EnergyCounter> cpu_e, gpu_e;
    if (energy) {
        cpu_e = power::make_cpu_energy_counter();
        gpu_e = power::make_gpu_energy_counter(0);
        std::printf("  [energy] cpu: %s  gpu: %s\n",
                    cpu_e->available() ? cpu_e->domain().c_str()
                                       : "unavailable",
                    gpu_e->available() ? gpu_e->domain().c_str()
                                       : "unavailable");
        if (!cpu_e->available())
            std::printf("  [energy] cpu unavailable — for RAPL without root: "
                        "sysctl kernel.perf_event_paranoid=0 (perf PMU), or "
                        "chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj\n");
        std::printf("\n");
    }
    const bool cpu_e_ok = cpu_e && cpu_e->available();
    const bool gpu_e_ok = gpu_e && gpu_e->available();

    // Throughput, not "speedup": CPU and GPU cores aren't commensurable, so a
    // single ratio is misleading. The architectural result is that independent
    // devices ADD — running both concurrently gives cpu + rtx, which scaling a
    // single path can't reach (you can't turn 12 CPU cores into 512). Energy
    // (pix/kWh) and cost (pix/$) are separate axes for separate questions.
    if (energy)
        std::printf("  %5s  %7s  %12s  %12s  %12s  %14s  %14s\n",
                    "N", "groups", "cpu Mpix/s", "rtx Mpix/s", "sum Mpix/s",
                    "cpu Mpix/kWh", "rtx Mpix/kWh");
    else
        std::printf("  %5s  %7s  %12s  %12s  %12s\n",
                    "N", "groups", "cpu Mpix/s", "rtx Mpix/s", "sum Mpix/s");
    std::printf("  --------------------------------------------------------------"
                "%s\n", energy ? "--------------------------------" : "");

    const double mpix = (double)W * H / 1e6;

    TracerConfig cfg;
    exec::CpuIrExecutor cpu(SceneCodegen::SceneSdfMode::Inlined, cfg);

    for (int n : counts) {
        SceneGraph scene = bench::make_sphere_grid(n);
        auto root = scene_root(scene);
        auto groups = gpu::partition_csg_groups(root);

        // CPU reference: the executor's render_ms is the raymarch hot loop
        // (excludes JIT compile), the like-for-like against the RT trace_ms.
        // Both become throughput below. Energy is measured around the render
        // call (a warmup render first so JIT compile isn't charged to energy).
        cpu.render(scene, W, H, exec::Tile{0, 0, W, H});  // warmup (JIT)
        if (cpu_e_ok) cpu_e->begin();
        auto rc = cpu.render(scene, W, H, exec::Tile{0, 0, W, H});
        std::optional<double> cpu_j = cpu_e_ok ? cpu_e->end() : std::nullopt;
        if (!rc.ok) { std::printf("  %5d  cpu render failed: %s\n", n, rc.error.c_str()); continue; }
        double cpu_trace_ms = rc.render_ms;  // raymarch hot loop only

        // RT multi-BLAS path.
        double rtx_trace_ms = -1.0;
        double gpu_j = -1.0;
        std::string err;
        do {
            auto ctx = gpu::RtxCtx::create();
            if (!ctx) { err = ctx.error(); break; }

            // Per-group scenes + shaders.
            std::vector<SceneGraph> gscenes;
            for (auto& g : groups) { SceneGraph s; s.add_object(g.root); gscenes.push_back(std::move(s)); }
            auto sh = gpu::emit_rt_group_shaders(scene, gscenes, cfg);
            if (!sh) { err = sh.error(); break; }

            auto rgen  = compile_stage(sh->rgen,  "rgen",  err); if (!err.empty()) break;
            auto rchit = compile_stage(sh->rchit, "rchit", err); if (!err.empty()) break;
            auto rmiss = compile_stage(sh->rmiss, "rmiss", err); if (!err.empty()) break;
            std::vector<std::vector<std::uint32_t>> rints;
            for (auto& src : sh->rint_per_group) {
                rints.push_back(compile_stage(src, "rint", err));
                if (!err.empty()) break;
            }
            if (!err.empty()) break;

            // Per-group AABBs → multi-BLAS.
            std::vector<gpu::RtAabb> boxes;
            for (auto& g : groups) boxes.push_back(to_aabb(g.box));
            auto accel = gpu::RtAccel::build_groups(*ctx, boxes);
            if (!accel) { err = accel.error(); break; }

            gpu::ShaderPush sp = gpu::build_push_from_scene(scene, W, H);
            gpu::RtPushConstants pc;
            std::memcpy(&pc, &sp, sizeof(pc));

            // GPU energy around the trace (the recurring per-frame GPU work).
            if (gpu_e_ok) gpu_e->begin();
            auto img = gpu::rtx_trace_groups(*ctx, *accel, rgen, rints, rchit, rmiss, pc, W, H);
            if (!img) { err = img.error(); break; }
            rtx_trace_ms = img->trace_ms;
            if (gpu_e_ok) { auto j = gpu_e->end(); if (j) gpu_j = *j; }
        } while (false);

        if (rtx_trace_ms < 0) {
            std::printf("  %5d  %7zu  %12.1f  rtx failed: %s\n",
                        n, groups.size(), mpix / (cpu_trace_ms / 1000.0), err.c_str());
        } else {
            double cpu_tp = mpix / (cpu_trace_ms / 1000.0);
            double rtx_tp = mpix / (rtx_trace_ms / 1000.0);
            if (energy) {
                // Mpix/kWh = pix/kWh / 1e6. A counter that read nothing prints
                // a dash rather than a fabricated number.
                double px = (double)W * H;
                char cpu_ek[16] = "      -", rtx_ek[16] = "      -";
                if (cpu_j && *cpu_j > 0)
                    std::snprintf(cpu_ek, sizeof(cpu_ek), "%.1f",
                                  power::pixels_per_kwh(px, *cpu_j) / 1e6);
                if (gpu_j > 0)
                    std::snprintf(rtx_ek, sizeof(rtx_ek), "%.1f",
                                  power::pixels_per_kwh(px, gpu_j) / 1e6);
                std::printf("  %5d  %7zu  %12.1f  %12.1f  %12.1f  %14s  %14s\n",
                            n, groups.size(), cpu_tp, rtx_tp, cpu_tp + rtx_tp,
                            cpu_ek, rtx_ek);
            } else {
                std::printf("  %5d  %7zu  %12.1f  %12.1f  %12.1f\n",
                            n, groups.size(), cpu_tp, rtx_tp, cpu_tp + rtx_tp);
            }
        }
    }
    std::printf(
        "\n  Throughput = full-frame pixels / hot-loop time (cpu raymarch excl.\n"
        "  JIT; rtx vkCmdTraceRays excl. pipeline/SBT/AS setup).\n"
        "  sum = cpu + rtx: what running both paths concurrently delivers — the\n"
        "  heterogeneous win, since scaling one device can't reach it. Adding\n"
        "  more paths (gpu_glsl, gpu_ir, remote nodes) raises the sum further.\n");
    if (energy)
        std::printf(
            "  Mpix/kWh = energy efficiency from RAPL (cpu) / NVML (gpu) counters.\n"
            "  This is the separate axis that decides *which* devices to add to a\n"
            "  datacenter; a dash means the counter wasn't available. Note rtx\n"
            "  energy is trace-only and excludes the amortizable setup.\n");
    else
        std::printf(
            "  Energy (pix/kWh) and cost (pix/$) are separate axes; rerun with\n"
            "  --energy for RAPL/NVML measurement of the efficiency axis.\n");
    return 0;
}
