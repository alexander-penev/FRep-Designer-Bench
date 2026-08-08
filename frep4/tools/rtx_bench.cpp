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
#include "core/gpu/rtx_instances.hpp"
#include "core/frep/smart_import.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/compiler/codegen.hpp"
#include "core/power/energy_meter.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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

// The first *visible* object's geometry. Visibility is not cosmetic here:
// factor_instances adds each shared prototype to the scene as a hidden
// definition, and taking the map's first entry would hand back that lone
// prototype instead of the grid — partitioning to one group and quietly
// benchmarking a single sphere. Match what the executors do.
FRepNode::Ptr scene_root(const SceneGraph& s) {
    for (auto& [id, o] : s.objects())
        if (o.visible && o.geometry) return o.geometry;
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
    bool factor = false;
    bool verbose = false;
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
                "report Mpix/kWh\n"
                "  --factor          run factor_instances on the grid first, so\n"
                "                    the shared stages emit one subprogram per\n"
                "                    shape instead of one inlined copy per object\n",
                argv[0]);
            return 0;
        }
        else if (a == "--width") W = next(W);
        else if (a == "--height") H = next(H);
        else if (a == "--energy") energy = true;
        else if (a == "--factor") factor = true;
        else if (a == "--verbose") verbose = true;
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
    // "shaders" is the number of distinct intersection shaders the RT pipeline
    // has to hold — the count instancing collapses. "build ms" is the one-off
    // pipeline + SBT construction that scales with it; it is amortized over
    // frames and excluded from the throughput columns, but it is the reason the
    // per-group build ran out of road at a thousand objects.
    if (energy)
        std::printf("  %5s  %7s  %8s  %9s  %12s  %12s  %12s  %14s  %14s\n",
                    "N", "groups", "shaders", "build ms",
                    "cpu Mpix/s", "rtx Mpix/s", "sum Mpix/s",
                    "cpu Mpix/kWh", "rtx Mpix/kWh");
    else
        std::printf("  %5s  %7s  %8s  %9s  %12s  %12s  %12s\n",
                    "N", "groups", "shaders", "build ms",
                    "cpu Mpix/s", "rtx Mpix/s", "sum Mpix/s");
    std::printf("  ----------------------------------------------------------------"
                "----------------%s\n", energy ? "--------------------------------" : "");

    const double mpix = (double)W * H / 1e6;

    TracerConfig cfg;
    exec::CpuIrExecutor cpu(SceneCodegen::SceneSdfMode::Inlined, cfg);

    for (int n : counts) {
        SceneGraph scene = bench::make_sphere_grid(n);
        // The grid is built flat — N independent Translate(Sphere) members with
        // no recorded sharing — which is how an imported model usually arrives.
        // --factor runs the structure-recovery pass first, rewiring the copies
        // to one prototype through InstanceNodes. That is what lets the *shared*
        // stages shrink too: the emitters turn an InstanceNode into a call to
        // one subprogram instead of another inlined copy of the body, so the
        // closest-hit shader (which evaluates the whole scene for normals,
        // albedo and shadows) stops growing a sphere at a time.
        if (factor) frep::factor_instances(scene);
        auto root = scene_root(scene);
        auto groups = gpu::partition_csg_groups(root);

        // CPU reference: the executor's render_ms is the raymarch hot loop
        // (excludes JIT compile), the like-for-like against the RT trace_ms.
        // Both become throughput below. Energy is measured around the render
        // call (a warmup render first so JIT compile isn't charged to energy).
        cpu.render(scene, W, H, exec::Tile{0, 0, W, H});  // warmup (JIT)
        auto rc = cpu.render(scene, W, H, exec::Tile{0, 0, W, H});
        if (!rc.ok) { std::printf("  %5d  cpu render failed: %s\n", n, rc.error.c_str()); continue; }
        double cpu_trace_ms = rc.render_ms;   // provisional; replaced by the median below
        std::optional<double> cpu_j;

        // RT multi-BLAS path.
        double rtx_trace_ms = -1.0;
        double rtx_build_ms = -1.0;   // cold pipeline + SBT (amortized after)
        int    n_shaders = 0;         // intersection shaders the pipeline needs
        double gpu_j = -1.0;
        std::string err;
        do {
            auto ctx = gpu::RtxCtx::create();
            if (!ctx) { err = ctx.error(); break; }

            // Prefer the instanced build: the grid is N copies of one sphere,
            // so one prototype BLAS + N transformed TLAS instances replaces N
            // BLASes and, more to the point, N intersection shaders. Without it
            // the pipeline build is O(N) and swamps the trace it is meant to
            // measure. FREP_RTX_NO_INSTANCING=1 pins the old per-group build so
            // the two constructions can be compared on the same scene.
            const bool no_inst = [] {
                const char* v = std::getenv("FREP_RTX_NO_INSTANCING");
                return v && v[0] == '1';
            }();
            auto inst = no_inst ? std::nullopt : gpu::factor_rt_instances(root);

            // Scenes to emit an intersection shader from: one per prototype on
            // the instanced path, one per group otherwise.
            std::vector<SceneGraph> gscenes;
            if (inst) {
                gscenes = gpu::make_proto_scenes(scene, *inst);
            } else {
                for (auto& g : groups) {
                    SceneGraph s; s.add_object(g.root); gscenes.push_back(std::move(s));
                }
            }
            auto sh = inst ? gpu::emit_rt_instanced_shaders(scene, gscenes, cfg)
                           : gpu::emit_rt_group_shaders(scene, gscenes, cfg);
            if (!sh) { err = sh.error(); break; }
            n_shaders = (int)gscenes.size();

            auto rgen  = compile_stage(sh->rgen,  "rgen",  err); if (!err.empty()) break;
            auto rchit = compile_stage(sh->rchit, "rchit", err); if (!err.empty()) break;
            auto rmiss = compile_stage(sh->rmiss, "rmiss", err); if (!err.empty()) break;
            std::vector<std::vector<std::uint32_t>> rints;
            for (auto& src : sh->rint_per_group) {
                rints.push_back(compile_stage(src, "rint", err));
                if (!err.empty()) break;
            }
            if (!err.empty()) break;
            // SPIR-V sizes, because "the build is slow" is not a diagnosis. The
            // driver compiles every module to machine code, so the cost tracks
            // module size — and only measuring says whether the weight sits in
            // the per-shape intersection shaders or in the shared closest-hit
            // stage, which still evaluates the WHOLE scene for the normal, the
            // albedo and the shadow ray.
            if (verbose) {
                std::size_t rint_words = 0;
                for (auto& r : rints) rint_words += r.size();
                std::fprintf(stderr,
                    "  [n=%d] spirv words: rgen %zu  rchit %zu  rmiss %zu  "
                    "rint x%zu = %zu\n", n, rgen.size(), rchit.size(),
                    rmiss.size(), rints.size(), rint_words);
            }

            // Boxes → multi-BLAS. On the instanced path they are the
            // prototypes' object-space bounds and the placement matrices put
            // the copies in the world; otherwise they are the groups' own
            // world-space bounds at the identity.
            std::vector<gpu::RtAabb> boxes;
            if (inst) for (auto& p : inst->protos) boxes.push_back(to_aabb(p.box));
            else      for (auto& g : groups)       boxes.push_back(to_aabb(g.box));
            auto accel = inst
                ? gpu::RtAccel::build_instanced(*ctx, boxes, inst->placements)
                : gpu::RtAccel::build_groups(*ctx, boxes);
            if (!accel) { err = accel.error(); break; }

            gpu::ShaderPush sp = gpu::build_push_from_scene(scene, W, H);
            gpu::RtPushConstants pc;
            std::memcpy(&pc, &sp, sizeof(pc));

            // Interleaved BURSTS, medians per path. Alternating one CPU render
            // with one GPU trace was the previous scheme, and at high N it
            // measured the driver's power-state machine instead of the
            // renderer: the CPU render grows to ~270 ms at N=256 while the
            // trace stays tens of ms, so the GPU sits idle ~87% of the time,
            // parks in P5/P8 (585 MHz core, 810 MHz memory were observed), and
            // each trace runs partly at a low clock. The symptom was a fourfold
            // spread in the same measurement -- N=256 read 28, 78 and 121 Mpix/s
            // depending only on how loaded the machine had been -- while N=64,
            // whose CPU phase is four times shorter, held to 1%.
            //
            // So: keep each device busy for a stretch long enough to hold its
            // clocks, and alternate at the level of stretches rather than
            // single samples. Slow drift still averages across both paths;
            // neither path is charged for the other's idle gaps.
            const int    ROUNDS = 5;
            const double BURST_MS = 300.0;
            auto burst = [&](auto&& one_sample) {
                auto b0 = std::chrono::steady_clock::now();
                do { if (!one_sample()) return false; }
                while (std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - b0).count() < BURST_MS);
                return true;
            };
            gpu::RtxPipelineCache pcache;
            std::vector<double> cpu_s, rtx_s;

            for (int w = 0; w < 2; ++w) {   // warm both devices and the cache
                cpu.render(scene, W, H, exec::Tile{0, 0, W, H});
                auto wi = gpu::rtx_trace_groups_cached(*ctx, *accel, pcache,
                                                       rgen, rints, rchit, rmiss, pc, W, H);
                if (!wi) { err = wi.error(); break; }
                if (w == 0) rtx_build_ms = wi->pipeline_ms;   // the cold build
            }
            if (!err.empty()) { pcache.release(*ctx); break; }

            if (cpu_e_ok) cpu_e->begin();
            if (gpu_e_ok) gpu_e->begin();
            for (int r = 0; r < ROUNDS; ++r) {
                burst([&] {
                    auto rr = cpu.render(scene, W, H, exec::Tile{0, 0, W, H});
                    if (rr.ok) cpu_s.push_back(rr.render_ms);
                    return true;
                });
                if (!burst([&] {
                        auto im = gpu::rtx_trace_groups_cached(
                            *ctx, *accel, pcache, rgen, rints, rchit, rmiss, pc, W, H);
                        if (!im) { err = im.error(); return false; }
                        rtx_s.push_back(im->trace_ms);
                        return true;
                    })) break;
            }
            // Energy per frame divides by the frames each device actually ran,
            // which bursting makes unequal between the two.
            if (cpu_e_ok && !cpu_s.empty())
                { auto j = cpu_e->end(); if (j) cpu_j = *j / (double)cpu_s.size(); }
            if (gpu_e_ok && !rtx_s.empty())
                { auto j = gpu_e->end(); if (j) gpu_j = *j / (double)rtx_s.size(); }
            pcache.release(*ctx);
            if (!err.empty()) break;

            auto med = [](std::vector<double> v) {
                std::sort(v.begin(), v.end()); return v.empty() ? -1.0 : v[v.size() / 2];
            };
            if (!cpu_s.empty()) cpu_trace_ms = med(cpu_s);
            rtx_trace_ms = med(rtx_s);
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
                std::printf("  %5d  %7zu  %8d  %9.1f  %12.1f  %12.1f  %12.1f  %14s  %14s\n",
                            n, groups.size(), n_shaders, rtx_build_ms,
                            cpu_tp, rtx_tp, cpu_tp + rtx_tp, cpu_ek, rtx_ek);
            } else {
                std::printf("  %5d  %7zu  %8d  %9.1f  %12.1f  %12.1f  %12.1f\n",
                            n, groups.size(), n_shaders, rtx_build_ms,
                            cpu_tp, rtx_tp, cpu_tp + rtx_tp);
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
