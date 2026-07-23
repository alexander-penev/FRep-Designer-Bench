// tools/multipath_driver.cpp
//
// frep_multipath — drive the Model D multi-path executor from the CLI.
// Selects paths (executors), a decompose/dispatch/merge strategy, and a
// run mode, renders a scene, and writes the merged image (PPM) plus a
// textual report. This is the tool for measuring cross-path visual
// equivalence on real hardware (compare merge) and for trying frame-split
// configurations (stitch merge).
//
// Examples:
//   frep_multipath scene.json --paths cpu_ir,gpu_glsl --merge compare
//       → both paths render the whole frame; report max/mean pixel diff.
//   frep_multipath scene.json --paths cpu_ir,gpu_glsl \
//                  --decompose halves --merge stitch --out split.ppm
//       → top/bottom split across the two paths, stitched into one image.

#include "core/io/scene_io.hpp"
#include "core/frep/scene.hpp"
#include "core/power/energy_meter.hpp"
#include "core/exec/multipath.hpp"
#include "core/postprocess/post_process.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/gpu_ir_executor.hpp"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <tuple>
#include <vector>

using namespace frep;
using namespace frep::exec;

namespace {

void write_ppm(const std::string& path, const std::vector<float>& img,
               int W, int H) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = img[(std::size_t)i * 4 + c];
            v = v < 0 ? 0 : (v > 1 ? 1 : v);
            f.put((char)(unsigned char)(v * 255.0f + 0.5f));
        }
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init_llvm(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        std::printf(
            "usage: %s <scene.json> [options]\n"
            "  --paths LIST      comma-separated: cpu_ir,gpu_glsl,gpu_ir  (default cpu_ir)\n"
            "  --decompose K     whole | halves | bands:N | weighted[:w0,w1,..] | grid:WxH   (default whole)\n"
            "  --dispatch K      one_per_tile | all_paths             (default: all_paths\n"
            "                    for compare, one_per_tile otherwise)\n"
            "  --merge K         compare | stitch                     (default compare)\n"
            "  --mode K          concurrent | serial | dynamic        (default concurrent)\n"
            "                    dynamic = work-stealing tile queue (self-balancing,\n"
            "                    use with --decompose grid:WxH)\n"
            "  --width N --height N                                   (default 400x300)\n"
            "  --ssaa N          supersample N×, box-downsample stitched frame\n"
            "  --denoise         bilateral denoise (edge-preserving)\n"
            "  --tonemap OP      reinhard | aces  (HDR→LDR on the stitched frame)\n"
            "  --gamma G         gamma-encode the stitched frame (e.g. 2.2)\n"
            "  --tolerance F     compare tolerance, 0..1 per channel  (default %.4f)\n"
            "  --energy          measure CPU (RAPL) + GPU (NVML) energy, report Mpix/kWh\n"
            "  --out FILE.ppm    write merged image                   (default multipath.ppm)\n",
            argv[0], 2.0 / 255.0);
        return argc < 2 ? 1 : 0;
    }

    std::string scene_path = argv[1];
    std::string paths_s = "cpu_ir", decompose_s = "whole", dispatch_s = "",
                merge_s = "compare", mode_s = "concurrent", out_s = "multipath.ppm",
                diff_s = "";
    bool diag_no_shadows = false, diag_no_ao = false, diag_flat = false,
         diag_no_specular = false;
    int W = 400, H = 300;
    int ssaa = 1;
    std::string tonemap_s = "";
    double gamma_v = 0.0;
    bool denoise_on = false;
    double tolerance = 2.0 / 255.0;
    bool energy = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if (a == "--paths")      paths_s = next("cpu_ir");
        else if (a == "--decompose") decompose_s = next("whole");
        else if (a == "--dispatch")  dispatch_s = next("");
        else if (a == "--merge")     merge_s = next("compare");
        else if (a == "--mode")      mode_s = next("concurrent");
        else if (a == "--width")     W = std::atoi(next("400").c_str());
        else if (a == "--height")    H = std::atoi(next("300").c_str());
        else if (a == "--ssaa")      ssaa = std::atoi(next("1").c_str());
        else if (a == "--tonemap")   tonemap_s = next("");      // reinhard | aces
        else if (a == "--gamma")     gamma_v = std::atof(next("0").c_str());
        else if (a == "--denoise")   denoise_on = true;
        else if (a == "--tolerance") tolerance = std::atof(next("0").c_str());
        else if (a == "--out")       out_s = next("multipath.ppm");
        else if (a == "--diff")      diff_s = next("");
        else if (a == "--no-shadows") diag_no_shadows = true;
        else if (a == "--no-ao")      diag_no_ao = true;
        else if (a == "--no-specular") diag_no_specular = true;
        else if (a == "--flat")       diag_flat = true;
        else if (a == "--energy")     energy = true;
    }

    // Diagnostic TracerConfig: disable shading components to isolate where
    // two paths diverge. Both executors get the SAME cfg so a compare is
    // fair — only the chosen component is toggled.
    TracerConfig diag_cfg;
    if (diag_no_shadows) diag_cfg.enable_shadows = false;
    if (diag_no_ao)      diag_cfg.enable_ao = false;
    if (diag_no_specular) diag_cfg.enable_specular = false;
    if (diag_flat)       diag_cfg.shading_model = TracerConfig::ShadingModel::BlinnPhong;
    if (diag_no_shadows || diag_no_ao || diag_flat || diag_no_specular)
        std::printf("  diagnostic: shadows=%d ao=%d specular=%d shading=%s\n",
                    diag_cfg.enable_shadows, diag_cfg.enable_ao,
                    diag_cfg.enable_specular,
                    diag_flat ? "BlinnPhong" : "CookTorrance");

    // Load scene.
    SceneGraph scene;
    try {
        scene = io::load_scene(scene_path, nullptr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }

    // Build executors from the path list. Own them here.
    std::vector<std::unique_ptr<IExecutor>> owned;
    std::vector<IExecutor*> execs;
    for (const auto& p : split_csv(paths_s)) {
        if (p == "cpu_ir") owned.push_back(std::make_unique<CpuIrExecutor>(
                SceneCodegen::SceneSdfMode::Inlined, diag_cfg));
        else if (p == "gpu_glsl") owned.push_back(std::make_unique<GpuGlslExecutor>(diag_cfg));
        else if (p == "gpu_ir") owned.push_back(std::make_unique<GpuIrExecutor>());
        else { std::fprintf(stderr, "unknown path '%s' (cpu_ir|gpu_glsl|gpu_ir)\n", p.c_str()); return 1; }
        execs.push_back(owned.back().get());
    }
    if (execs.empty()) { std::fprintf(stderr, "no paths\n"); return 1; }

    // Decompose strategy.
    std::unique_ptr<IDecomposeStrategy> dec;
    if (decompose_s == "whole")  dec = std::make_unique<WholeFrame>();
    else if (decompose_s == "halves") dec = std::make_unique<HorizontalBands>(2);
    else if (decompose_s.rfind("bands:", 0) == 0)
        dec = std::make_unique<HorizontalBands>(std::atoi(decompose_s.c_str() + 6));
    else if (decompose_s == "weighted" || decompose_s.rfind("weighted:", 0) == 0) {
        // Load-balanced split: band heights ∝ executor throughput so all
        // bands finish together. Weights come either from an explicit
        // "weighted:w0,w1,..." spec or from a calibration pass that renders
        // a small trial band on each executor and sets weight = 1/render_ms.
        std::vector<double> weights;
        if (decompose_s.rfind("weighted:", 0) == 0) {
            for (const auto& tok : split_csv(decompose_s.substr(9)))
                weights.push_back(std::atof(tok.c_str()));
        } else {
            // Calibrate: render a thin full-width strip on each executor and
            // set weight = 1/render_ms. The strip is taken from the VERTICAL
            // CENTRE of the frame, not the top: the top band is usually empty
            // sky (cheap miss rays), which wildly underestimates the cost of a
            // band that actually covers geometry. A central strip samples the
            // representative per-row cost so the weights reflect the real
            // per-band time. (A scattered-row sample would be even more
            // representative, but a central strip captures the main object
            // mass for these scenes and keeps the probe one contiguous tile.)
            std::printf("  calibrating weights (trial render per path)...\n");
            const int strip_h = std::max(8, H / 12);
            const int y0 = std::max(0, (H - strip_h) / 2);
            Tile probe{0, y0, W, std::min(H, y0 + strip_h)};
            for (auto* e : execs) {
                if (!e->available()) { weights.push_back(0.0); continue; }
                auto r = e->render(scene, W, H, probe);
                double ms = r.ok ? std::max(r.render_ms, 0.01) : 1e9;
                double w = r.ok ? 1.0 / ms : 0.0;
                weights.push_back(w);
                std::printf("    [%s] probe %dx%d @ y=%d render %.2f ms → weight %.4f\n",
                            path_kind_name(e->path()), W, strip_h, y0, r.render_ms, w);
            }
        }
        if (weights.size() != execs.size()) {
            std::fprintf(stderr, "weighted: %zu weights for %zu paths\n",
                         weights.size(), execs.size());
            return 1;
        }
        dec = std::make_unique<WeightedBands>(std::move(weights));
    }
    else if (decompose_s.rfind("grid:", 0) == 0) {
        const char* spec = decompose_s.c_str() + 5;
        int tw = std::atoi(spec);
        const char* x = std::strchr(spec, 'x');
        int th = x ? std::atoi(x + 1) : tw;
        dec = std::make_unique<GridDecompose>(tw, th);
    } else { std::fprintf(stderr, "unknown decompose '%s'\n", decompose_s.c_str()); return 1; }

    // Dispatch strategy (default depends on merge).
    if (dispatch_s.empty()) dispatch_s = (merge_s == "compare") ? "all_paths" : "one_per_tile";
    std::unique_ptr<IDispatchStrategy> dsp;
    if (dispatch_s == "all_paths") dsp = std::make_unique<AllPathsPerTile>();
    else if (dispatch_s == "one_per_tile") dsp = std::make_unique<OnePathPerTile>();
    else { std::fprintf(stderr, "unknown dispatch '%s'\n", dispatch_s.c_str()); return 1; }

    // Merge strategy.
    std::unique_ptr<IMergeStrategy> mrg;
    if (merge_s == "compare") mrg = std::make_unique<CompareMerge>(tolerance);
    else if (merge_s == "stitch") mrg = std::make_unique<StitchMerge>();
    else { std::fprintf(stderr, "unknown merge '%s'\n", merge_s.c_str()); return 1; }

    RunMode mode = (mode_s == "serial") ? RunMode::Serial
                 : (mode_s == "dynamic") ? RunMode::DynamicQueue
                 : RunMode::Concurrent;

    if (ssaa < 1) ssaa = 1;
    // With SSAA, render the whole pipeline at ss× resolution, then box-
    // downsample the STITCHED frame as a post-process stage. Doing the
    // downsample after the stitch (not per tile) is the point: a CPU+GPU
    // split supersamples correctly across the seam, because neighbouring
    // hi-res samples that straddle the band boundary are averaged together.
    const int rw = W * ssaa;
    const int rh = H * ssaa;

    // Report availability up front.
    std::printf("frep_multipath: %s  %dx%d", scene_path.c_str(), W, H);
    if (ssaa > 1) std::printf("  (render %dx%d, SSAA %dx)", rw, rh, ssaa);
    std::printf("\n  paths:");
    for (auto* e : execs)
        std::printf(" %s%s", path_kind_name(e->path()), e->available() ? "" : "(unavailable)");
    std::printf("\n  decompose=%s dispatch=%s merge=%s mode=%s\n",
                decompose_s.c_str(), dispatch_s.c_str(), merge_s.c_str(), mode_s.c_str());

    MultiPathExecutor mpe(*dec, *dsp, *mrg);

    // Optional energy: bracket the whole multipath render with CPU (RAPL) and
    // GPU (NVML) counters. The heterogeneous aggregate is exactly where energy
    // efficiency (Σ Mpix/kWh) is the honest cross-device metric — throughput
    // alone hides that a path may be fast but power-hungry. Counters that
    // aren't available are simply omitted, never invented.
    std::unique_ptr<power::EnergyCounter> cpu_e, gpu_e;
    if (energy) {
        cpu_e = power::make_cpu_energy_counter();
        gpu_e = power::make_gpu_energy_counter(0);
        std::printf("  [energy] cpu: %s  gpu: %s\n",
                    (cpu_e && cpu_e->available()) ? cpu_e->domain().c_str()
                                                  : "unavailable",
                    (gpu_e && gpu_e->available()) ? gpu_e->domain().c_str()
                                                  : "unavailable");
        if (!cpu_e || !cpu_e->available())
            std::printf("  [energy] cpu unavailable — for RAPL without root: "
                        "set perf_event_paranoid<=0, or "
                        "chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj\n");
        if (cpu_e && cpu_e->available()) cpu_e->begin();
        if (gpu_e && gpu_e->available()) gpu_e->begin();
    }

    auto res = mpe.run(scene, rw, rh, execs, mode);

    double cpu_j = 0, gpu_j = 0;
    bool have_cpu_j = false, have_gpu_j = false;
    if (energy) {
        if (cpu_e && cpu_e->available()) {
            if (auto j = cpu_e->end()) { cpu_j = *j; have_cpu_j = true; }
        }
        if (gpu_e && gpu_e->available()) {
            if (auto j = gpu_e->end()) { gpu_j = *j; have_gpu_j = true; }
        }
    }

    // Per-tile metrics.
    std::printf("  jobs: %d across %d path(s)\n", res.timings.job_count, res.timings.path_count);
    for (const auto& t : res.tiles) {
        std::printf("    [%s] tile (%d,%d)-(%d,%d)  %s  compile %.1f ms  render %.1f ms\n",
                    path_kind_name(t.path), t.tile.x0, t.tile.y0, t.tile.x1, t.tile.y1,
                    t.ok ? "ok" : ("FAIL: " + t.error).c_str(),
                    t.compile_ms, t.render_ms);
    }
    std::printf("  timing: wall %.1f ms, Σrender %.1f ms (speedup vs serial-sum %.2fx)\n",
                res.timings.wall_ms, res.timings.sum_render_ms,
                res.timings.wall_ms > 0 ? res.timings.sum_render_ms / res.timings.wall_ms : 0.0);

    // Energy efficiency: Mpix per kWh for the whole heterogeneous render.
    // Pixels are the delivered output frame (W*H, not the SSAA-scaled buffer).
    if (energy && (have_cpu_j || have_gpu_j)) {
        double total_j = cpu_j + gpu_j;
        double pix = (double)W * H;
        std::printf("  energy:");
        if (have_cpu_j) std::printf(" cpu %.2f J", cpu_j);
        if (have_gpu_j) std::printf(" gpu %.2f J", gpu_j);
        std::printf(" total %.2f J", total_j);
        if (total_j > 0)
            std::printf("  (%.0f Mpix/kWh)", power::pixels_per_kwh(pix, total_j) / 1e6);
        std::printf("\n");
    }

    // Per-executor distribution: how many tiles each path took and its share
    // of compile vs render time. Makes the load balance (and whether compile
    // dominates) legible at a glance — especially for the dynamic queue.
    if (res.timings.job_count > 1) {
        struct Acc { int tiles = 0; double compile = 0, render = 0; };
        std::map<PathKind, Acc> by_path;
        for (const auto& t : res.tiles) {
            auto& a = by_path[t.path];
            a.tiles++; a.compile += t.compile_ms; a.render += t.render_ms;
        }
        std::printf("  distribution:\n");
        for (const auto& [k, a] : by_path)
            std::printf("    [%s] %d tile(s)  compile %.1f ms  render %.1f ms\n",
                        path_kind_name(k), a.tiles, a.compile, a.render);
    }
    std::printf("  merge: %s\n", res.merged.report.c_str());

    // Post-process pipeline: applied to the whole stitched frame (rw×rh →
    // W×H after SSAA). Order: downsample → denoise → tone-map → gamma, the
    // conventional sequence (resolve, clean, compress range, encode). Each is
    // a composable stage; they run once on the assembled image, after the
    // split is merged, so they are correct across tile/band seams.
    post::BoxDownsampleSSAA ssaa_stage(ssaa);
    post::BilateralDenoise  denoise_stage(2, 2.0f, 0.1f);
    post::ToneMap           tonemap_stage(
        tonemap_s == "aces" ? post::ToneMap::Op::ACES
                            : post::ToneMap::Op::Reinhard);
    post::GammaCorrect      gamma_stage(gamma_v > 0 ? (float)gamma_v : 2.2f);

    post::PostProcessPipeline pipe;
    if (ssaa > 1)            pipe.add(&ssaa_stage);
    if (denoise_on)          pipe.add(&denoise_stage);
    if (!tonemap_s.empty())  pipe.add(&tonemap_stage);
    if (gamma_v > 0)         pipe.add(&gamma_stage);

    if (!pipe.empty()) {
        std::printf("  post-process:");
        if (ssaa > 1)           std::printf(" SSAA%dx", ssaa);
        if (denoise_on)         std::printf(" denoise");
        if (!tonemap_s.empty()) std::printf(" tonemap(%s)",
                                    tonemap_s == "aces" ? "aces" : "reinhard");
        if (gamma_v > 0)        std::printf(" gamma(%.1f)", gamma_v);
        std::printf("\n");
    }

    auto finalize = [&](std::vector<float> img, int iw, int ih)
        -> std::tuple<std::vector<float>, int, int> {
        if (pipe.empty() || img.empty()) return {std::move(img), iw, ih};
        post::Frame f(std::move(img), iw, ih);
        post::Frame o = pipe.apply(f);
        return {std::move(o.rgba), o.w, o.h};
    };

    // Write merged image if present (post-processed).
    if (!res.merged.image.empty()) {
        auto [img, iw, ih] = finalize(res.merged.image, rw, rh);
        write_ppm(out_s, img, iw, ih);
        std::printf("  wrote %s%s\n", out_s.c_str(),
                    pipe.empty() ? "" : " (post-processed)");
    }
    // Write the diff heat map (compare merge). Only the SSAA downsample
    // applies here (to match output dims); tone-map/gamma are radiance
    // operators and would distort a diagnostic heat map, so skip them.
    if (!res.merged.diff_image.empty()) {
        std::string diff_path = diff_s;
        if (diff_path.empty()) {
            auto dot = out_s.rfind(".ppm");
            diff_path = (dot != std::string::npos ? out_s.substr(0, dot) : out_s) + ".diff.ppm";
        }
        std::vector<float> dimg = res.merged.diff_image; int diw = rw, dih = rh;
        if (ssaa > 1) {
            post::Frame o = ssaa_stage.apply(post::Frame(std::move(dimg), rw, rh));
            dimg = std::move(o.rgba); diw = o.w; dih = o.h;
        }
        write_ppm(diff_path, dimg, diw, dih);
        std::printf("  wrote %s (diff heat map)\n", diff_path.c_str());
    }

    // Exit non-zero if a compare found divergence — handy for scripting the
    // equivalence check in CI / experiments.
    if (merge_s == "compare" && !res.merged.consistent) {
        std::printf("  → DIVERGENT (max |Δ| %.4f > tolerance %.4f)\n",
                    res.merged.max_abs_diff, tolerance);
        return 2;
    }
    return 0;
}
