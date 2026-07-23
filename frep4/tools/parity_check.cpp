// tools/parity_check.cpp
//
// Cross-path equivalence harness. Renders every feature scene from
// parity_scenes.hpp on two paths and prints a per-scene table of mean / max
// absolute per-channel difference, plus an aggregate. This is how the
// equivalence claim is demonstrated across the whole feature surface rather
// than from a single scene.
//
//   parity_check --paths cpu_ir,gpu_glsl   (default)
//   parity_check --paths cpu_ir,gpu_ir
//   parity_check --width N --height N --tolerance F
//
// On hardware with a GPU this compares CPU_IR against the GPU paths. In a
// CPU-only sandbox the GPU executors report unavailable; pass
// --paths cpu_ir,cpu_ir to exercise the harness itself (it will report all
// zeros, confirming determinism).

#include "core/exec/parity_scenes.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/gpu_ir_executor.hpp"
#include "core/exec/rtx_executor.hpp"
#include "core/exec/executor_factory.hpp"

#include <chrono>
#include "core/exec/multipath.hpp"
#include "core/postprocess/post_process.hpp"
#include "core/io/scene_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace frep;
using namespace frep::exec;
using frep::post::BoxDownsampleSSAA;
namespace post = frep::post;

namespace {

using exec::make_executor;  // shared factory (core/exec/executor_factory.hpp)

struct Diff { double mean = 0, max = 0; };

Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    std::size_t n = std::min(a.size(), b.size());
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        sum += e; if (e > d.max) d.max = e;
    }
    d.mean = n ? sum / n : 0;
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    std::string paths = "cpu_ir,gpu_glsl";
    int W = 200, H = 150, ssaa = 1;
    double tol = 2.0 / 255.0;
    bool no_shadows = false, no_ao = false, no_specular = false;
    bool show_timing = false;
    std::string dump_dir, only_scene, dump_images_dir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* d) { return i + 1 < argc ? argv[++i] : d; };
        if      (a == "--paths")     paths = next("cpu_ir,gpu_glsl");
        else if (a == "--width")     W = std::atoi(next("200"));
        else if (a == "--height")    H = std::atoi(next("150"));
        else if (a == "--ssaa")      ssaa = std::atoi(next("1"));
        else if (a == "--tolerance") tol = std::atof(next("0.0078"));
        else if (a == "--no-shadows")  no_shadows = true;
        else if (a == "--no-ao")       no_ao = true;
        else if (a == "--no-specular") no_specular = true;
        else if (a == "--dump-scene")  dump_dir = next(".");
        else if (a == "--only")        only_scene = next("");
        else if (a == "--dump-images")  dump_images_dir = next(".");
        else if (a == "--timing")       show_timing = true;
        else if (a == "--help") {
            std::printf("usage: parity_check [--paths A,B] [--width N] "
                        "[--height N] [--ssaa N] [--tolerance F]\n"
                        "            [--no-shadows] [--no-ao] [--no-specular]\n"
                        "            [--dump-scene DIR]\n"
                        "  renders every parity scene on paths A and B and\n"
                        "  reports per-scene mean/max |Δ| (default cpu_ir,gpu_glsl)\n"
                        "  --ssaa N supersamples N× then downsamples before compare\n"
                        "  --no-* toggles isolate which shading term causes a divergence\n"
                        "  --dump-scene DIR writes each scene as <name>.json into DIR\n"
                        "    (then render with frep_multipath to inspect a divergence)\n");
            return 0;
        }
    }
    if (ssaa < 1) ssaa = 1;

    // Dump scenes as JSON and exit, for manual rendering / inspection.
    if (!dump_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        if (ec) {
            std::fprintf(stderr, "cannot create %s: %s\n",
                         dump_dir.c_str(), ec.message().c_str());
            return 1;
        }
        for (const auto& ns : parity::all_scenes()) {
            SceneGraph s = ns.make();
            std::string path = dump_dir + "/" + ns.name + ".json";
            std::ofstream f(path);
            if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); continue; }
            f << io::serialize_scene(s);
            std::printf("  wrote %s\n", path.c_str());
        }
        return 0;
    }

    auto comma = paths.find(',');
    if (comma == std::string::npos) {
        std::fprintf(stderr, "need two paths, e.g. --paths cpu_ir,gpu_glsl\n");
        return 1;
    }
    std::string pa = paths.substr(0, comma), pb = paths.substr(comma + 1);
    if (pb.find(',') != std::string::npos) {
        std::fprintf(stderr,
            "parity_check compares exactly two paths at a time (a reference vs\n"
            "one target). For three paths, run each pair against the reference:\n"
            "  parity_check --paths cpu_ir,gpu_glsl ...\n"
            "  parity_check --paths cpu_ir,gpu_ir   ...\n"
            "  parity_check --paths cpu_ir,gpu_rtx  ...\n");
        return 1;
    }

    TracerConfig cfg;
    if (no_shadows)  cfg.enable_shadows = false;
    if (no_ao)       cfg.enable_ao = false;
    if (no_specular) cfg.enable_specular = false;
    auto ea = make_executor(pa, cfg), eb = make_executor(pb, cfg);
    if (!ea || !eb) { std::fprintf(stderr, "unknown path in '%s'\n", paths.c_str()); return 1; }

    std::printf("parity_check: %s vs %s  %dx%d", pa.c_str(), pb.c_str(), W, H);
    if (ssaa > 1) std::printf("  (SSAA %dx)", ssaa);
    std::printf("  tol=%.4f\n\n", tol);
    if (!ea->available()) std::printf("  [warning] %s unavailable\n", pa.c_str());
    if (!eb->available()) std::printf("  [warning] %s unavailable\n", pb.c_str());

    // When the RT path is involved, say which backend it's running on, so a
    // result is unambiguous about whether it was real RT cores or a software
    // emulation (this matters for what the paper can claim).
    if (pa == "gpu_rtx" || pb == "gpu_rtx") {
        auto caps = gpu::detect_rtx_caps();
        std::printf("  [gpu_rtx backend] %s\n", caps.describe().c_str());
    }

    std::printf("  %-16s %12s %12s   %s", "scene", "mean|Δ|", "max|Δ|", "status");
    if (show_timing) std::printf("   %10s / %10s", (pa+" ms").c_str(), (pb+" ms").c_str());
    std::printf("\n");
    std::printf("  %s\n", std::string(56, '-').c_str());

    const int rw = W * ssaa, rh = H * ssaa;
    post::BoxDownsampleSSAA ss(ssaa);
    auto downsample = [&](std::vector<float> img) -> std::vector<float> {
        if (ssaa <= 1) return img;
        post::Frame o = ss.apply(post::Frame(std::move(img), rw, rh));
        return std::move(o.rgba);
    };

    double worst_mean = 0, worst_max = 0;
    int n_ok = 0, n_total = 0;
    for (const auto& ns : parity::all_scenes()) {
        if (!only_scene.empty() && only_scene != ns.name) continue;
        SceneGraph s = ns.make();
        // Recreate executors per scene so each render is independent — a
        // reused GPU executor carries Vulkan context/buffer state across
        // scenes, which is not what a parity comparison should measure (and
        // can inflate per-scene divergence). This matches how frep_multipath
        // renders a single scene in a fresh executor.
        auto ea_s = make_executor(pa, cfg);
        auto eb_s = make_executor(pb, cfg);
        auto t0 = std::chrono::steady_clock::now();
        auto ra = ea_s->render(s, rw, rh, Tile{0, 0, rw, rh});
        auto t1 = std::chrono::steady_clock::now();
        auto rb = eb_s->render(s, rw, rh, Tile{0, 0, rw, rh});
        auto t2 = std::chrono::steady_clock::now();
        double ms_a = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_b = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (!ra.ok || !rb.ok) {
            // A documented "not yet supported on this path" (e.g. textures on
            // gpu_rtx on an unsupported feature) is a SKIP, not a failure — the path simply
            // doesn't cover that feature yet. Anything else is a real FAIL.
            const std::string& err = !ra.ok ? ra.error : rb.error;
            bool unsupported = err.find("not yet supported") != std::string::npos ||
                               err.find("not supported") != std::string::npos;
            std::printf("  %-16s %12s %12s   %s\n", ns.name, "-", "-",
                        unsupported ? "SKIP" : "RENDER FAIL");
            if (!unsupported) {
                if (!ra.ok && !ra.error.empty())
                    std::printf("    [%s] %s\n", pa.c_str(), ra.error.c_str());
                if (!rb.ok && !rb.error.empty())
                    std::printf("    [%s] %s\n", pb.c_str(), rb.error.c_str());
            }
            continue;
        }
        auto a_img = downsample(std::move(ra.rgba));
        auto b_img = downsample(std::move(rb.rgba));
        // Optionally write both paths' frames as PPM for offline pixel
        // comparison (e.g. to localize a per-scene divergence by hand).
        if (!dump_images_dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dump_images_dir, ec);
            auto write_ppm = [&](const std::string& path,
                                 const std::vector<float>& img) {
                std::ofstream f(path, std::ios::binary);
                f << "P6\n" << W << " " << H << "\n255\n";
                for (int i = 0; i < W * H; ++i)
                    for (int c = 0; c < 3; ++c) {
                        float v = img[i * 4 + c];
                        v = v < 0 ? 0 : v > 1 ? 1 : v;
                        unsigned char byte = (unsigned char)(v * 255.0f + 0.5f);
                        f.put((char)byte);
                    }
            };
            write_ppm(dump_images_dir + "/" + std::string(ns.name) + "_" + pa + ".ppm", a_img);
            write_ppm(dump_images_dir + "/" + std::string(ns.name) + "_" + pb + ".ppm", b_img);
            std::printf("  wrote %s_{%s,%s}.ppm\n", ns.name,
                        pa.c_str(), pb.c_str());
        }
        Diff d = compare(a_img, b_img);
        bool ok = d.mean <= tol;
        ++n_total; if (ok) ++n_ok;
        worst_mean = std::max(worst_mean, d.mean);
        worst_max  = std::max(worst_max, d.max);
        std::printf("  %-16s %12.6f %12.6f   %s",
                    ns.name, d.mean, d.max, ok ? "ok" : "DIVERGENT");
        if (show_timing) {
            // Total wall time per path, plus (for paths that report it) the
            // trace-only cost — so RT's amortizable per-frame setup doesn't
            // get conflated with the recurring GPU trace cost.
            std::printf("   %8.1f / %8.1f ms", ms_a, ms_b);
            if (rb.render_ms > 0.0 && rb.compile_ms > 0.0)
                std::printf("  [%s trace %.1f + setup %.1f]",
                            pb.c_str(), rb.render_ms, rb.compile_ms);
        }
        std::printf("\n");
    }

    std::printf("  %s\n", std::string(56, '-').c_str());
    std::printf("  %-16s %12.6f %12.6f   %d/%d within tol\n",
                "worst", worst_mean, worst_max, n_ok, n_total);
    return (n_ok == n_total) ? 0 : 2;
}
