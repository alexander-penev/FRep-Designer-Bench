// scene_parity — cross-path image equivalence for a scene *file*.
//
// frep4's own parity_check sweeps the built-in feature scenes; it cannot see
// the structured benchmark scenes (c2_colonnade_struct, c1_gear_struct) that
// live as JSON in scenes/frep4. Those are exactly the ones the RT instanced
// build changes: a scene whose union members repeat a shape now renders through
// one shared prototype BLAS with a per-instance transform, and its intersection
// shader marches in object space rather than world space.
//
// That rewrite is only legitimate if the picture does not move. This renders a
// scene on two paths and reports mean / max per-channel |difference|, so the
// claim "the instanced path is the same renderer" is a measurement rather than
// an argument.
//
//   scene_parity <scene.json> [--paths cpu_ir,gpu_rtx] [--res N] [--tolerance F]
#include "core/exec/multipath.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/io/scene_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace frep;
using namespace frep::exec;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <scene.json> [--paths A,B] [--res N] [--tolerance F]\n"
            "  renders the scene on both paths and reports mean/max |delta|\n",
            argv[0]);
        return 2;
    }
    std::string scene_path = argv[1];
    std::string paths = "cpu_ir,gpu_rtx";
    int R = 512;
    double tol = 2.0 / 255.0;
    // Shading toggles, so a divergence can be attributed. The instanced RT
    // build changes where shadows and ambient occlusion get their answer from,
    // and those are the two terms that read the whole scene rather than the
    // shape that was hit.
    bool no_shadows = false, no_ao = false;
    int shadow_samples = -1;
    // Payload to request from both paths. The default, float, compares the
    // paths' arithmetic; --format rgba8 compares what the GPU paths hand back
    // by default (and the CPU path quantised the same way), which checks the
    // device-side packing itself.
    auto fmt = TracerConfig::OutputFormat::Float;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* d) { return i + 1 < argc ? argv[++i] : d; };
        if      (a == "--paths")     paths = next("cpu_ir,gpu_rtx");
        else if (a == "--res")       R = std::atoi(next("512"));
        else if (a == "--tolerance") tol = std::atof(next("0.0078"));
        else if (a == "--no-shadows")  no_shadows = true;
        else if (a == "--no-ao")       no_ao = true;
        else if (a == "--shadow-samples") shadow_samples = std::atoi(next("1"));
        else if (a == "--format") {
            std::string f = next("float");
            fmt = f == "rgba8" ? TracerConfig::OutputFormat::RGBA8
                               : TracerConfig::OutputFormat::Float;
        }
    }
    auto comma = paths.find(',');
    if (comma == std::string::npos) {
        std::fprintf(stderr, "need two paths, e.g. --paths cpu_ir,gpu_rtx\n");
        return 2;
    }
    std::string pa = paths.substr(0, comma), pb = paths.substr(comma + 1);

    SceneGraph scene;
    try { scene = io::load_scene(scene_path); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "load %s: %s\n", scene_path.c_str(), e.what());
        return 1;
    }

    TracerConfig cfg;
    if (no_shadows) cfg.enable_shadows = false;
    if (no_ao)      cfg.enable_ao = false;
    if (shadow_samples > 0) cfg.shadow_samples = shadow_samples;
    cfg.output_format = fmt;
    auto ea = make_executor(pa, cfg), eb = make_executor(pb, cfg);
    if (!ea || !eb) { std::fprintf(stderr, "unknown path in '%s'\n", paths.c_str()); return 1; }
    if (!ea->available()) { std::fprintf(stderr, "%s unavailable\n", pa.c_str()); return 1; }
    if (!eb->available()) { std::fprintf(stderr, "%s unavailable\n", pb.c_str()); return 1; }

    Tile full{0, 0, R, R};
    TileResult ra = ea->render(scene, R, R, full);
    TileResult rb = eb->render(scene, R, R, full);
    if (!ra.ok) { std::fprintf(stderr, "%s: %s\n", pa.c_str(), ra.error.c_str()); return 1; }
    if (!rb.ok) { std::fprintf(stderr, "%s: %s\n", pb.c_str(), rb.error.c_str()); return 1; }
    ra.to_float(); rb.to_float();   // compare as floats whatever was requested

    std::size_t n = std::min(ra.rgba.size(), rb.rgba.size());
    if (n == 0) { std::fprintf(stderr, "empty frame\n"); return 1; }
    double sum = 0, mx = 0;
    std::size_t over = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)ra.rgba[i] - (double)rb.rgba[i]);
        sum += e;
        if (e > mx) mx = e;
        if (e > tol) ++over;
    }
    double mean = sum / (double)n;
    std::printf("%s  %dx%d  %s vs %s\n", scene_path.c_str(), R, R,
                pa.c_str(), pb.c_str());
    std::printf("  mean |d| = %.6f   max |d| = %.6f   over tol (%.4f): "
                "%zu / %zu channels (%.3f%%)\n",
                mean, mx, tol, over, n, 100.0 * (double)over / (double)n);
    // A max over tolerance on a hard silhouette is expected — two marchers stop
    // at different sub-pixel points on a near-tangent ray. The mean is the
    // honest summary: a real divergence moves the whole image, not its edges.
    std::printf("  %s\n", mean <= tol ? "MATCH (mean within tolerance)"
                                      : "DIVERGED");
    return mean <= tol ? 0 : 1;
}
