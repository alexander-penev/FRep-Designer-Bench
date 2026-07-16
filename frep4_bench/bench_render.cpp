// Render-path benchmark for FRep Designer 4.x. Complements bench_grid (raw SDF
// evaluation throughput) by measuring the *render* pipeline: for each scene and
// each available render path (CpuIr, GpuGlsl, GpuIr, GpuRtx) it renders a full
// frame and reports steady-state render time (median, with RAPL energy when
// available) plus the one-off compile time (codegen + JIT / SPIR-V build).
//
// Paths not available in the current environment (e.g. GPU paths with no Vulkan
// device) are skipped, not errored — same policy as the app.
//
// CSV columns match the suite: system,backend,scene,metric,size,ms,throughput,
// joules,uj_per_unit. metric="render", size=resolution, ms=median render time,
// throughput=Mpixel/s.
#include "core/exec/multipath.hpp"
#include "core/exec/executor_factory.hpp"
#include "core/io/scene_io.hpp"
#include "../common/timing.hpp"
#include <cstdio>
#include <string>
#include <memory>
#include <cmath>
#include <vector>

using namespace frep;
using namespace frep::exec;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <res> <scene.json>...\n"
            "  renders each scene on every available render path.\n", argv[0]);
        return 2;
    }
    int R = std::atoi(argv[1]);
    if (R <= 0) R = 256;

    const PathKind paths[] = {
        PathKind::CpuIr, PathKind::GpuGlsl, PathKind::GpuIr, PathKind::GpuRtx };

    for (int a = 2; a < argc; ++a) {
        std::string path = argv[a];
        std::string name = path;
        auto slash = name.find_last_of('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);
        auto dot = name.rfind(".json");
        if (dot != std::string::npos) name = name.substr(0, dot);

        SceneGraph scene;
        try {
            scene = io::load_scene(path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", path.c_str(), e.what());
            continue;
        }

        // Per-scene reference frame (first available path) for cross-path visual
        // parity: every other path's frame is compared to it. Big differences
        // mean the paths disagree — a correctness bug, however fast they run.
        std::vector<float> ref_rgba;
        std::string ref_backend;

        for (PathKind pk : paths) {
            const char* backend = path_kind_name(pk);
            TracerConfig cfg;
            auto exec = make_executor(pk, cfg);
            if (!exec || !exec->available()) {
                std::fprintf(stderr, "  %-8s %-16s unavailable\n",
                             backend, name.c_str());
                continue;
            }
            Tile full{0, 0, R, R};

            TileResult first = exec->render(scene, R, R, full);
            if (!first.ok) {
                std::fprintf(stderr, "  %-8s %-16s render error: %s\n",
                             backend, name.c_str(), first.error.c_str());
                continue;
            }
            double compile_ms = first.compile_ms;

            // Optional PPM dump for visual inspection (FREP4_DUMP_PPM=dir).
            if (const char* dd = std::getenv("FREP4_DUMP_PPM")) {
                char path[512];
                std::snprintf(path, sizeof path, "%s/%s_%s_%d.ppm",
                              dd, name.c_str(), backend, R);
                if (FILE* fp = std::fopen(path, "wb")) {
                    std::fprintf(fp, "P6\n%d %d\n255\n", R, R);
                    const auto& px = first.rgba;
                    for (size_t i = 0; i + 3 < px.size(); i += 4)
                        for (int c = 0; c < 3; ++c) {
                            float v = px[i + c];
                            unsigned char u = v <= 0 ? 0 : v >= 1 ? 255
                                            : (unsigned char)(v * 255.0f + 0.5f);
                            std::fputc(u, fp);
                        }
                    std::fclose(fp);
                }
            }

            // Sanity: a fast render of a black screen is worthless. Flag frames
            // that are (nearly) empty or contain NaN/inf — this is what catches
            // a scene that "renders" quickly but shows nothing (e.g. an imported
            // expression that evaluates to NaN everywhere).
            {
                long lit = 0, bad = 0; const auto& px = first.rgba;
                for (size_t i = 0; i + 3 < px.size(); i += 4) {
                    float r0 = px[i], g0 = px[i+1], b0 = px[i+2];
                    if (std::isnan(r0)||std::isnan(g0)||std::isnan(b0)||
                        std::isinf(r0)||std::isinf(g0)||std::isinf(b0)) { ++bad; continue; }
                    if (r0 + g0 + b0 > 0.05f) ++lit;
                }
                long npx = (long)px.size() / 4;
                double lit_pct = npx ? 100.0 * lit / npx : 0;
                if (bad > 0)
                    std::fprintf(stderr, "  !! %-8s %-16s %ld NaN/inf pixels — "
                        "render is CORRUPT\n", backend, name.c_str(), bad);
                else if (lit_pct < 1.0)
                    std::fprintf(stderr, "  !! %-8s %-16s only %.2f%% of frame lit"
                        " — possible BLACK SCREEN\n", backend, name.c_str(), lit_pct);
            }

            auto pr = median_ms_energy([&] { exec->render(scene, R, R, full); });
            double ms = pr.first;
            double J  = pr.second;
            double mpix_s = (ms > 0) ? (double(R) * R / ms / 1e3) : -1;
            double uj_per_pix = (J >= 0 && R > 0)
                ? (J * 1e6 / (double(R) * R)) : -1;

            // Cross-path visual parity vs the scene's reference frame.
            if (ref_rgba.empty()) {
                ref_rgba = first.rgba; ref_backend = backend;
            } else if (first.rgba.size() == ref_rgba.size()) {
                double maxd = 0, sumd = 0; long n = 0;
                for (size_t i = 0; i < ref_rgba.size(); ++i) {
                    double d = std::fabs((double)first.rgba[i] - ref_rgba[i]);
                    maxd = std::max(maxd, d); sumd += d; ++n;
                }
                double meand = n ? sumd / n : 0;
                const char* verdict = (maxd < 0.02) ? "identical"
                                    : (meand < 0.02) ? "close (AA/silhouette)"
                                    : "DIVERGENT";
                std::fprintf(stderr, "  ~~ %-8s vs %-8s (%s): max %.4f, mean %.4f — %s\n",
                    backend, ref_backend.c_str(), name.c_str(), maxd, meand, verdict);
            }
            std::fprintf(stderr, "  %-8s %-16s compile %.1f ms, render %.3f ms\n",
                         backend, name.c_str(), compile_ms, ms);
        }
    }
    return 0;
}
