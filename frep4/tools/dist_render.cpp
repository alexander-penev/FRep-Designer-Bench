// tools/dist_render.cpp
//
// Distributed render driver. One binary, two roles:
//
//   dist_render --master scene.json --port P --workers N \
//               --decompose grid:WxH --width W --height H \
//               [--scheduler pull|push] [--ssaa N] [--out file.ppm]
//       Binds port P, waits for N workers, serves tiles (pull = work-
//       stealing, push = round-robin), stitches, post-processes, writes PPM.
//
//   dist_render --worker --host H --port P --paths cpu_ir|gpu_glsl|gpu_ir \
//               --width W --height H
//       Connects to the master, receives the scene, renders the tiles it is
//       given with its chosen path, sends pixels back.
//
// For the localhost PoC: run one --master and N --worker processes (each in
// its own terminal / backgrounded). The master and workers must agree on
// --width/--height (the full frame dimensions used for ray setup).

#include "core/dist/master.hpp"
#include "core/dist/worker.hpp"
#include "core/dist/scheduler.hpp"
#include "core/exec/multipath.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/gpu_ir_executor.hpp"
#include "core/postprocess/post_process.hpp"
#include "core/io/scene_io.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

using namespace frep;
using namespace frep::dist;
using namespace frep::exec;

namespace {

void write_ppm(const std::string& path, const std::vector<float>& img, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = img[i * 4 + c];
            v = v < 0 ? 0 : (v > 1 ? 1 : v);
            unsigned char b = (unsigned char)(v * 255.0f + 0.5f);
            f.put((char)b);
        }
    }
}

std::unique_ptr<IExecutor> make_executor(const std::string& path) {
    if (path == "cpu_ir")   return std::make_unique<CpuIrExecutor>(
                                   SceneCodegen::SceneSdfMode::Inlined, TracerConfig{});
    if (path == "gpu_glsl") return std::make_unique<GpuGlslExecutor>(TracerConfig{});
    if (path == "gpu_ir")   return std::make_unique<GpuIrExecutor>();
    return nullptr;
}

const char* next_arg(int& i, int argc, char** argv, const char* def) {
    if (i + 1 < argc) return argv[++i];
    return def;
}

int run_master(int argc, char** argv) {
    std::string scene_path, decompose_s = "grid:64x64", scheduler_s = "pull",
                out_s = "dist.ppm";
    int port = 53900, n_workers = 1, W = 400, H = 300, ssaa = 1;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--port")      port = std::atoi(next_arg(i, argc, argv, "53900"));
        else if (a == "--workers")   n_workers = std::atoi(next_arg(i, argc, argv, "1"));
        else if (a == "--decompose") decompose_s = next_arg(i, argc, argv, "grid:64x64");
        else if (a == "--scheduler") scheduler_s = next_arg(i, argc, argv, "pull");
        else if (a == "--width")     W = std::atoi(next_arg(i, argc, argv, "400"));
        else if (a == "--height")    H = std::atoi(next_arg(i, argc, argv, "300"));
        else if (a == "--ssaa")      ssaa = std::atoi(next_arg(i, argc, argv, "1"));
        else if (a == "--out")       out_s = next_arg(i, argc, argv, "dist.ppm");
        else if (a[0] != '-')        scene_path = a;
    }
    if (scene_path.empty()) { std::fprintf(stderr, "master: need a scene file\n"); return 1; }
    if (ssaa < 1) ssaa = 1;

    SceneGraph scene;
    try { scene = io::load_scene(scene_path, nullptr); }
    catch (const std::exception& e) { std::fprintf(stderr, "load: %s\n", e.what()); return 1; }

    const int rw = W * ssaa, rh = H * ssaa;

    // Decompose the (hi-res) frame into tiles.
    int tw = 64, th = 64;
    if (decompose_s.rfind("grid:", 0) == 0) {
        tw = std::atoi(decompose_s.c_str() + 5);
        const char* x = std::strchr(decompose_s.c_str() + 5, 'x');
        th = x ? std::atoi(x + 1) : tw;
    }
    GridDecompose dec(tw, th);
    auto tiles = dec.decompose(scene, rw, rh);

    std::unique_ptr<IScheduler> sched;
    if (scheduler_s == "push") sched = std::make_unique<PushScheduler>(tiles.size(), n_workers);
    else                       sched = std::make_unique<PullScheduler>(tiles.size());

    std::printf("dist_render master: %s  %dx%d", scene_path.c_str(), W, H);
    if (ssaa > 1) std::printf(" (render %dx%d, SSAA %dx)", rw, rh, ssaa);
    std::printf("\n  tiles=%zu  scheduler=%s  workers=%d\n",
                tiles.size(), scheduler_s.c_str(), n_workers);

    MasterConfig mc;
    mc.port = port; mc.n_workers = n_workers; mc.width = rw; mc.height = rh;
    // Embed texture pixels in the scene message: a remote worker rebuilds the
    // scene from this JSON alone and may have no access to the original
    // texture files (and procedural textures have no file at all).
    mc.scene_json = io::serialize_scene(scene, "", /*embed_textures=*/true);
    Master master(mc, tiles, *sched);

    auto t0 = std::chrono::steady_clock::now();
    auto res = master.run();
    double wall = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
    if (!res.ok) { std::fprintf(stderr, "master: %s\n", res.error.c_str()); return 1; }

    std::printf("  done: %d tiles in %.1f ms\n", res.tiles_done, wall);
    std::printf("  distribution:");
    for (std::size_t i = 0; i < res.per_worker_tiles.size(); ++i)
        std::printf(" worker%zu=%d", i, res.per_worker_tiles[i]);
    std::printf("\n");

    StitchMerge mrg;
    auto merged = mrg.merge(res.tiles, rw, rh);
    if (merged.image.empty()) { std::fprintf(stderr, "master: stitch failed\n"); return 1; }

    // Post-process: SSAA downsample on the assembled frame.
    std::vector<float> img = std::move(merged.image);
    int iw = rw, ih = rh;
    if (ssaa > 1) {
        post::BoxDownsampleSSAA ss(ssaa);
        post::Frame o = ss.apply(post::Frame(std::move(img), rw, rh));
        img = std::move(o.rgba); iw = o.w; ih = o.h;
    }
    write_ppm(out_s, img, iw, ih);
    std::printf("  wrote %s\n", out_s.c_str());
    return 0;
}

int run_worker(int argc, char** argv) {
    std::string host = "127.0.0.1", paths = "cpu_ir";
    int port = 53900, W = 400, H = 300;
    double retry = 0.0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--host")  host = next_arg(i, argc, argv, "127.0.0.1");
        else if (a == "--port")  port = std::atoi(next_arg(i, argc, argv, "53900"));
        else if (a == "--paths") paths = next_arg(i, argc, argv, "cpu_ir");
        else if (a == "--width") W = std::atoi(next_arg(i, argc, argv, "400"));
        else if (a == "--height")H = std::atoi(next_arg(i, argc, argv, "300"));
        else if (a == "--retry") retry = std::atof(next_arg(i, argc, argv, "10"));
    }
    // A worker currently runs exactly one path. --paths takes the common
    // comma-separated form for consistency with the other tools, but a worker
    // must be given exactly one; reject a list so the limitation is explicit.
    if (paths.find(',') != std::string::npos) {
        std::fprintf(stderr,
            "worker: --paths must name exactly one path (got '%s'); "
            "run one worker per path\n", paths.c_str());
        return 1;
    }
    const std::string& path = paths;
    std::printf("dist_render worker: %s:%d  path=%s\n", host.c_str(), port, path.c_str());

    WorkerConfig wc;
    wc.host = host; wc.port = port; wc.width = W; wc.height = H;
    wc.retry_secs = retry;
    Worker w(wc, [path] { return make_executor(path); });
    auto r = w.run();
    if (!r) { std::fprintf(stderr, "worker: %s\n", r.error().c_str()); return 1; }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help"
                 || std::string(argv[1]) == "-h") {
        std::fprintf(argc < 2 ? stderr : stdout,
            "usage:\n"
            "  dist_render --master scene.json --port P --workers N \\\n"
            "              --decompose grid:WxH --width W --height H \\\n"
            "              [--scheduler pull|push] [--ssaa N] [--out file.ppm]\n"
            "  dist_render --worker --host H --port P --paths cpu_ir|gpu_glsl|gpu_ir \\\n"
            "              --width W --height H\n");
        return argc < 2 ? 1 : 0;
    }
    std::string role = argv[1];
    if (role == "--master") return run_master(argc, argv);
    if (role == "--worker") return run_worker(argc, argv);
    std::fprintf(stderr, "first arg must be --master or --worker\n");
    return 1;
}
