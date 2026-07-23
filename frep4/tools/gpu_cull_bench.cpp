// GPU render benchmark for the workgroup tile cull.
//
// Renders a scene twice on the Vulkan compute path — once with cull off, once
// with cull_slabs depth slabs — and reports median frame time plus the cull
// on/off speedup. Writes both images so the result can be eyeballed for
// correctness (they should be identical wherever the cull is sound).
//
//   gpu_cull_bench scene.json [res] [slabs] [lipschitz] [frames] [method]
//
// method: auto (default) | lipschitz | interval | off. auto lets the executor
// probe both and pick the tighter; the explicit names force one path so you can
// compare. Off matches slabs=0.
//
// Defaults: res 512, slabs 32, lipschitz 1.0, frames 30.
// lipschitz must upper-bound |grad f|; 1.0 holds for node-tree SDFs, but a raw
// implicit (gyroid, gears) needs its own value or the cull is unsound.
#include "core/exec/gpu_executor.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace frep;
using namespace frep::exec;

// Programmatic node-tree scenes (no JSON path exists for these — JSON scenes are
// single CustomExpr). They exercise the per-node interval cull of v4.38.0.
static SceneGraph build_node_scene(const std::string& name) {
    SceneGraph s;
    if (name == "csg") {
        auto d = std::make_shared<DifferenceNode>(
            std::make_shared<BoxNode>(0.9f, 0.9f, 0.9f, "b"),
            std::make_shared<SphereNode>(1.1f, "s"), "d");
        s.add_object(std::make_shared<TranslateNode>(d, 0.2f, 0.0f, -0.1f, "t"));
    } else if (name == "twist") {
        s.add_object(std::make_shared<TwistYNode>(
            std::make_shared<BoxNode>(0.4f, 1.1f, 0.4f, "b"), 1.5f, "tw"));
    } else if (name == "blend") {
        auto a = std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.6f,"a"), -0.4f,0,0,"ta");
        auto b = std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.6f,"b"),  0.4f,0,0,"tb");
        s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.3f, "su"));
    } else if (name == "bend") {
        s.add_object(std::make_shared<BendXYNode>(
            std::make_shared<BoxNode>(1.2f, 0.25f, 0.35f, "b"), 0.9f, "bend"));
    } else if (name == "taper") {
        s.add_object(std::make_shared<TaperYNode>(
            std::make_shared<BoxNode>(0.5f, 1.2f, 0.5f, "b"), 0.3f, 2.4f, "taper"));
    } else {
        std::fprintf(stderr, "unknown node scene '%s' (use csg|twist|blend)\n", name.c_str());
    }
    return s;
}

static void write_ppm(const std::string& path, const std::vector<float>& rgba, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = std::clamp(rgba[size_t(i) * 4 + c], 0.0f, 1.0f);
            f.put((unsigned char)(v * 255.0f + 0.5f));
        }
}

static double bench(GpuGlslExecutor& ex, const SceneGraph& s, int R, int frames,
                    std::vector<float>& last_rgba, double& compile_ms, std::string& err) {
    Tile full{0, 0, R, R};
    auto w = ex.render(s, R, R, full);
    if (!w.ok) { err = w.error; return -1; }
    compile_ms = w.compile_ms;
    std::vector<double> t;
    for (int i = 0; i < frames; ++i) {
        auto r = ex.render(s, R, R, full);
        if (!r.ok) { err = r.error; return -1; }
        t.push_back(r.render_ms);
        if (i == frames - 1) last_rgba = std::move(r.rgba);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Interleaved A/B: alternate off/on every frame so GPU clock drift is common to
// both. Returns median-of-off, median-of-on, and the median per-pair ratio
// (off_i / on_i), which cancels common-mode variance — the trustworthy number
// when the frame time is noisy relative to the cull's effect.
static bool bench_ab(GpuGlslExecutor& off, GpuGlslExecutor& on,
                     const SceneGraph& s, int R, int frames,
                     std::vector<float>& img_off, std::vector<float>& img_on,
                     double& m_off, double& m_on, double& m_ratio,
                     double& c_off, double& c_on, std::string& err) {
    Tile full{0, 0, R, R};
    auto w0 = off.render(s, R, R, full); if (!w0.ok) { err = w0.error; return false; }
    auto w1 = on.render(s, R, R, full);  if (!w1.ok) { err = w1.error; return false; }
    c_off = w0.compile_ms; c_on = w1.compile_ms;
    std::vector<double> to, tn, tr;
    for (int i = 0; i < frames; ++i) {
        auto ro = off.render(s, R, R, full); if (!ro.ok) { err = ro.error; return false; }
        auto rn = on.render(s, R, R, full);  if (!rn.ok) { err = rn.error; return false; }
        to.push_back(ro.render_ms);
        tn.push_back(rn.render_ms);
        tr.push_back(ro.render_ms / rn.render_ms);
        if (i == frames - 1) { img_off = std::move(ro.rgba); img_on = std::move(rn.rgba); }
    }
    std::sort(to.begin(), to.end()); std::sort(tn.begin(), tn.end()); std::sort(tr.begin(), tr.end());
    m_off = to[to.size()/2]; m_on = tn[tn.size()/2]; m_ratio = tr[tr.size()/2];
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s scene.json [res] [slabs] [lipschitz] [frames]\n", argv[0]); return 1; }
    const int   R      = argc > 2 ? std::atoi(argv[2]) : 512;
    const int   SLABS  = argc > 3 ? std::atoi(argv[3]) : 32;
    const float LIP    = argc > 4 ? float(std::atof(argv[4])) : 1.0f;
    const int   FRAMES = argc > 5 ? std::atoi(argv[5]) : 30;
    const char* METHOD = argc > 6 ? argv[6] : "auto";

    if (!gpu::VulkanCtx::available()) {
        std::fprintf(stderr, "no Vulkan device available on this host\n");
        return 3;
    }

    // The scene argument is a .json path (a CustomExpr scene) or one of the
    // built-in programmatic node trees, given as "node:<name>", which exercise
    // the per-node interval cull that JSON CustomExpr scenes don't reach:
    //   node:csg     box(0.9) - sphere(1.1), translated   (metric node tree)
    //   node:twist   twisted box(0.4,1.1,0.4, k=1.5)      (non-metric)
    //   node:blend   smooth-union of two spheres           (metric)
    SceneGraph s;
    if (!std::strncmp(argv[1], "node:", 5)) {
        s = build_node_scene(argv[1] + 5);
    } else {
        s = io::load_scene(argv[1]);
    }

    TracerConfig off;  off.cull_slabs = 0;
    TracerConfig on;   on.cull_slabs = SLABS; on.cull_lipschitz = LIP;
    using CM = TracerConfig::CullMethod;
    if      (!std::strcmp(METHOD, "lipschitz")) on.cull_method = CM::Lipschitz;
    else if (!std::strcmp(METHOD, "interval"))  on.cull_method = CM::Interval;
    else if (!std::strcmp(METHOD, "off"))       on.cull_method = CM::Off;
    else                                         on.cull_method = CM::Auto;

    GpuGlslExecutor ex_off(off), ex_on(on);
    std::vector<float> img_off, img_on;
    double c_off = 0, c_on = 0, m_off = 0, m_on = 0, m_ratio = 0; std::string err;

    if (!bench_ab(ex_off, ex_on, s, R, FRAMES, img_off, img_on,
                  m_off, m_on, m_ratio, c_off, c_on, err)) {
        std::fprintf(stderr, "render failed: %s\n", err.c_str());
        return 2;
    }

    // Pixel difference (max channel delta) as a coarse correctness signal.
    double max_diff = 0; long diff_px = 0;
    for (size_t i = 0; i < img_off.size(); ++i) {
        double d = std::abs(double(img_off[i]) - double(img_on[i]));
        if (d > max_diff) max_diff = d;
        if ((i % 4) < 3 && d > 1.0/255.0 && (i % 4) == 0) ++diff_px;
    }

    std::string base = argv[1];
    for (char& ch : base) if (ch == ':') ch = '_';
    if (auto p = base.find_last_of('/'); p != std::string::npos) base = base.substr(p + 1);
    base = base.substr(0, base.find(".json"));
    write_ppm(base + "_cull_off.ppm", img_off, R, R);
    write_ppm(base + "_cull_on.ppm",  img_on,  R, R);

    std::printf("scene=%s res=%d slabs=%d lipschitz=%.2f frames=%d method=%s (interleaved A/B)\n",
                argv[1], R, SLABS, LIP, FRAMES, METHOD);
    std::printf("  cull off : %8.3f ms/frame median  (compile %.1f ms)\n", m_off, c_off);
    std::printf("  cull on  : %8.3f ms/frame median  (compile %.1f ms)\n", m_on, c_on);
    std::printf("  speedup  : %.2fx  (median of per-frame-pair ratios; cancels clock drift)\n", m_ratio);
    std::printf("  max pixel diff = %.4f   (wrote %s_cull_off.ppm / _cull_on.ppm)\n",
                max_diff, base.c_str());
    if (max_diff > 0.02)
        std::printf("  NOTE: images differ. This scene's field is not %.1f-Lipschitz; "
                    "raise lipschitz (a true implicit like the gyroid needs its max |grad f|, "
                    "~6.5) or the cull removes real surface.\n", LIP);
    return 0;
}
