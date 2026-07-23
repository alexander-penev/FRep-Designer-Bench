// core/accel/guard_calibration.cpp

#include "core/accel/guard_calibration.hpp"

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/deformations.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace frep::accel {

using clk = std::chrono::steady_clock;

std::string host_cpu_id() {
    // Linux: first "model name" line from /proc/cpuinfo. Trim to something
    // filename/JSON-safe and short.
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find("model name");
        if (pos != std::string::npos) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string v = line.substr(colon + 1);
                // trim leading space
                std::size_t b = v.find_first_not_of(" \t");
                if (b != std::string::npos) v = v.substr(b);
                return v;
            }
        }
    }
    return "unknown-cpu";
}

std::string calibration_cache_path() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    std::string base;
    if (xdg && *xdg) base = xdg;
    else {
        const char* home = std::getenv("HOME");
        base = (home && *home) ? std::string(home) + "/.cache" : "/tmp";
    }
    return base + "/frep_guard_calibration.txt";
}

std::optional<GuardCalibration> load_calibration() {
    std::ifstream f(calibration_cache_path());
    if (!f) return std::nullopt;
    // Simple line format: "version 1\nthreshold N\ncpu <rest of line>\n"
    GuardCalibration c;
    std::string key;
    std::string cpu_line;
    int version = 0;
    std::string tag;
    f >> tag >> version;
    if (tag != "version" || version != 1) return std::nullopt;
    f >> tag >> c.node_threshold;
    if (tag != "threshold") return std::nullopt;
    f >> tag;
    if (tag != "cpu") return std::nullopt;
    std::getline(f, cpu_line);
    std::size_t b = cpu_line.find_first_not_of(" \t");
    c.cpu_id = (b == std::string::npos) ? "" : cpu_line.substr(b);
    if (c.cpu_id != host_cpu_id()) return std::nullopt;   // different machine
    c.valid = true;
    return c;
}

bool save_calibration(const GuardCalibration& c) {
    const std::string path = calibration_cache_path();
    // std::ofstream won't create missing parent directories (e.g. a
    // fresh ~/.cache), so the write would silently fail. Create them.
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    f << "version 1\n"
      << "threshold " << c.node_threshold << "\n"
      << "cpu " << c.cpu_id << "\n";
    return (bool)f;
}

namespace {

// Build a scene of `count` objects, each with controlled complexity:
//   complexity 1 → bare sphere (cheapest)
//   2 → translated box
//   higher → progressively deeper twist + smooth-union chains
// so we can sweep per-object node count and find where guarding wins.
SceneGraph make_complexity_scene(int count, int complexity) {
    SceneGraph s;
    int side = std::max(1, (int)std::ceil(std::cbrt((double)count)));
    float sp = 1.8f, off = (side - 1) * sp * 0.5f;
    int m = 0;
    for (int i = 0; i < side && m < count; ++i)
      for (int j = 0; j < side && m < count; ++j)
        for (int k = 0; k < side && m < count; ++k, ++m) {
            std::string id = std::to_string(m);
            FRepNode::Ptr g;
            if (complexity <= 1) {
                g = std::make_shared<SphereNode>(0.5f, "s" + id);
            } else {
                // Box, then add twist + smooth-union sphere layers to grow
                // the node count roughly linearly with `complexity`.
                g = std::make_shared<BoxNode>(0.4f, 0.5f, 0.4f, "b" + id);
                int layers = complexity - 1;
                for (int L = 0; L < layers; ++L) {
                    if (L % 2 == 0) {
                        g = std::make_shared<TwistYNode>(g, 1.5f, "tw" + id + "_" + std::to_string(L));
                    } else {
                        g = std::make_shared<SmoothUnionNode>(
                            g,
                            std::make_shared<TranslateNode>(
                                std::make_shared<SphereNode>(0.25f, "ss" + id + "_" + std::to_string(L)),
                                0.3f, 0.1f, 0, "st" + id + "_" + std::to_string(L)),
                            0.2f, "su" + id + "_" + std::to_string(L));
                    }
                }
            }
            s.add_object(std::make_shared<TranslateNode>(
                g, i*sp-off, j*sp-off + 1.0f, k*sp-off, "t" + id));
        }
    float view = off + side * sp;
    s.camera().position = {view, view*0.7f, view*1.2f};
    s.camera().target   = {0, 1.0f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{view, view*1.5f, view}, {1,1,1}, 1.0f});
    return s;
}

// Compile + render a scene in a given SDF mode; return render ms (best of
// 2) or -1 on failure. Tiny resolution keeps calibration fast.
double render_ms(const SceneGraph& s, SceneCodegen::SceneSdfMode mode,
                 int W, int H) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = false; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s, mode);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn = jit.load(std::move(mod), std::move(ctx));
    if (!fn) return -1;
    std::vector<float> px((std::size_t)W * H * 4);
    RenderParams rp; rp.width = W; rp.height = H;
    double best = 1e30;
    for (int r = 0; r < 2; ++r) {
        auto t = clk::now();
        TileScheduler::render(*fn, px.data(), s.camera(), rp);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t).count();
        best = ms < best ? ms : best;
    }
    return best;
}

// Average node count per object in a scene (excludes the auto floor/plane
// the generators don't add here).
double avg_nodes(const SceneGraph& s) {
    long total = 0; int n = 0;
    for (const auto& [id, obj] : s.objects()) {
        if (!obj.geometry) continue;
        total += node_count(*obj.geometry);
        ++n;
    }
    return n ? (double)total / n : 0.0;
}

} // namespace

GuardCalibration calibrate() {
    GuardCalibration cal;
    cal.cpu_id = host_cpu_id();
    auto t0 = clk::now();

    // Probe sizing. The guard's effect must clear measurement noise, so we
    // use enough objects and resolution that render times are well above
    // jitter, but small enough to keep calibration fast (well under a
    // second). 32 objects at 128×128 puts the inline baseline in the tens
    // of ms for the cheap cases — enough signal — while the early-exit
    // below avoids ever rendering the expensive high-complexity scenes
    // once a threshold is found.
    const int kObjects = 32;
    const int W = 128, H = 128;
    // Complexity sweep, lowest first so we stop at the first winning level
    // and never pay for the costly ones. cx=1 is a bare sphere (must NOT
    // win); each higher level adds a box/twist/smooth-union layer.
    const int complexities[] = {1, 2, 3, 4};

    // A clear, above-noise win: guarded at least 15% faster. Bare
    // primitives sit near 1.0 and must not qualify.
    constexpr double kWinRatio = 0.85;

    // Bare-sphere (cx=1) node count: the threshold must be strictly above
    // it, so jitter on the sphere case can never enable guarding for
    // simple scenes. We only need its node count, not a render.
    SceneGraph sphere_scene = make_complexity_scene(kObjects, 1);
    const double sphere_nodes = avg_nodes(sphere_scene);

    int threshold = kNeverGuard;
    for (int cx : complexities) {
        SceneGraph s = make_complexity_scene(kObjects, cx);
        const double nodes = avg_nodes(s);
        if (nodes <= sphere_nodes) continue;   // never guard at/below sphere
        const double inl = render_ms(s, SceneCodegen::SceneSdfMode::Inlined,  W, H);
        const double grd = render_ms(s, SceneCodegen::SceneSdfMode::Guarded, W, H);
        if (inl > 0 && grd > 0 && grd < inl * kWinRatio) {
            threshold = (int)std::floor(nodes);
            break;
        }
    }

    cal.node_threshold  = threshold;
    cal.calibration_ms  = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    cal.valid           = true;
    return cal;
}

GuardCalibration get_or_calibrate(bool force) {
    if (!force) {
        if (auto c = load_calibration()) return *c;
    }
    GuardCalibration c = calibrate();
    save_calibration(c);   // best-effort
    return c;
}

} // namespace frep::accel
