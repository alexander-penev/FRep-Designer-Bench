// Loads the user's horizon.json at its real camera and renders it with the
// CPU-IR path at several max_steps values. A high max_steps render (4096) is the
// hole-free reference; a "hole" is a pixel that differs from the reference
// because a grazing ray exhausted the step cap and escaped instead of hitting
// the floor/silhouette. Writes a PPM per setting so the holes can be viewed.

#include "core/io/scene_io.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/codegen.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

using namespace frep;
static inline int u8(float f){int v=int(f*255.0f+0.5f);return v<0?0:(v>255?255:v);}

static bool render_at(const SceneGraph& s, int W, int H, int max_steps,
                      std::vector<std::uint8_t>& rgb) {
    TracerConfig cfg{};
    cfg.enable_shadows = false;
    cfg.enable_ao      = false;
    cfg.max_steps      = max_steps;
    exec::CpuIrExecutor ex(SceneCodegen::SceneSdfMode::Inlined, cfg);
    auto r = ex.render(s, W, H, exec::Tile{0, 0, W, H});
    if (r.rgba.empty()) return false;
    rgb.resize((size_t)W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        rgb[i*3+0] = u8(r.rgba[(size_t)i*4+0]);
        rgb[i*3+1] = u8(r.rgba[(size_t)i*4+1]);
        rgb[i*3+2] = u8(r.rgba[(size_t)i*4+2]);
    }
    return true;
}

static void write_ppm(const std::string& path, int W, int H,
                      const std::vector<std::uint8_t>& rgb) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
}

// Count pixels that differ from the reference by more than `thr` per channel
// (max abs). These are the rays the step cap couldn't resolve.
static int diff_holes(const std::vector<std::uint8_t>& a,
                      const std::vector<std::uint8_t>& ref, int thr) {
    int n = 0;
    for (size_t i = 0; i + 2 < a.size(); i += 3) {
        int dr = std::abs((int)a[i]   - (int)ref[i]);
        int dg = std::abs((int)a[i+1] - (int)ref[i+1]);
        int db = std::abs((int)a[i+2] - (int)ref[i+2]);
        if (std::max({dr, dg, db}) > thr) ++n;
    }
    return n;
}

int main(int argc, char** argv) {
    const char* scene_path = argc > 1 ? argv[1] : "/tmp/horizon.json";
    const int W = 640, H = 560;   // ~ the user's viewport aspect

    SceneGraph s = io::load_scene(scene_path);
    printf("loaded %s : %zu objects, camera pos(%.3f %.3f %.3f) target(%.3f %.3f %.3f) fov %.0f\n",
           scene_path, s.objects().size(),
           s.camera().position[0], s.camera().position[1], s.camera().position[2],
           s.camera().target[0], s.camera().target[1], s.camera().target[2],
           s.camera().fov_deg);

    // Reference: high step cap (hole-free).
    std::vector<std::uint8_t> ref;
    if (!render_at(s, W, H, 4096, ref)) { printf("reference render failed\n"); return 1; }
    write_ppm("/tmp/horizon_ref4096.ppm", W, H, ref);

    { TracerConfig d{}; printf("new TracerConfig default max_steps = %d\n\n", d.max_steps); }
    int steps[] = {192, 384, 512, 768};
    for (int ms : steps) {
        std::vector<std::uint8_t> img;
        if (!render_at(s, W, H, ms, img)) { printf("render %d failed\n", ms); continue; }
        char name[64]; std::snprintf(name, sizeof(name), "/tmp/horizon_%d.ppm", ms);
        write_ppm(name, W, H, img);
        int h = diff_holes(img, ref, 24);
        printf("max_steps=%4d : %6d hole-pixels vs ref (%.3f%% of frame)  -> %s\n",
               ms, h, 100.0 * h / (W * H), name);
    }
    printf("\nreference (max_steps=4096): /tmp/horizon_ref4096.ppm\n");
    return 0;
}
