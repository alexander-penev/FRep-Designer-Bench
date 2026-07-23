// Emits the tile-cull compute shader for a scene and measures what it buys.
//
// The GPU kernel walks, per screen tile, a set of depth slabs and keeps only the
// span [t0,t1] whose slabs may contain the surface. The render kernel then
// marches that span instead of [near,far]. This tool runs the identical logic on
// the CPU against the JIT'd interval SDF, so the numbers predict the GPU pass
// without needing a GPU:
//
//   skipped_volume  fraction of (tile x slab) cells the interval rejects
//   empty_tiles     tiles rejected outright (no march at all)
//   span_frac       mean kept depth span / [near,far], over non-empty tiles
//   wrong_tiles     tiles culled although a scalar march finds a hit (must be 0)
//
//   bench_tile_cull scene.json [res] [tile] [slabs] [lipschitz] [--dump]
//
// A positive [lipschitz] switches from interval arithmetic to the Lipschitz box
// rule f(box) subset [f(c)-L*r, f(c)+L*r] evaluated with the scalar SDF — which
// is exactly what the emitted render shader does per workgroup (cull_slabs /
// cull_lipschitz in TracerConfig). Interval mode instead mirrors the standalone
// tile_cull_shader in core/gpu/glsl_interval.hpp.
#include "core/compiler/compile_sdf.hpp"
#include "core/gpu/glsl_interval.hpp"
#include "core/io/scene_io.hpp"
#include "core/render/cpu_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace frep;

namespace {

struct Slab { float t0, t1; };

// AABB of the frustum slab [a,b] for tile (tx,ty) — mirrors slab_box() in GLSL.
void slab_box(const render::Camera& cam, int tx, int ty, int tile, int R,
              float a, float b, float* lo, float* hi) {
    lo[0]=lo[1]=lo[2] =  1e30f;
    hi[0]=hi[1]=hi[2] = -1e30f;
    for (int c = 0; c < 4; ++c) {
        int px = std::min((tx + (c & 1)) * tile, R - 1);
        int py = std::min((ty + ((c >> 1) & 1)) * tile, R - 1);
        float dx, dy, dz;
        render::ray_dir(cam, px, py, R, R, dx, dy, dz);
        for (int e = 0; e < 2; ++e) {
            float t = e ? b : a;
            float q[3] = {cam.ox + dx*t, cam.oy + dy*t, cam.oz + dz*t};
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], q[k]); hi[k] = std::max(hi[k], q[k]); }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s scene.json [res] [tile] [slabs] [--dump]\n", argv[0]); return 1; }
    const int R     = argc > 2 ? std::atoi(argv[2]) : 256;
    const int TILE  = argc > 3 ? std::atoi(argv[3]) : 8;
    const int SLABS = argc > 4 ? std::atoi(argv[4]) : 32;
    const float LIP = argc > 5 && argv[5][0] != '-' ? float(std::atof(argv[5])) : 0.0f;
    bool dump = false;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--dump")) dump = true;

    SceneGraph s = io::load_scene(argv[1]);

    if (dump) {
        const void* p = nullptr;
        for (auto& [id, obj] : s.objects()) if (obj.visible) p = obj.geometry->custom_expr_ast();
        if (!p) { std::fprintf(stderr, "--dump needs a CustomExpr scene\n"); return 2; }
        std::printf("%s", gpu::GlslIntervalEmitter::tile_cull_shader(
                        **static_cast<const expr::NodePtr*>(p), TILE, SLABS).c_str());
        return 0;
    }

    auto iv = jit::compile_scene_sdf_interval(s);
    if (!iv) { std::fprintf(stderr, "interval: %s\n", iv.error().c_str()); return 2; }
    auto sc = jit::compile_scene_sdf(s);
    if (!sc) { std::fprintf(stderr, "scalar: %s\n", sc.error().c_str()); return 2; }

    render::Camera cam;
    const float NEAR = 1e-3f, FAR = 12.0f;
    const int TX = (R + TILE - 1) / TILE, TY = TX;
    const float step = (FAR - NEAR) / SLABS;

    std::vector<Slab> span(size_t(TX) * TY);
    long cells_kept = 0;
    auto c0 = std::chrono::steady_clock::now();
    for (int ty = 0; ty < TY; ++ty)
    for (int tx = 0; tx < TX; ++tx) {
        float t0 = 1e30f, t1 = -1e30f;
        for (int i = 0; i < SLABS; ++i) {
            float a = NEAR + step * i, b = a + step;
            float lo[3], hi[3];
            slab_box(cam, tx, ty, TILE, R, a, b, lo, hi);
            float B[6] = {lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]}, O[2];
            if (LIP > 0) {                             // Lipschitz box rule (shader path)
                float cx=0.5f*(lo[0]+hi[0]), cy=0.5f*(lo[1]+hi[1]), cz=0.5f*(lo[2]+hi[2]);
                float hx=0.5f*(hi[0]-lo[0]), hy=0.5f*(hi[1]-lo[1]), hz=0.5f*(hi[2]-lo[2]);
                float rr = LIP * std::sqrt(hx*hx+hy*hy+hz*hz);
                float f  = sc->fn(cx,cy,cz);
                O[0] = f - rr; O[1] = f + rr;
            } else {
                iv->fn(B, O);                          // interval arithmetic
            }
            if (O[0] > 0 || O[1] < 0) continue;        // slab certainly empty
            t0 = std::min(t0, a); t1 = std::max(t1, b);
            ++cells_kept;
        }
        span[size_t(ty) * TX + tx] = {t0, t1};
    }
    auto c1 = std::chrono::steady_clock::now();
    double cull_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    const long cells = long(TX) * TY * SLABS;
    long empty_tiles = 0; double span_sum = 0;
    for (auto& sp : span) {
        if (sp.t1 < sp.t0) { ++empty_tiles; continue; }
        span_sum += (sp.t1 - sp.t0) / (FAR - NEAR);
    }

    // Correctness: a tile with an actual hit must not be culled, and the hit
    // depth must lie inside the kept span (one slab of slack for the march
    // epsilon).
    render::MarchCfg cfg;
    std::vector<float> depth; long steps = 0;
    render::trace_scalar(sc->fn, cam, cfg, R, R, depth, steps);
    long wrong_tiles = 0, hits_outside_span = 0;
    for (int ty = 0; ty < TY; ++ty)
    for (int tx = 0; tx < TX; ++tx) {
        const Slab& sp = span[size_t(ty) * TX + tx];
        bool any_hit = false;
        for (int py = ty*TILE; py < std::min((ty+1)*TILE, R) && !any_hit; ++py)
        for (int px = tx*TILE; px < std::min((tx+1)*TILE, R); ++px) {
            float d = depth[size_t(py)*R + px];
            if (d < 0) continue;
            any_hit = true;
            if (sp.t1 >= sp.t0 && (d < sp.t0 - step || d > sp.t1 + step)) ++hits_outside_span;
            break;
        }
        if (any_hit && sp.t1 < sp.t0) ++wrong_tiles;
    }

    std::printf("%-24s res=%d tile=%d slabs=%d %-12s| skipped=%.1f%% "
                "empty_tiles=%ld/%d span=%.3f cull=%.2fms "
                "wrong_tiles=%ld hits_outside=%ld\n",
                argv[1], R, TILE, SLABS,
                LIP > 0 ? ("lipschitz=" + std::to_string(LIP).substr(0,4)).c_str() : "interval",
                100.0 * double(cells - cells_kept) / cells,
                empty_tiles, TX*TY,
                span_sum / std::max<long>(1, TX*TY - empty_tiles),
                cull_ms, wrong_tiles, hits_outside_span);
    return 0;
}
