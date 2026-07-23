#pragma once
// core/tracer/tile_scheduler.hpp
//
// Distributes the render across tiles between threads using std::jthread + an atomic counter.

#include "core/compiler/jit_engine.hpp"
#include "core/frep/scene.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace frep {

struct RenderParams {
    int width       = 800;
    int height      = 600;
    int tile_size   = 64;
    int num_threads = 0;

    // Super-sampling anti-aliasing factor. 1 = no AA; 2 = 4x rays per pixel
    // (2x2 grid), 3 = 9x rays, etc. Internally the scheduler renders the
    // scene at width*ssaa x height*ssaa, then box-filters down to the
    // requested resolution. Quadratic cost in ssaa.
    int ssaa        = 1;

    // Optional render region (in output-pixel coordinates): when width/
    // height of the region are > 0, only this sub-rectangle is rendered and
    // written; pixels outside are left untouched. {0,0,0,0} (the default)
    // means the whole frame. Lets a caller render just one tile of the
    // image — used by the multi-path executor so a CPU path can render
    // exactly its assigned region rather than the whole frame and crop.
    int region_x0 = 0, region_y0 = 0, region_x1 = 0, region_y1 = 0;

    bool has_region() const { return region_x1 > region_x0 && region_y1 > region_y0; }
};

class TileScheduler {
public:
    // Renders `cam` into `out_rgba` using the JIT'd render_tile.
    //
    // `params` is an optional float buffer for Incremental compilation
    // mode — the buffer's contents are loaded by render_tile in place of
    // node-parameter constants. Pass nullptr in Constant mode (the
    // buffer is unused in that case; render_tile does not dereference it).
    static void render(RenderTileFn        fn,
                       float*              out_rgba,
                       const Camera&       cam,
                       const RenderParams& p_in,
                       float*              params = nullptr)
    {
        // SSAA: render at higher resolution, then box-filter down.
        const int ssaa = std::max(1, p_in.ssaa);
        const int W_render = p_in.width  * ssaa;
        const int H_render = p_in.height * ssaa;

        // Direct path: no AA → render straight into the caller's buffer.
        // Supersampled path: render to a scratch buffer at W_render x H_render,
        // then average each ssaa x ssaa block into one output pixel.
        std::vector<float> scratch;
        float* render_target = out_rgba;
        if (ssaa > 1) {
            scratch.assign(static_cast<size_t>(W_render) * H_render * 4, 0.0f);
            render_target = scratch.data();
        }

        RenderParams p = p_in;
        p.width  = W_render;
        p.height = H_render;
        int n_threads = p.num_threads > 0
            ? p.num_threads
            : static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;

        // ── Camera basis ────────────────────────────────────────────────────
        auto v3sub = [](auto a, auto b) {
            return std::array<float,3>{a[0]-b[0], a[1]-b[1], a[2]-b[2]};
        };
        auto v3norm = [](std::array<float,3> v) {
            float l = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (l < 1e-8f) return std::array<float,3>{0.0f, 0.0f, -1.0f};
            return std::array<float,3>{v[0]/l, v[1]/l, v[2]/l};
        };
        auto v3cross = [](auto a, auto b) {
            return std::array<float,3>{
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            };
        };

        auto fwd   = v3norm(v3sub(cam.target, cam.position));
        auto right = v3norm(v3cross(fwd, cam.up));
        auto up    = v3cross(right, fwd);
        // Projection-aware scale:
        //   perspective  → positive: half the height of the view frustum at unit distance
        //   orthographic → negative: -half the view-rectangle height in world units
        // The JIT-emitted render_tile distinguishes the two by the sign.
        float view_scale = (cam.projection == Camera::Projection::Orthographic)
            ? -0.5f * cam.ortho_size
            :  std::tan(cam.fov_deg * 3.14159265f / 360.0f);

        struct Tile { int x, y, w, h; };
        std::vector<Tile> tiles;
        // Region bounds in render-space (scaled by ssaa). Default = whole
        // frame. Tiles are clipped to this rectangle so only the requested
        // sub-region is computed and written.
        int rx0 = 0, ry0 = 0, rx1 = p.width, ry1 = p.height;
        if (p_in.has_region()) {
            rx0 = std::max(0, p_in.region_x0 * ssaa);
            ry0 = std::max(0, p_in.region_y0 * ssaa);
            rx1 = std::min(p.width,  p_in.region_x1 * ssaa);
            ry1 = std::min(p.height, p_in.region_y1 * ssaa);
        }
        for (int ty = ry0; ty < ry1; ty += p.tile_size)
            for (int tx = rx0; tx < rx1; tx += p.tile_size)
                tiles.push_back({
                    tx, ty,
                    std::min(p.tile_size, rx1 - tx),
                    std::min(p.tile_size, ry1 - ty)
                });

        std::atomic<int> next{0};
        const int total = static_cast<int>(tiles.size());

        auto worker = [&]() {
            while (true) {
                int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= total) break;
                const auto& t = tiles[i];
                fn(render_target,
                   t.x, t.y, t.w, t.h, p.width, p.height,
                   cam.position[0], cam.position[1], cam.position[2],
                   fwd[0],   fwd[1],   fwd[2],
                   right[0], right[1], right[2],
                   up[0],    up[1],    up[2],
                   view_scale,
                   params);
            }
        };

        {
            std::vector<std::jthread> ts;
            ts.reserve(n_threads);
            for (int i = 0; i < n_threads; ++i) ts.emplace_back(worker);
        }  // jthreads join here.

        // ── Downsample if SSAA was active ────────────────────────────────────
        if (ssaa > 1) {
            const int W = p_in.width;
            const int H = p_in.height;
            const float inv_n = 1.0f / static_cast<float>(ssaa * ssaa);
            // Downsample only the region that was rendered (whole frame by
            // default). Region is in output coordinates.
            int dx0 = 0, dy0 = 0, dx1 = W, dy1 = H;
            if (p_in.has_region()) {
                dx0 = std::max(0, p_in.region_x0); dy0 = std::max(0, p_in.region_y0);
                dx1 = std::min(W, p_in.region_x1); dy1 = std::min(H, p_in.region_y1);
            }
            for (int y = dy0; y < dy1; ++y) {
                for (int x = dx0; x < dx1; ++x) {
                    float r = 0, g = 0, b = 0, a = 0;
                    for (int sy = 0; sy < ssaa; ++sy) {
                        const int src_y = y * ssaa + sy;
                        const float* src_row =
                            render_target + (src_y * W_render + x * ssaa) * 4;
                        for (int sx = 0; sx < ssaa; ++sx) {
                            r += src_row[sx*4 + 0];
                            g += src_row[sx*4 + 1];
                            b += src_row[sx*4 + 2];
                            a += src_row[sx*4 + 3];
                        }
                    }
                    float* dst = out_rgba + (y * W + x) * 4;
                    dst[0] = r * inv_n;
                    dst[1] = g * inv_n;
                    dst[2] = b * inv_n;
                    dst[3] = a * inv_n;
                }
            }
        }
    }
};

} // namespace frep
