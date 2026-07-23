// Host-side CPU sphere tracer over a JIT-compiled scene SDF.
//
// Three march strategies, all sharing the same camera/hit conventions:
//
//   trace_scalar   — one ray at a time, step = safety * f(p).
//   trace_simd     — 8 rays per packet through scene_sdf_simd; a lane that has
//                    hit or escaped is masked out but still evaluated (the SIMD
//                    fn has no mask), so the packet runs until every lane is done.
//   trace_simd_lip — same packet march, but the global `safety` is replaced by a
//                    per-region 1/L taken from a coarse Lipschitz grid. Where the
//                    field is gentle (L < 1/safety) the step is *larger* than the
//                    conservative global one, and it stays sound because
//                    |f(a)-f(b)| <= L*|a-b| bounds the distance to the surface by
//                    f(p)/L.
//
// The grid is built once per scene from finite differences of the scalar SDF.
#pragma once
#include "core/compiler/compile_sdf.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace frep::render {

struct Camera {
    float ox = 0, oy = 0, oz = 3.2f;      // eye
    float dx = 0, dy = 0, dz = -1;        // forward
    float rx = 1, ry = 0, rz = 0;         // right
    float ux = 0, uy = 1, uz = 0;         // up
    float fov_scale = 0.577f;             // tan(fov/2)
};

struct MarchCfg {
    int   max_steps = 256;
    float t_min     = 1e-3f;
    float t_max     = 12.0f;
    float epsilon   = 1e-4f;
    float safety    = 0.85f;              // global conservative step scale
};

// Ray direction for pixel (px,py) of a (w,h) image.
inline void ray_dir(const Camera& c, int px, int py, int w, int h,
                    float& dx, float& dy, float& dz) {
    const float aspect = float(w) / float(h);
    const float sx = (2.0f * (px + 0.5f) / w - 1.0f) * aspect * c.fov_scale;
    const float sy = (1.0f - 2.0f * (py + 0.5f) / h) * c.fov_scale;
    dx = c.dx + sx * c.rx + sy * c.ux;
    dy = c.dy + sx * c.ry + sy * c.uy;
    dz = c.dz + sx * c.rz + sy * c.uz;
    const float inv = 1.0f / std::sqrt(dx*dx + dy*dy + dz*dz);
    dx *= inv; dy *= inv; dz *= inv;
}

// ── Coarse Lipschitz grid ───────────────────────────────────────────────────
// Cell (i,j,k) stores an upper bound on |grad f| over that cell, estimated by
// central differences at the cell centre and its 6 face midpoints, scaled by a
// safety margin. A ray inside the cell may step f(p)/L_cell.
//
// The bound is a *sample-based estimate*, not a proof: the same status as the
// stock safety_factor, which is likewise a hand-tuned constant.
struct LipGrid {
    int   n = 0;
    float lo = 0, hi = 0;
    std::vector<float> inv_l;             // 1 / L_cell, clamped to [0, 1]

    float cell_inv_l(float x, float y, float z) const {
        const float s = (n) / (hi - lo);
        int i = std::clamp(int((x - lo) * s), 0, n - 1);
        int j = std::clamp(int((y - lo) * s), 0, n - 1);
        int k = std::clamp(int((z - lo) * s), 0, n - 1);
        return inv_l[(size_t(i) * n + j) * n + k];
    }

    // Largest gradient magnitude seen. A true SDF gives ~1; a raw implicit
    // (gyroid, gears) gives much more, and only then does the per-region step
    // pay for itself — otherwise the lookup is pure overhead.
    float max_l() const {
        float m = 0;
        for (float v : inv_l) m = std::max(m, 1.0f / std::max(v, 1e-6f));
        return m;
    }
    bool worth_using(float safety) const { return max_l() > 1.0f / safety; }
};

inline LipGrid build_lip_grid(jit::SceneSdfFn fn, int n, float lo, float hi,
                              float margin = 1.15f) {
    LipGrid g; g.n = n; g.lo = lo; g.hi = hi;
    g.inv_l.resize(size_t(n) * n * n, 1.0f);
    const float cs = (hi - lo) / n;       // cell size
    const float h  = 0.25f * cs;          // difference step
    auto grad = [&](float x, float y, float z) {
        float gx = (fn(x+h,y,z) - fn(x-h,y,z)) / (2*h);
        float gy = (fn(x,y+h,z) - fn(x,y-h,z)) / (2*h);
        float gz = (fn(x,y,z+h) - fn(x,y,z-h)) / (2*h);
        return std::sqrt(gx*gx + gy*gy + gz*gz);
    };
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k) {
        float cx = lo + (i + 0.5f) * cs;
        float cy = lo + (j + 0.5f) * cs;
        float cz = lo + (k + 0.5f) * cs;
        float d  = 0.5f * cs;
        float L = grad(cx, cy, cz);
        L = std::max(L, grad(cx-d, cy, cz)); L = std::max(L, grad(cx+d, cy, cz));
        L = std::max(L, grad(cx, cy-d, cz)); L = std::max(L, grad(cx, cy+d, cz));
        L = std::max(L, grad(cx, cy, cz-d)); L = std::max(L, grad(cx, cy, cz+d));
        L = std::max(L * margin, 1e-3f);
        g.inv_l[(size_t(i) * n + j) * n + k] = std::min(1.0f / L, 1.0f);
    }
    return g;
}

// ── Tracers ─────────────────────────────────────────────────────────────────
// depth[i] = hit distance, or -1 when the ray escaped. `steps` accumulates the
// total SDF evaluations so callers can report work, not just wall time.

inline void trace_scalar(jit::SceneSdfFn fn, const Camera& c, const MarchCfg& m,
                         int w, int h, std::vector<float>& depth, long& steps) {
    depth.assign(size_t(w) * h, -1.0f);
    steps = 0;
    for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
        float dx, dy, dz; ray_dir(c, px, py, w, h, dx, dy, dz);
        float t = m.t_min;
        for (int s = 0; s < m.max_steps && t < m.t_max; ++s) {
            float f = fn(c.ox + dx*t, c.oy + dy*t, c.oz + dz*t);
            ++steps;
            if (f < m.epsilon) { depth[size_t(py) * w + px] = t; break; }
            t += m.safety * f;
        }
    }
}

// Packet march. `inv_l` non-null switches the step scale to the per-region 1/L.
inline void trace_simd(jit::SceneSdfSimdFn vfn, unsigned W, const Camera& c,
                       const MarchCfg& m, int w, int h,
                       std::vector<float>& depth, long& steps,
                       const LipGrid* lip = nullptr) {
    depth.assign(size_t(w) * h, -1.0f);
    steps = 0;
    std::vector<float> X(W), Y(W), Z(W), O(W), T(W), DX(W), DY(W), DZ(W);
    std::vector<uint8_t> live(W);

    const long total = long(w) * h;
    for (long base = 0; base < total; base += W) {
        const unsigned lanes = unsigned(std::min<long>(W, total - base));
        for (unsigned l = 0; l < W; ++l) {
            const long idx = base + std::min<long>(l, lanes - 1);   // clamp tail
            ray_dir(c, int(idx % w), int(idx / w), w, h, DX[l], DY[l], DZ[l]);
            T[l] = m.t_min;
            live[l] = (l < lanes);
        }
        for (int s = 0; s < m.max_steps; ++s) {
            bool any = false;
            for (unsigned l = 0; l < W; ++l) {
                X[l] = c.ox + DX[l]*T[l];
                Y[l] = c.oy + DY[l]*T[l];
                Z[l] = c.oz + DZ[l]*T[l];
                any |= live[l];
            }
            if (!any) break;
            vfn(X.data(), Y.data(), Z.data(), O.data());
            steps += lanes;
            for (unsigned l = 0; l < W; ++l) {
                if (!live[l]) continue;
                const float f = O[l];
                if (f < m.epsilon) {
                    depth[size_t(base + l)] = T[l];
                    live[l] = 0;
                    continue;
                }
                const float scale = lip ? lip->cell_inv_l(X[l], Y[l], Z[l]) : m.safety;
                T[l] += scale * f;
                if (T[l] >= m.t_max) live[l] = 0;
            }
        }
    }
}

} // namespace frep::render
