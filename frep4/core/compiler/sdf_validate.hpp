#pragma once
// core/compiler/sdf_validate.hpp
//
// Smart-import guard: check whether a scene's field is actually a usable signed
// distance field before the renderer trusts it. Imported models (e.g. HyperFun
// sources converted to a single CustomExpr) frequently are NOT valid SDFs — a
// field built from atan2 / division / unclamped roots can evaluate to NaN/inf
// over part of the volume, or violate the Lipschitz-1 property that sphere
// tracing relies on (a step of `d` never oversteps the surface). Such a field
// renders as holes, spikes, or a black screen.
//
// validate_sdf JITs the scene SDF, samples it on a grid, and reports:
//   * finite   — no NaN/inf anywhere sampled (a hard requirement),
//   * lipschitz — an estimate of max |grad f| by finite differences over the
//                 grid; ~1 means a true metric distance field, >1 means the
//                 renderer must clamp its step (safety_factor < 1), and a large
//                 value means the field is far from metric (slow, fragile),
//   * metric   — lipschitz within a small tolerance of 1.
//
// It is advisory: the caller decides whether to reject, repair, or just render
// with a conservative step. The point is that "smart import" can *tell* rather
// than silently produce a broken render.

#include "core/compiler/compile_sdf.hpp"
#include "core/frep/scene.hpp"

#include <cmath>
#include <string>

namespace frep::jit {

struct SdfReport {
    bool        finite    = false;  // no NaN/inf in the sampled volume
    double      nan_frac  = 0.0;    // fraction of samples that were NaN/inf
    double      lipschitz = 0.0;    // estimated max |grad f| over the grid
    bool        metric    = false;  // lipschitz ~ 1 (a true distance field)
    std::string note;               // human-readable verdict

    bool usable_sdf() const { return finite; }  // hard requirement for tracing
};

// Sample the scene SDF on an N^3 grid over its bounds (padded) and judge it.
inline SdfReport validate_sdf(const SceneGraph& scene, int N = 24, float pad = 0.5f) {
    SdfReport r;

    auto compiled = compile_scene_sdf(scene);
    if (!compiled) { r.note = "compile failed: " + compiled.error(); return r; }
    auto fn = compiled->fn;

    // Scene bounds (fall back to a default box for infinite/unknown bounds).
    FRepNode::AABB b{-2, -2, -2, 2, 2, 2};
    bool first = true;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        auto ob = obj.geometry->aabb();
        bool inf = !std::isfinite(ob.min_x) || !std::isfinite(ob.max_x);
        if (inf) continue;
        if (first) { b = ob; first = false; }
        else b = FRepNode::AABB::merge(b, ob);
    }
    b.min_x -= pad; b.min_y -= pad; b.min_z -= pad;
    b.max_x += pad; b.max_y += pad; b.max_z += pad;

    const float dx = (b.max_x - b.min_x) / (N - 1);
    const float dy = (b.max_y - b.min_y) / (N - 1);
    const float dz = (b.max_z - b.min_z) / (N - 1);
    const float cell = std::min({dx, dy, dz});

    long total = 0, bad = 0;
    double max_grad = 0.0;
    // Cache one row along x to estimate |df/dx| cheaply; also probe y/z steps.
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            float x = b.min_x + dx * i, y = b.min_y + dy * j, z = b.min_z + dz * k;
            float f = fn(x, y, z);
            ++total;
            if (!std::isfinite(f)) { ++bad; continue; }
            // Finite-difference gradient magnitude vs the +x/+y/+z neighbours.
            if (i + 1 < N) {
                float fx = fn(x + dx, y, z);
                if (std::isfinite(fx)) max_grad = std::max(max_grad, double(std::fabs(fx - f) / dx));
            }
            if (j + 1 < N) {
                float fy = fn(x, y + dy, z);
                if (std::isfinite(fy)) max_grad = std::max(max_grad, double(std::fabs(fy - f) / dy));
            }
            if (k + 1 < N) {
                float fz = fn(x, y, z + dz);
                if (std::isfinite(fz)) max_grad = std::max(max_grad, double(std::fabs(fz - f) / dz));
            }
        }
    (void)cell;

    r.nan_frac  = total ? double(bad) / total : 1.0;
    r.finite    = (bad == 0);
    r.lipschitz = max_grad;
    r.metric    = r.finite && max_grad < 1.3;   // ~1 with finite-diff slack

    if (!r.finite)
        r.note = "NOT a usable SDF: " + std::to_string(int(r.nan_frac * 100)) +
                 "% of the volume is NaN/inf (division/atan2/unclamped root?)";
    else if (r.metric)
        r.note = "metric SDF (Lipschitz ~1): sphere tracing at full step is safe";
    else
        r.note = "finite but non-metric (Lipschitz ~" +
                 std::to_string(r.lipschitz).substr(0, 4) +
                 "): renderable, but needs a reduced march step";
    return r;
}

}  // namespace frep::jit
