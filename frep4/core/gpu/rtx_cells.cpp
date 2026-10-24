// core/gpu/rtx_cells.cpp — see rtx_cells.hpp.

#include "core/gpu/rtx_cells.hpp"
#include "core/compiler/node_interval.hpp"

#include <chrono>

namespace frep::gpu {

namespace {

using jit::Iv;

// Interval of the scene's field over a box: the visible objects' bounds,
// joined as a union (the scene is their minimum). An object the evaluator
// cannot bound returns the whole line, which keeps every cell it touches.
Iv scene_interval(const std::vector<const FRepNode*>& geoms, Iv X, Iv Y, Iv Z) {
    Iv r{1e30f, 1e30f};
    bool first = true;
    for (const FRepNode* g : geoms) {
        Iv v = jit::node_interval(*g, X, Y, Z);
        if (first) { r = v; first = false; }
        else r = Iv{std::min(r.lo, v.lo), std::min(r.hi, v.hi)};
    }
    return r;
}

struct Walk {
    const std::vector<const FRepNode*>& geoms;
    std::vector<RtAabb>& out;
    CellStats& st;

    void visit(float x0, float y0, float z0, float x1, float y1, float z1, int depth) {
        Iv f = scene_interval(geoms, Iv{x0, x1}, Iv{y0, y1}, Iv{z0, z1});
        ++st.evaluated;
        if (f.lo > 0.0f) { ++st.culled_out; return; }
        if (f.hi < 0.0f) { ++st.culled_in;  return; }
        if (depth == 0) {
            out.push_back(RtAabb{{x0, y0, z0}, {x1, y1, z1}});
            ++st.kept;
            return;
        }
        const float xm = 0.5f * (x0 + x1), ym = 0.5f * (y0 + y1), zm = 0.5f * (z0 + z1);
        for (int c = 0; c < 8; ++c) {
            const bool hx = c & 1, hy = c & 2, hz = c & 4;
            visit(hx ? xm : x0, hy ? ym : y0, hz ? zm : z0,
                  hx ? x1 : xm, hy ? y1 : ym, hz ? z1 : zm, depth - 1);
        }
    }
};

}  // namespace

std::vector<RtAabb> shell_cells(const SceneGraph& scene, const FRepNode::AABB& box,
                                int depth, CellStats* stats) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    CellStats local;
    CellStats& st = stats ? *stats : local;
    st = CellStats{};
    st.depth = depth < 0 ? 0 : depth;

    const RtAabb whole{{box.min_x, box.min_y, box.min_z}, {box.max_x, box.max_y, box.max_z}};
    std::vector<RtAabb> out;
    if (st.depth == 0) {
        out.push_back(whole);
        st.kept = 1;
        st.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        return out;
    }

    std::vector<const FRepNode*> geoms;
    for (const auto& [id, obj] : scene.objects())
        if (obj.visible && obj.geometry) geoms.push_back(obj.geometry.get());

    Walk w{geoms, out, st};
    w.visit(box.min_x, box.min_y, box.min_z, box.max_x, box.max_y, box.max_z, st.depth);

    // Leave neighbouring cells a hair of overlap so a ray exiting one cell
    // through a face enters the next one without a gap of unmarched
    // parameter between them: the shader clips the march to the cell it is
    // in, and the two clips meet exactly only with coincident faces, which
    // float arithmetic on the ray-plane intersections does not quite give.
    const float side = (box.max_x - box.min_x) / float(1 << st.depth);
    const float eps = side * 1e-4f;
    for (RtAabb& c : out)
        for (int k = 0; k < 3; ++k) { c.min[k] -= eps; c.max[k] += eps; }

    if (out.empty()) { out.push_back(whole); st.kept = 1; }
    st.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return out;
}

}  // namespace frep::gpu
