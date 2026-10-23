#pragma once
// core/frep/scene_bounds.hpp
//
// The one box every path clips its march to. The union of the visible
// objects' AABBs, widened by 5% of its largest side (at least 0.005), and
// only when every box is finite — AABB::infinite() is ±1e9, which counts as
// unbounded. The IR codegen, the GLSL compute emitter and the RTX
// whole-scene acceleration structure all take their box from here, so a ray
// starts its march at the same t on every path; a start that differs by a
// box face converges to a hit a fraction of epsilon apart, which on a crease
// is enough to flip which branch owns the normal.
#include "core/frep/scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace frep {

// False when the scene is unbounded (or empty) or FREP_BBOX_CLIP=0 is set;
// then `box` is untouched and the caller marches from the near plane.
inline bool scene_clip_box(const SceneGraph& scene, FRepNode::AABB& box) {
    static const bool clip_env_off = [] {
        const char* s = std::getenv("FREP_BBOX_CLIP");
        return s && s[0] == '0';
    }();
    if (clip_env_off) return false;
    auto finite = [](float v) { return std::isfinite(v) && std::fabs(v) < 1e8f; };
    FRepNode::AABB acc{};
    bool any = false;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        FRepNode::AABB b = obj.geometry->aabb();
        if (!(finite(b.min_x) && finite(b.min_y) && finite(b.min_z) &&
              finite(b.max_x) && finite(b.max_y) && finite(b.max_z)))
            return false;
        acc = any ? FRepNode::AABB::merge(acc, b) : b;
        any = true;
    }
    if (!any) return false;
    const float m = 0.05f * std::max({acc.max_x - acc.min_x, acc.max_y - acc.min_y,
                                      acc.max_z - acc.min_z, 0.1f});
    acc.min_x -= m; acc.min_y -= m; acc.min_z -= m;
    acc.max_x += m; acc.max_y += m; acc.max_z += m;
    box = acc;
    return true;
}

} // namespace frep
