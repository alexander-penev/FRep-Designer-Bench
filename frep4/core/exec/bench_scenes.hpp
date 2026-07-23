// core/exec/bench_scenes.hpp
//
// Benchmark scenes for the GpuRtx broad-phase study. The headline
// measurement is a scaling curve: render a grid of N separated spheres at
// several N and compare the RT path (per-group BLAS, RT cores cull groups a
// ray misses) against the compute path (O(N) flat-union scene_sdf per march
// step). As N grows the compute path scales linearly; the RT broad-phase
// should scale far better.
//
// The spheres are laid out on a square grid in the z=0 plane, spaced so their
// bounding boxes don't overlap, so partition_csg_groups yields exactly N
// independent groups (one BLAS each). The whole grid is framed by the scene
// camera placed back along +z.

#pragma once

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"

#include <cmath>
#include <memory>
#include <string>

namespace frep::bench {

// Build a k×k grid of unit-ish spheres (k = ceil(sqrt(n)), first n filled) as a
// flat union. radius and gap are chosen so boxes are disjoint → n CSG groups.
inline SceneGraph make_sphere_grid(int n) {
    if (n < 1) n = 1;
    int k = (int)std::ceil(std::sqrt((double)n));
    const float r = 0.35f;     // sphere radius
    const float step = 1.0f;   // center-to-center spacing (> 2r → disjoint)
    const float origin = -0.5f * step * (k - 1);  // center the grid on 0

    FRepNode::Ptr root;
    int placed = 0;
    for (int gy = 0; gy < k && placed < n; ++gy) {
        for (int gx = 0; gx < k && placed < n; ++gx, ++placed) {
            float x = origin + gx * step;
            float y = origin + gy * step;
            auto sph = std::make_shared<SphereNode>(r, "s" + std::to_string(placed));
            FRepNode::Ptr obj = std::make_shared<TranslateNode>(sph, x, y, 0.0f);
            root = root ? std::static_pointer_cast<FRepNode>(
                              std::make_shared<UnionNode>(root, obj))
                        : obj;
        }
    }

    SceneGraph s;
    s.add_object(root);
    // Camera back along +z, looking at the grid center, far enough to frame k.
    float dist = 2.0f + step * k;
    s.camera().position[0] = 0.0f;
    s.camera().position[1] = 0.0f;
    s.camera().position[2] = dist;
    s.camera().target[0] = 0.0f;
    s.camera().target[1] = 0.0f;
    s.camera().target[2] = 0.0f;
    return s;
}

}  // namespace frep::bench
