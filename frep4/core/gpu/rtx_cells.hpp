// core/gpu/rtx_cells.hpp
//
// Shell cells: the boxes the RTX path's acceleration structure is built from
// when a scene is one field rather than many objects.
//
// With a single box the hardware has nothing to do -- every ray that enters
// the box runs the whole march in the intersection shader, exactly as the
// compute path does, and the ray-tracing pipeline only adds its per-ray cost
// (measured at about 0.13 ns per ray on an RTX 2080, which is a third of a
// sphere frame and nothing on the colonnade). What the traversal unit is good
// at is skipping space, so give it space to skip: subdivide the scene box as
// an octree, keep only the leaves whose interval bound of the field straddles
// zero (the cells the surface passes through), and build the bottom-level
// structure from those. A ray then enters only the cells that can contain the
// surface, in roughly front-to-back order, and the intersection shader marches
// within one cell at a time -- a background ray that misses every cell costs
// no field evaluation at all, and a ray that hits starts its march a cell away
// from the surface instead of a box away.
//
// Cells strictly inside the solid (upper bound below zero) are dropped too:
// the surface is not in them, and a ray reaches the solid only through a
// shell cell. The bound is the numeric interval evaluator the tile cull uses
// (core/compiler/node_interval.hpp), unions as interval minima; a node kind it
// does not know bounds as everything, so such a scene degrades to the uniform
// grid rather than to a wrong picture.

#pragma once

#include "core/frep/node.hpp"
#include "core/frep/scene.hpp"
#include "core/gpu/rtx_accel.hpp"

#include <cstddef>
#include <vector>

namespace frep::gpu {

struct CellStats {
    std::size_t evaluated  = 0;   // interval bounds taken
    std::size_t kept       = 0;   // leaves that straddle zero (the output)
    std::size_t culled_out = 0;   // cells entirely outside the solid
    std::size_t culled_in  = 0;   // cells entirely inside it
    int         depth      = 0;   // octree depth used (cells per axis = 2^depth)
    double      ms         = 0;   // wall time of the walk
};

// The shell cells of `scene`'s field over `box` at octree depth `depth`
// (2^depth cells per axis at the leaves). depth <= 0 returns the box itself.
// Never empty: a field with no zero set in the box, or one the evaluator
// cannot bound, comes back as the single box, so the caller always has a
// valid acceleration structure to build.
std::vector<RtAabb> shell_cells(const SceneGraph& scene, const FRepNode::AABB& box,
                                int depth, CellStats* stats = nullptr);

}  // namespace frep::gpu
