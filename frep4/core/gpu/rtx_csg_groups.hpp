// core/gpu/rtx_csg_groups.hpp
//
// CSG-aware scene partitioning for the GpuRtx broad-phase.
//
// A single-BLAS scene puts everything in one box, so the intersection shader
// evaluates the full O(N) flat-union scene_sdf at every march step — the RT
// cores do no useful broad-phase. Splitting the scene into CSG-independent
// groups, one BLAS each, so the RT cores cull groups a ray never enters and
// each intersection shader evaluates only its sub-tree.
//
// Splitting rule — only at hard unions:
//   Union (min) is separable: min(a,b) lets a ray test each side
//     independently and keep the nearer hit, which is exactly what BVH
//     broad-phase does. So we cut the tree at every UnionNode.
//   SmoothUnion (smin), Intersection (max), Difference (max(a,-b)) are NOT
//     separable — they blend / require both operands at a point — so a whole
//     such sub-tree stays in one group.
//
// The result is a list of groups; each group is an FRepNode sub-tree plus its
// world-space AABB. The single-BLAS path is just the degenerate case of
// one group (a scene with no top-level union).

#pragma once

#include "core/frep/node.hpp"

#include <vector>

namespace frep::gpu {

struct CsgGroup {
    FRepNode::Ptr   root;   // sub-tree evaluated by this group's intersection shader
    FRepNode::AABB  box;    // world-space bounds of the sub-tree
};

// Partition a scene root into CSG-independent groups by cutting at hard unions.
// Always returns at least one group (the whole tree if nothing splits).
std::vector<CsgGroup> partition_csg_groups(const FRepNode::Ptr& root);

}  // namespace frep::gpu
