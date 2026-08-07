// core/gpu/rtx_instances.hpp
//
// Shared-prototype instancing for the GpuRtx path.
//
// partition_csg_groups gives one group per union member, and the multi-BLAS
// build then gives each group its own BLAS *and its own intersection shader*.
// That is correct but it does not exploit repetition: a colonnade of 25
// identical columns compiles 25 byte-identical intersection shaders, builds 25
// BLASes over 25 world-space boxes, and fills 25 SBT hit records. The cost is
// linear in the placement count even though there is only one distinct shape,
// and at a thousand placements the pipeline build dominates everything else.
//
// The hardware already has the mechanism for this: a TLAS instance carries an
// object->world transform, so one BLAS can appear at N places and the traversal
// unit transforms the ray into object space on the way in. What that needs from
// us is the split the flat union throws away — which sub-tree is the shape, and
// which part of the node chain is the placement.
//
// factor_rt_instances recovers exactly that split. For each union member it
// walks down the *affine* transform chain (Translate / Scale / RotateX / Y / Z),
// accumulating the world->object map those nodes apply to the query point, and
// stops at the first node that is not one of them. What remains is the shape in
// its own object space; members whose shapes hash identically share one
// prototype. Inverting the accumulated map gives the object->world matrix the
// TLAS instance wants.
//
// Non-affine deformations (TwistY, BendXY, TaperY) are deliberately not walked
// through: they are not representable as an instance transform, so they stay
// inside the prototype body where the SDF still evaluates them correctly.
//
// The result feeds three things that then stop scaling with the placement
// count: one BLAS per prototype, one intersection shader per prototype, and one
// SBT hit record per prototype. Only the TLAS instance array stays O(N), which
// is the one part the hardware is built to chew through.

#pragma once

#include "core/frep/node.hpp"
#include "core/frep/scene.hpp"
#include "core/gpu/rtx_accel.hpp"

#include <optional>
#include <vector>

namespace frep::gpu {

// One shared shape: a sub-tree in its own object space, plus its object-space
// bounds. `body` is what the prototype's intersection shader sphere-traces.
struct RtProto {
    FRepNode::Ptr  body;
    FRepNode::AABB box;
};

// The factoring of a scene root into prototypes + placements. `placements[i]`
// refers to `protos[placements[i].blas]`.
struct RtInstancing {
    std::vector<RtProto>          protos;
    std::vector<RtInstanceXform>  placements;
};

// Factor a scene root into shared prototypes and TLAS placements.
//
// Returns nullopt when instancing would buy nothing — fewer than two members,
// or every member a distinct shape (protos.size() == placements.size()), in
// which case the caller should use the plain multi-BLAS group path, which is
// the same construction with the transforms left in the shaders.
std::optional<RtInstancing> factor_rt_instances(const FRepNode::Ptr& root);

// Wrap each prototype body in a SceneGraph so GlslEmitter can emit its SDF.
// `full` supplies the template definitions the bodies may reference.
std::vector<SceneGraph> make_proto_scenes(const SceneGraph& full,
                                          const RtInstancing& inst);

}  // namespace frep::gpu
