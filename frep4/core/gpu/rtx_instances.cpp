// core/gpu/rtx_instances.cpp — see rtx_instances.hpp.

#include "core/gpu/rtx_instances.hpp"
#include "core/frep/instance.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>

namespace frep::gpu {

namespace {

// A 3x4 affine map p' = M p + t, stored as the 3x3 linear part plus the
// translation. Doubles because the chain is composed then inverted, and a
// rotation composed in floats drifts enough to open a seam between the box the
// BLAS was built for and the box the shader marches in.
struct Affine {
    double m[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double t[3]    = {0,0,0};
};

// b applied after a: p -> b(a(p)).
Affine compose(const Affine& b, const Affine& a) {
    Affine r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += b.m[i][k] * a.m[k][j];
            r.m[i][j] = s;
        }
        double s = b.t[i];
        for (int k = 0; k < 3; ++k) s += b.m[i][k] * a.t[k];
        r.t[i] = s;
    }
    return r;
}

// Inverse of an affine map. Returns false for a singular linear part (a zero
// scale factor), which has no object->world direction and cannot be instanced.
bool invert(const Affine& a, Affine& out) {
    const auto& m = a.m;
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (std::fabs(det) < 1e-12) return false;
    double id = 1.0 / det;
    out.m[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id;
    out.m[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * id;
    out.m[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    out.m[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * id;
    out.m[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
    out.m[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * id;
    out.m[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id;
    out.m[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * id;
    out.m[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; ++i) {
        double s = 0;
        for (int k = 0; k < 3; ++k) s += out.m[i][k] * a.t[k];
        out.t[i] = -s;
    }
    return true;
}

// The world->object map a single transform node applies to its query point.
// These mirror the node's own codegen exactly — Translate subtracts the offset,
// Scale divides, and the rotations apply the *inverse* rotation — so the
// accumulated chain is the same coordinate change the SDF would have performed.
bool node_affine(const FRepNode& n, Affine& out) {
    auto p = [&](const char* k) -> double {
        auto it = n.params.find(k);
        return it == n.params.end() ? 0.0 : (double)it->second;
    };
    switch (n.kind) {
        case NodeKind::Translate:
            out = Affine{};
            out.t[0] = -p("tx"); out.t[1] = -p("ty"); out.t[2] = -p("tz");
            return true;
        case NodeKind::Scale: {
            double sx = p("sx"), sy = p("sy"), sz = p("sz");
            if (sx == 0 || sy == 0 || sz == 0) return false;
            out = Affine{};
            out.m[0][0] = 1.0 / sx; out.m[1][1] = 1.0 / sy; out.m[2][2] = 1.0 / sz;
            return true;
        }
        case NodeKind::RotateY: {
            double a = p("a"), c = std::cos(a), s = std::sin(a);
            out = Affine{};
            out.m[0][0] =  c; out.m[0][2] = s;
            out.m[2][0] = -s; out.m[2][2] = c;
            return true;
        }
        case NodeKind::RotateX: {
            double a = p("a"), c = std::cos(a), s = std::sin(a);
            out = Affine{};
            out.m[1][1] =  c; out.m[1][2] = s;
            out.m[2][1] = -s; out.m[2][2] = c;
            return true;
        }
        case NodeKind::RotateZ: {
            double a = p("a"), c = std::cos(a), s = std::sin(a);
            out = Affine{};
            out.m[0][0] =  c; out.m[0][1] = s;
            out.m[1][0] = -s; out.m[1][1] = c;
            return true;
        }
        default:
            return false;
    }
}

// Flatten a union tree into its members, the same cut partition_csg_groups
// makes: union is separable, everything else is one indivisible member.
void collect_members(const FRepNode::Ptr& n, std::vector<FRepNode::Ptr>& out) {
    if (!n) return;
    if (n->kind == NodeKind::Union)
        for (const auto& c : n->children) collect_members(c, out);
    else
        out.push_back(n);
}

}  // namespace

std::optional<RtInstancing> factor_rt_instances(const FRepNode::Ptr& root) {
    if (!root) return std::nullopt;

    std::vector<FRepNode::Ptr> members;
    collect_members(root, members);
    if (members.size() < 2) return std::nullopt;

    RtInstancing out;
    std::unordered_map<std::size_t, int> proto_of;   // shape hash -> prototype

    for (const auto& m : members) {
        // Walk down the affine chain, accumulating world->object. Each node's
        // map applies to the point its parent already produced, so the new node
        // composes *after* what we have: p_obj = T_k(...T_1(p_world)).
        Affine w2o;
        FRepNode::Ptr body = m;
        while (body) {
            Affine step;
            if (body->children.size() != 1 || !node_affine(*body, step)) break;
            w2o = compose(step, w2o);
            body = body->children[0];
        }

        // An InstanceNode is a reference, not a shape: descend to the shared
        // target so the prototype is the geometry itself. (This is the same
        // sharing factor_instances recorded; here we finally consume it.)
        while (body && body->kind == NodeKind::Instance &&
               body->children.size() == 1 && body->children[0])
            body = body->children[0];

        // A member we cannot place — a collapsed scale with no inverse, or an
        // empty chain — becomes its own prototype at the identity, which is
        // exactly what the per-group build would have done with it. One
        // degenerate member must not cost the other thousand their sharing.
        Affine o2w;
        if (!body || !invert(w2o, o2w)) {
            body = m;
            o2w  = Affine{};
        }

        std::size_t h = body->structural_hash();
        auto it = proto_of.find(h);
        int pi;
        if (it == proto_of.end()) {
            pi = (int)out.protos.size();
            proto_of.emplace(h, pi);
            out.protos.push_back(RtProto{body, body->aabb()});
        } else {
            pi = it->second;
        }

        RtInstanceXform x;
        x.blas = (std::uint32_t)pi;
        // VkTransformMatrixKHR is row-major 3x4: [R | t].
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) x.m[r * 4 + c] = (float)o2w.m[r][c];
            x.m[r * 4 + 3] = (float)o2w.t[r];
        }
        out.placements.push_back(x);
    }

    // Nothing repeats: every member is its own prototype, so an instanced build
    // would be the group build with extra matrix arithmetic. Let the caller take
    // the simpler path.
    if (out.protos.size() >= out.placements.size()) return std::nullopt;
    return out;
}

std::vector<SceneGraph> make_proto_scenes(const SceneGraph& full,
                                          const RtInstancing& inst) {
    std::vector<SceneGraph> scenes;
    scenes.reserve(inst.protos.size());
    for (const auto& p : inst.protos) {
        SceneGraph s;
        s.add_object(p.body);
        s.templates() = full.templates();   // bodies may call frep_tmpl_* helpers
        scenes.push_back(std::move(s));
    }
    return scenes;
}

}  // namespace frep::gpu
