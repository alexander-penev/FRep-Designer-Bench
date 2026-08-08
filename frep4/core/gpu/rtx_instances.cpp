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

// Structural equality: same node kind, same parameters, same children, all the
// way down. Node identity short-circuits it, which is the common case once
// factor_instances has rewired copies to a shared pointer.
//
// This exists because a hash is not enough to decide sharing. Everywhere else
// in the system a structural_hash collision costs a missed cache hit; here it
// would make every copy of one shape render as another, and the failure is
// silent and global rather than local. The id is deliberately excluded -- it is
// per-placement bookkeeping, not geometry.
bool structurally_equal(const FRepNode& a, const FRepNode& b) {
    if (&a == &b) return true;
    if (a.kind != b.kind) return false;
    if (a.children.size() != b.children.size()) return false;
    if (a.params.size() != b.params.size()) return false;
    for (const auto& [k, v] : a.params) {
        auto it = b.params.find(k);
        if (it == b.params.end() || it->second != v) return false;
    }
    // Plugin and expression nodes carry state the base class cannot see
    // (CustomExpr's source text, a plugin's own fields). Their type_name and
    // structural_hash fold it in, so require both to agree before trusting the
    // parameter comparison above.
    if (std::strcmp(a.type_name(), b.type_name()) != 0) return false;
    if (a.structural_hash() != b.structural_hash()) return false;
    for (std::size_t i = 0; i < a.children.size(); ++i) {
        if (!a.children[i] || !b.children[i]) {
            if (a.children[i] != b.children[i]) return false;
            continue;
        }
        if (!structurally_equal(*a.children[i], *b.children[i])) return false;
    }
    return true;
}

// A box the BLAS can actually prune with. FRepNode::AABB::infinite() is the
// default for anything that cannot bound itself (a Plane, a CustomExpr with no
// declared bounds), and a BLAS built over it is not an acceleration structure:
// every ray enters every instance. On the per-group path that wastes one
// object; shared across a thousand placements it would make the TLAS actively
// harmful, so such a body is not eligible to be a prototype.
bool bounded(const FRepNode::AABB& b) {
    const float lim = 1e8f;   // infinite() uses 1e9
    const float e[6] = {b.min_x, b.min_y, b.min_z, b.max_x, b.max_y, b.max_z};
    for (float v : e) if (!(std::fabs(v) < lim)) return false;
    return b.max_x >= b.min_x && b.max_y >= b.min_y && b.max_z >= b.min_z;
}

}  // namespace

std::optional<RtInstancing> factor_rt_instances(const SceneGraph& scene) {
    // Members with the material of the object each came from. The executor
    // unions every visible object's geometry, so a member's material is the
    // material of the object whose tree it was cut out of -- and since the
    // closest-hit shader now shades from the prototype, that material has to
    // travel with it rather than be looked up in the whole scene.
    std::vector<FRepNode::Ptr> members;
    std::vector<const Material*> member_mat;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        std::size_t before = members.size();
        collect_members(obj.geometry, members);
        member_mat.resize(members.size(), &obj.material);
        for (std::size_t i = before; i < members.size(); ++i)
            member_mat[i] = &obj.material;
    }
    if (members.size() < 2) return std::nullopt;

    RtInstancing out;
    // Candidate prototypes bucketed by hash; the hash only narrows the search,
    // and structural equality decides. Buckets keep this near-linear without
    // letting a collision merge two different shapes.
    std::unordered_map<std::size_t, std::vector<int>> buckets;

    for (std::size_t mi = 0; mi < members.size(); ++mi) {
        const FRepNode::Ptr& m = members[mi];
        const Material& mat = *member_mat[mi];
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

        // A member we cannot place becomes its own prototype at the identity,
        // which is exactly what the per-group build would have done with it.
        // One awkward member must not cost the other thousand their sharing.
        // Three ways to land here: a collapsed scale with no inverse, an empty
        // chain, and a body that cannot bound itself -- the last because a BLAS
        // over an infinite box prunes nothing, and shared across every
        // placement that would be worse than not instancing at all.
        Affine o2w;
        if (!body || !invert(w2o, o2w) || !bounded(body->aabb())) {
            body = m;
            o2w  = Affine{};
        }

        // Hash to narrow, structural equality + material to decide.
        std::size_t h = body->structural_hash();
        auto& bucket = buckets[h];
        int pi = -1;
        for (int cand : bucket) {
            if (out.protos[cand].material == mat &&
                structurally_equal(*out.protos[cand].body, *body)) { pi = cand; break; }
        }
        if (pi < 0) {
            pi = (int)out.protos.size();
            bucket.push_back(pi);
            out.protos.push_back(RtProto{body, body->aabb(), mat});
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
        s.add_object(p.body, p.material);   // material is part of the prototype
        s.templates() = full.templates();   // bodies may call frep_tmpl_* helpers
        scenes.push_back(std::move(s));
    }
    return scenes;
}

}  // namespace frep::gpu
