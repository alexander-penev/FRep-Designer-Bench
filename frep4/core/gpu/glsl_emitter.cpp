// core/gpu/glsl_emitter.cpp

#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_interval.hpp"
#include "core/gpu/glsl_node_interval.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/mesh_sdf.hpp"

#include <cmath>
#include <iomanip>
#include <functional>

namespace frep::gpu {

namespace {

// Helper: emit a float literal with enough precision for round-trip.
std::string flit(float v) {
    std::ostringstream os;
    os << std::setprecision(9) << v;
    std::string s = os.str();
    // Ensure GLSL parses it as a float (presence of '.' or 'e').
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos
        && s.find('E') == std::string::npos)
    {
        s += ".0";
    }
    return s;
}

} // anon

// Parameter choke point: literal when Constant / no table, P.v[slot] when
// Runtime. All node emitters below build their GLSL arithmetic over pval()
// leaves, so the Constant case folds at shader-compile time (bit-identical to
// baking) while the Runtime case reads the shared buffer and the shader source
// is invariant to that parameter's value.
std::string GlslEmitter::pval(Ctx& c, const FRepNode& n, const char* name) {
    float def = n.params.at(name);
    int slot = c.bindings ? c.bindings->slot_of(n.id, name) : -1;
    if (slot < 0) return flit(def);
    return "P.v[" + std::to_string(slot) + "]";
}

// ── Primitives ─────────────────────────────────────────────────────────────

std::string GlslEmitter::emit_sphere(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto v = c.fresh();
    // Use the explicit sqrt(x*x+y*y+z*z) form rather than GLSL length(),
    // matching the CPU JIT path bit-for-operation. length() is permitted to
    // compile to a reduced-precision reciprocal-sqrt sequence on some GPUs,
    // which made the sphere SDF — and therefore every hit point, normal, and
    // shaded colour downstream — differ systematically from the CPU on every
    // pixel (a constant offset, not edge-only). GPU_IR doesn't have this
    // because it shares the CPU's exact sqrt; this is the GLSL-specific gap.
    c.sdf_body << "    float " << v << " = sqrt((" << x << ")*(" << x
               << ") + (" << y << ")*(" << y << ") + (" << z << ")*(" << z
               << ")) - " << pval(c, n, "r") << ";\n";
    return v;
}

std::string GlslEmitter::emit_box(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    // Standard SDF box: q = abs(p) - half_extents; outside dist is
    //   length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0)
    auto q = c.fresh("q");
    auto v = c.fresh();
    c.sdf_body
        << "    vec3 " << q << " = abs(vec3(" << x << ", " << y << ", " << z
        << ")) - vec3(" << pval(c, n, "hx") << ", " << pval(c, n, "hy") << ", "
        << pval(c, n, "hz") << ");\n"
        << "    float " << v << " = length(max(" << q << ", 0.0))"
        << " + min(max(" << q << ".x, max(" << q << ".y, " << q << ".z)),"
        << " 0.0);\n";
    return v;
}

std::string GlslEmitter::emit_plane(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto v = c.fresh();
    c.sdf_body << "    float " << v << " = dot(vec3("
               << x << ", " << y << ", " << z << "), vec3("
               << pval(c, n, "nx") << ", " << pval(c, n, "ny") << ", "
               << pval(c, n, "nz") << ")) + " << pval(c, n, "d") << ";\n";
    return v;
}

// ── Transforms ─────────────────────────────────────────────────────────────

std::string GlslEmitter::emit_translate(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto xp = c.fresh("tx");
    auto yp = c.fresh("ty");
    auto zp = c.fresh("tz");
    c.sdf_body << "    float " << xp << " = " << x << " - " << pval(c, n, "tx") << ";\n"
               << "    float " << yp << " = " << y << " - " << pval(c, n, "ty") << ";\n"
               << "    float " << zp << " = " << z << " - " << pval(c, n, "tz") << ";\n";
    auto child = emit_node(c, *n.children[0], xp, yp, zp);
    return child.value_or("0.0");
}

std::string GlslEmitter::emit_scale(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto sxv=c.fresh("sx"), syv=c.fresh("sy"), szv=c.fresh("sz");
    auto xp=c.fresh("xs"), yp=c.fresh("ys"), zp=c.fresh("zs");
    auto mnv=c.fresh("smn");
    // Per-axis factors read at runtime so an edit needs no re-emit; baked
    // factors fold. Non-uniform scale is not distance-preserving, so scale the
    // resulting distance by the SMALLEST factor to keep a conservative SDF.
    c.sdf_body << "    float " << sxv << " = " << pval(c, n, "sx") << ";\n"
               << "    float " << syv << " = " << pval(c, n, "sy") << ";\n"
               << "    float " << szv << " = " << pval(c, n, "sz") << ";\n"
               << "    float " << xp  << " = " << x << " / " << sxv << ";\n"
               << "    float " << yp  << " = " << y << " / " << syv << ";\n"
               << "    float " << zp  << " = " << z << " / " << szv << ";\n";
    auto child = emit_node(c, *n.children[0], xp, yp, zp);
    if (!child) return "0.0";
    auto v = c.fresh();
    c.sdf_body << "    float " << mnv << " = min(" << sxv << ", min(" << syv << ", " << szv << "));\n"
               << "    float " << v << " = " << *child << " * " << mnv << ";\n";
    return v;
}

std::string GlslEmitter::emit_rotate_y(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto av  = c.fresh("a");
    auto cav = c.fresh("ca");
    auto sav = c.fresh("sa");
    auto xp  = c.fresh("rx");
    auto zp  = c.fresh("rz");
    // Inverse rotation around Y: x' =  cos*x + sin*z, z' = -sin*x + cos*z.
    // cos/sin evaluated in-shader (fold for a baked angle).
    c.sdf_body << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
               << "    float " << cav << " = cos(" << av << ");\n"
               << "    float " << sav << " = sin(" << av << ");\n"
               << "    float " << xp  << " = " << cav << " * " << x
               << " + " << sav << " * " << z << ";\n"
               << "    float " << zp  << " = -" << sav << " * " << x
               << " + " << cav << " * " << z << ";\n";
    auto child = emit_node(c, *n.children[0], xp, y, zp);
    return child.value_or("0.0");
}

std::string GlslEmitter::emit_rotate_x(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto av=c.fresh("a"), cav=c.fresh("ca"), sav=c.fresh("sa");
    auto yp=c.fresh("ry"), zp=c.fresh("rz");
    c.sdf_body << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
               << "    float " << cav << " = cos(" << av << ");\n"
               << "    float " << sav << " = sin(" << av << ");\n"
               << "    float " << yp  << " = " << cav << " * " << y << " + " << sav << " * " << z << ";\n"
               << "    float " << zp  << " = -" << sav << " * " << y << " + " << cav << " * " << z << ";\n";
    auto child = emit_node(c, *n.children[0], x, yp, zp);
    return child.value_or("0.0");
}

std::string GlslEmitter::emit_rotate_z(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto av=c.fresh("a"), cav=c.fresh("ca"), sav=c.fresh("sa");
    auto xp=c.fresh("rx"), yp=c.fresh("ry");
    c.sdf_body << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
               << "    float " << cav << " = cos(" << av << ");\n"
               << "    float " << sav << " = sin(" << av << ");\n"
               << "    float " << xp  << " = " << cav << " * " << x << " + " << sav << " * " << y << ";\n"
               << "    float " << yp  << " = -" << sav << " * " << x << " + " << cav << " * " << y << ";\n";
    auto child = emit_node(c, *n.children[0], xp, yp, z);
    return child.value_or("0.0");
}

// ── Deformations ───────────────────────────────────────────────────────────

std::string GlslEmitter::emit_twist_y(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto kv  = c.fresh("k");
    auto a   = c.fresh("a");
    auto ca  = c.fresh("ca");
    auto sa  = c.fresh("sa");
    auto xp  = c.fresh("twx");
    auto zp  = c.fresh("twz");
    c.sdf_body
        << "    float " << kv << " = " << pval(c, n, "k") << ";\n"
        << "    float " << a  << " = " << kv << " * " << y << ";\n"
        << "    float " << ca << " = cos(" << a << ");\n"
        << "    float " << sa << " = sin(" << a << ");\n"
        << "    float " << xp << " = " << ca << " * " << x << " + " << sa << " * " << z << ";\n"
        << "    float " << zp << " = -" << sa << " * " << x << " + " << ca << " * " << z << ";\n";
    auto child = emit_node(c, *n.children[0], xp, y, zp);
    if (!child) return "0.0";
    // Lipschitz: sqrt(1 + (k * r)^2), r^2 = x*x + z*z.
    auto lip = c.fresh("lip");
    auto v   = c.fresh();
    c.sdf_body
        << "    float " << lip << " = sqrt(1.0 + (" << kv << " * " << kv
        << ") * (" << x << "*" << x << " + " << z << "*" << z << "));\n"
        << "    float " << v << " = " << *child << " / " << lip << ";\n";
    return v;
}

std::string GlslEmitter::emit_bend_xy(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    // A baked, ~zero k is an exact no-op (descend unchanged). A runtime k
    // cannot be inspected at emit time, so emit the full bend with an in-shader
    // guard against k≈0 (imperceptible at the clamp floor).
    int slot = c.bindings ? c.bindings->slot_of(n.id, "k") : -1;
    float kc = n.params.at("k");
    if (slot < 0 && std::abs(kc) < 1e-6f) {
        auto child = emit_node(c, *n.children[0], x, y, z);
        return child.value_or("0.0");
    }
    auto kv   = c.fresh("k");
    auto ks   = c.fresh("ks");
    auto invk = c.fresh("ik");
    auto th   = c.fresh("th");
    auto rv   = c.fresh("r");
    auto xp   = c.fresh("bx");
    auto yp   = c.fresh("by");
    c.sdf_body
        << "    float " << kv   << " = " << pval(c, n, "k") << ";\n"
        << "    float " << ks   << " = (abs(" << kv << ") < 1.0e-6) ? 1.0e-6 : " << kv << ";\n"
        << "    float " << invk << " = 1.0 / " << ks << ";\n"
        << "    float " << th   << " = " << ks << " * " << x << ";\n"
        << "    float " << rv   << " = " << invk << " + " << y << ";\n"
        << "    float " << xp   << " = " << rv << " * sin(" << th << ");\n"
        << "    float " << yp   << " = " << rv << " * cos(" << th << ") - " << invk << ";\n";
    auto child = emit_node(c, *n.children[0], xp, yp, z);
    if (!child) return "0.0";
    auto lip = c.fresh("lip");
    auto v   = c.fresh();
    c.sdf_body
        << "    float " << lip << " = max(1.0, abs(" << ks << " * " << rv << "));\n"
        << "    float " << v << " = " << *child << " / " << lip << ";\n";
    return v;
}

std::string GlslEmitter::emit_taper_y(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    auto tv = c.fresh("t");
    auto hv = c.fresh("h");
    auto u  = c.fresh("u");
    auto s  = c.fresh("s");
    auto is = c.fresh("is");
    auto xp = c.fresh("tpx");
    auto zp = c.fresh("tpz");
    c.sdf_body
        << "    float " << tv << " = " << pval(c, n, "t") << ";\n"
        << "    float " << hv << " = " << pval(c, n, "h") << ";\n"
        << "    float " << u  << " = clamp((" << y << " + 0.5 * " << hv
        << ") / " << hv << ", 0.0, 1.0);\n"
        << "    float " << s  << " = max(1.0 + " << u << " * ("
        << tv << " - 1.0), 1.0e-3);\n"
        << "    float " << is << " = 1.0 / " << s << ";\n"
        << "    float " << xp << " = " << x << " * " << is << ";\n"
        << "    float " << zp << " = " << z << " * " << is << ";\n";
    auto child = emit_node(c, *n.children[0], xp, y, zp);
    if (!child) return "0.0";
    auto lip = c.fresh("lip");
    auto v   = c.fresh();
    c.sdf_body
        << "    float " << lip << " = max(1.0, " << is << ");\n"
        << "    float " << v << " = " << *child << " / " << lip << ";\n";
    return v;
}

// ── MeshSDF: trilinear interp of a voxel grid in a storage buffer ──────────

std::string GlslEmitter::emit_mesh_sdf(Ctx& c, const FRepNode& n,
    const std::string& x, const std::string& y, const std::string& z)
{
    // We arrive here only after the dispatcher confirmed the node is a
    // MeshSDFNode via its type_name(); a static_cast is safe.
    const auto* m = static_cast<const MeshSDFNode*>(&n);
    if (!c.mesh_accum) {
        // Defensive fallback — emit a constant positive distance.
        return "1.0";
    }
    MeshAccum& acc = *c.mesh_accum;

    // If this exact node was already emitted (the albedo body re-visits every
    // node the SDF body did), reuse its slot — don't append the voxels again.
    int gpu_index;
    auto it = acc.node_to_index.find(static_cast<const void*>(m));
    if (it != acc.node_to_index.end()) {
        gpu_index = it->second;
    } else {
        // Allocate a slot for this mesh and append its voxels to the global
        // buffer. Each mesh becomes a separate GLSL `sample_mesh_<i>` function
        // by index in `acc.meshes`.
        MeshMeta meta;
        meta.offset    = static_cast<int>(acc.mesh_voxels.size());
        meta.res       = m->resolution();
        meta.gpu_index = static_cast<int>(acc.meshes.size());
        m->bbox_min(meta.bmin);
        m->bbox_max(meta.bmax);
        m->cell_size(meta.cell);
        int N = meta.res;
        acc.mesh_voxels.insert(acc.mesh_voxels.end(),
            m->grid_data(), m->grid_data() + N * N * N);
        acc.meshes.push_back(meta);
        gpu_index = meta.gpu_index;
        acc.node_to_index.emplace(static_cast<const void*>(m), gpu_index);
    }

    // Emit a call to the per-mesh sample function. The function's body
    // is generated once at the end of emit() so all meshes share the
    // same storage buffer.
    auto v = c.fresh();
    c.sdf_body << "    float " << v << " = sample_mesh_" << gpu_index
               << "(" << x << ", " << y << ", " << z << ");\n";
    return v;
}

// ── Dispatch on node kind ──────────────────────────────────────────────────

// Emit `target` as a standalone GLSL function once, keyed by pointer. The body
// is produced by emit_node into a fresh sub-Ctx (so its temporaries are local to
// the function), then wrapped as `float _inst_fn_N(float x,float y,float z)`.
// The sub-Ctx shares the mesh accumulator (so a mesh inside an instanced subtree
// still registers its voxels once) and the binding table, but gets its OWN
// inst_funcs pointer set to the same table, so a nested instance inside the
// target also emits/reuses functions. Returns "" on failure.
std::string GlslEmitter::emit_instance_fn(Ctx& c, const FRepNode& target) {
    auto& F = *c.inst_funcs;
    if (auto it = F.ptr_to_fn.find(&target); it != F.ptr_to_fn.end())
        return it->second;                          // already emitted

    std::string name = "_inst_fn_" + std::to_string(F.next_fn++);
    // Reserve the name before emitting the body so a self-consistent (acyclic)
    // nested reference can find it; cycles are already excluded upstream.
    F.ptr_to_fn[&target] = name;

    Ctx sub;
    sub.mesh_accum = c.mesh_accum;
    sub.bindings   = c.bindings;
    sub.inst_funcs = c.inst_funcs;
    auto body = emit_node(sub, target, "x", "y", "z");
    if (!body) return {};

    F.defs << "float " << name << "(float x, float y, float z) {\n"
           << sub.sdf_body.str()
           << "    return " << *body << ";\n"
           << "}\n";
    return name;
}

std::string GlslEmitter::emit_instance_grad_fn(Ctx& c, const FRepNode& target) {
    auto& F = *c.inst_funcs;
    if (auto it = F.ptr_to_grad_fn.find(&target); it != F.ptr_to_grad_fn.end())
        return it->second;

    std::string name = "_inst_grad_fn_" + std::to_string(F.next_grad_fn++);
    F.ptr_to_grad_fn[&target] = name;

    Ctx sub;
    sub.mesh_accum = c.mesh_accum;
    sub.bindings   = c.bindings;
    sub.inst_funcs = c.inst_funcs;
    // The function takes the incoming Duals (which may already carry an outer
    // transform's derivative) rather than resetting them to identity, so AD stays
    // correct when the instance is wrapped in Translate/Rotate/etc.
    auto body = emit_node_dual(sub, target, "x", "y", "z");
    if (!body) return {};   // e.g. BendXY has no dual emitter -> caller inlines

    F.grad_defs << "Dual " << name << "(Dual x, Dual y, Dual z) {\n"
                << sub.grad_body.str()
                << "    return " << *body << ";\n"
                << "}\n";
    return name;
}

std::expected<std::string, std::string>
GlslEmitter::emit_node(Ctx& c, const FRepNode& n,
                       const std::string& x, const std::string& y,
                       const std::string& z)
{
    using K = NodeKind;
    // MeshSDFNode doesn't have its own NodeKind enum value — detect via
    // type_name() (RTTI is disabled project-wide).
    if (std::string_view(n.type_name()) == "MeshSDF") {
        return emit_mesh_sdf(c, n, x, y, z);
    }
    switch (n.kind) {
        case K::Sphere:    return emit_sphere(c, n, x, y, z);
        case K::Box:       return emit_box(c, n, x, y, z);
        case K::Plane:     return emit_plane(c, n, x, y, z);
        case K::Translate: return emit_translate(c, n, x, y, z);
        case K::Scale:     return emit_scale(c, n, x, y, z);
        case K::RotateY:   return emit_rotate_y(c, n, x, y, z);
        case K::RotateX:   return emit_rotate_x(c, n, x, y, z);
        case K::RotateZ:   return emit_rotate_z(c, n, x, y, z);
        case K::TwistY:    return emit_twist_y(c, n, x, y, z);
        case K::BendXY:    return emit_bend_xy(c, n, x, y, z);
        case K::TaperY:    return emit_taper_y(c, n, x, y, z);

        case K::Instance: {
            if (n.children.empty() || !n.children[0])
                return std::string("1e30");        // dangling -> empty
            const FRepNode* target = n.children[0].get();
            // Level 2: if this target is shared (referenced by >=1 instance) and
            // we have a function table, emit the target as a function once and
            // call it here. Otherwise fall back to inlining (Level 1 behaviour).
            if (c.inst_funcs &&
                c.inst_funcs->shared_targets.count(target)) {
                std::string fn = emit_instance_fn(c, *target);
                if (fn.empty())
                    return std::unexpected(std::string("instance fn emit failed"));
                std::string v = c.fresh();
                c.sdf_body << "    float " << v << " = " << fn
                           << "(" << x << ", " << y << ", " << z << ");\n";
                return v;
            }
            return emit_node(c, *target, x, y, z);   // inline (no instances of it)
        }

        case K::Union:
        case K::Intersection:
        case K::Difference:
        case K::SmoothUnion: {
            // Binary operators expect two children.
            if (n.children.size() < 2)
                return std::unexpected(
                    "binary op '" + std::string(n.type_name()) + "' has < 2 children");
            auto a = emit_node(c, *n.children[0], x, y, z);
            if (!a) return a;
            auto b = emit_node(c, *n.children[1], x, y, z);
            if (!b) return b;
            auto v = c.fresh();
            switch (n.kind) {
                case K::Union:
                    c.sdf_body << "    float " << v << " = min("
                               << *a << ", " << *b << ");\n";
                    break;
                case K::Intersection:
                    c.sdf_body << "    float " << v << " = max("
                               << *a << ", " << *b << ");\n";
                    break;
                case K::Difference:
                    c.sdf_body << "    float " << v << " = max("
                               << *a << ", -(" << *b << "));\n";
                    break;
                case K::SmoothUnion: {
                    float kk = 0.1f;
                    auto it = n.params.find("k");
                    if (it != n.params.end()) kk = it->second;
                    int kslot = (c.bindings && it != n.params.end())
                                ? c.bindings->slot_of(n.id, "k") : -1;
                    std::string kbase = (kslot < 0)
                        ? flit(kk)
                        : ("P.v[" + std::to_string(kslot) + "]");
                    // Cubic polynomial smin (Inigo Quilez): C2-continuous, so
                    // the normal has no kink at the blend boundary (the visible
                    // "edges" the quadratic smin produced). Must match the CPU
                    // codegen() in operations.hpp exactly for cross-path parity.
                    //   kk6 = k*6;  h = max(kk6 - |a-b|, 0)/kk6;
                    //   smin = min(a,b) - h*h*h*kk6/6
                    c.sdf_body
                        << "    float " << v << "; {\n"
                        << "        float kk = (" << kbase << ") * 2.0;\n"
                        << "        float h = max(kk - abs((" << *a << ") - ("
                        << *b << ")), 0.0) / kk;\n"
                        << "        " << v << " = min(" << *a << ", " << *b
                        << ") - h*h*h*kk*(1.0/6.0);\n"
                        << "    }\n";
                    break;
                }
                default: break;
            }
            return v;
        }

        case K::Negate: {
            if (n.children.empty())
                return std::unexpected("Negate has no child");
            auto child = emit_node(c, *n.children[0], x, y, z);
            if (!child) return child;
            auto v = c.fresh();
            c.sdf_body << "    float " << v << " = -(" << *child << ");\n";
            return v;
        }

        default: {
            // Plugin fallback: ask the node to emit its own GLSL. First
            // recursively emit each child expression, then hand them to
            // the node's emit_glsl() method. This lets plugin authors
            // ship GPU support alongside their CPU codegen without
            // touching the emitter itself.
            std::vector<std::string> child_exprs;
            child_exprs.reserve(n.children.size());
            for (const auto& ch : n.children) {
                auto ce = emit_node(c, *ch, x, y, z);
                if (!ce) return ce;
                child_exprs.push_back(*ce);
            }
            auto v = c.fresh();
            std::ostringstream node_expr;
            std::string prefix = v + "_";
            if (n.emit_glsl(node_expr, child_exprs, prefix)) {
                c.sdf_body << "    float " << v << " = "
                           << node_expr.str() << ";\n";
                return v;
            }
            return std::unexpected(
                std::string("unsupported node type for GLSL emit: ")
                + n.type_name());
        }
    }
}

// ── Dual-number AD emitter ──────────────────────────────────────────────────
// Emits the SDF and its analytic gradient together, in GLSL `Dual` arithmetic
// (see the Dual struct + d_* helpers emitted into the shader). Mirrors
// emit_node but every value is a Dual {float v; vec3 g;}. The result is an
// exact analytic normal, matching the CPU's forward-mode AD instead of the
// old central differences.
std::expected<std::string, std::string>
GlslEmitter::emit_node_dual(Ctx& c, const FRepNode& n,
                            const std::string& x, const std::string& y,
                            const std::string& z)
{
    using K = NodeKind;
    auto& g = c.grad_body;

    switch (n.kind) {
        case K::Sphere: {
            // |p| - r ; gradient of |p| is p/|p|.
            auto v = c.fresh_d();
            // length of the dual vec3 (x,y,z): d_len3 returns Dual.
            g << "    Dual " << v << " = d_sub_s(d_len3(" << x << ", " << y
              << ", " << z << "), " << pval(c, n, "r") << ");\n";
            return v;
        }
        case K::Box: {
            auto v = c.fresh_d();
            g << "    Dual " << v << " = d_box(" << x << ", " << y << ", " << z
              << ", " << pval(c, n, "hx") << ", " << pval(c, n, "hy") << ", "
              << pval(c, n, "hz") << ");\n";
            return v;
        }
        case K::Plane: {
            auto v = c.fresh_d();
            // dot(p,n)+d : linear, gradient is constant n.
            g << "    Dual " << v << " = d_add_s(d_dot3(" << x << ", " << y
              << ", " << z << ", " << pval(c, n, "nx") << ", " << pval(c, n, "ny") << ", "
              << pval(c, n, "nz") << "), " << pval(c, n, "d") << ");\n";
            return v;
        }
        case K::Translate: {
            auto xp = c.fresh_d(), yp = c.fresh_d(), zp = c.fresh_d();
            // Subtracting a (possibly runtime) offset doesn't change the gradient.
            g << "    Dual " << xp << " = d_sub_s(" << x << ", " << pval(c, n, "tx") << ");\n"
              << "    Dual " << yp << " = d_sub_s(" << y << ", " << pval(c, n, "ty") << ");\n"
              << "    Dual " << zp << " = d_sub_s(" << z << ", " << pval(c, n, "tz") << ");\n";
            return emit_node_dual(c, *n.children[0], xp, yp, zp);
        }
        case K::Scale: {
            auto sxv=c.fresh("sx"), syv=c.fresh("sy"), szv=c.fresh("sz"), mnv=c.fresh("smn");
            g << "    float " << sxv << " = " << pval(c, n, "sx") << ";\n"
              << "    float " << syv << " = " << pval(c, n, "sy") << ";\n"
              << "    float " << szv << " = " << pval(c, n, "sz") << ";\n";
            auto xp = c.fresh_d(), yp = c.fresh_d(), zp = c.fresh_d();
            g << "    Dual " << xp << " = d_mul_s(" << x << ", 1.0 / " << sxv << ");\n"
              << "    Dual " << yp << " = d_mul_s(" << y << ", 1.0 / " << syv << ");\n"
              << "    Dual " << zp << " = d_mul_s(" << z << ", 1.0 / " << szv << ");\n";
            auto child = emit_node_dual(c, *n.children[0], xp, yp, zp);
            if (!child) return child;
            auto v = c.fresh_d();
            g << "    float " << mnv << " = min(" << sxv << ", min(" << syv << ", " << szv << "));\n"
              << "    Dual " << v << " = d_mul_s(" << *child << ", " << mnv << ");\n";
            return v;
        }
        case K::RotateY: {
            auto av  = c.fresh("a");
            auto cav = c.fresh("ca");
            auto sav = c.fresh("sa");
            // Inverse rotation; cos/sin in-shader (fold for a baked angle).
            g << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
              << "    float " << cav << " = cos(" << av << ");\n"
              << "    float " << sav << " = sin(" << av << ");\n";
            auto xp = c.fresh_d(), zp = c.fresh_d();
            g << "    Dual " << xp << " = d_add(d_mul_s(" << x << ", " << cav
              << "), d_mul_s(" << z << ", " << sav << "));\n"
              << "    Dual " << zp << " = d_add(d_mul_s(" << x << ", -" << sav
              << "), d_mul_s(" << z << ", " << cav << "));\n";
            return emit_node_dual(c, *n.children[0], xp, y, zp);
        }
        case K::RotateX: {
            auto av=c.fresh("a"), cav=c.fresh("ca"), sav=c.fresh("sa");
            g << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
              << "    float " << cav << " = cos(" << av << ");\n"
              << "    float " << sav << " = sin(" << av << ");\n";
            auto yp = c.fresh_d(), zp = c.fresh_d();
            g << "    Dual " << yp << " = d_add(d_mul_s(" << y << ", " << cav
              << "), d_mul_s(" << z << ", " << sav << "));\n"
              << "    Dual " << zp << " = d_add(d_mul_s(" << y << ", -" << sav
              << "), d_mul_s(" << z << ", " << cav << "));\n";
            return emit_node_dual(c, *n.children[0], x, yp, zp);
        }
        case K::RotateZ: {
            auto av=c.fresh("a"), cav=c.fresh("ca"), sav=c.fresh("sa");
            g << "    float " << av  << " = " << pval(c, n, "a") << ";\n"
              << "    float " << cav << " = cos(" << av << ");\n"
              << "    float " << sav << " = sin(" << av << ");\n";
            auto xp = c.fresh_d(), yp = c.fresh_d();
            g << "    Dual " << xp << " = d_add(d_mul_s(" << x << ", " << cav
              << "), d_mul_s(" << y << ", " << sav << "));\n"
              << "    Dual " << yp << " = d_add(d_mul_s(" << x << ", -" << sav
              << "), d_mul_s(" << y << ", " << cav << "));\n";
            return emit_node_dual(c, *n.children[0], xp, yp, z);
        }
        case K::TwistY: {
            auto kv = c.fresh("k");
            g << "    float " << kv << " = " << pval(c, n, "k") << ";\n";
            auto a = c.fresh_d(), ca = c.fresh_d(), sa = c.fresh_d();
            auto xp = c.fresh_d(), zp = c.fresh_d();
            g << "    Dual " << a  << " = d_mul_s(" << y << ", " << kv << ");\n"
              << "    Dual " << ca << " = d_cos(" << a << ");\n"
              << "    Dual " << sa << " = d_sin(" << a << ");\n"
              << "    Dual " << xp << " = d_add(d_mul(" << ca << ", " << x
              << "), d_mul(" << sa << ", " << z << "));\n"
              << "    Dual " << zp << " = d_sub(d_mul(" << ca << ", " << z
              << "), d_mul(" << sa << ", " << x << "));\n";
            auto child = emit_node_dual(c, *n.children[0], xp, y, zp);
            if (!child) return child;
            // Lipschitz divisor sqrt(1 + (k r)^2), r^2 = x.v^2 + z.v^2.
            auto v = c.fresh_d();
            g << "    Dual " << v << " = d_div(" << *child
              << ", d_sqrt(d_add_s(d_mul_s(d_add(d_mul(" << x << ", " << x
              << "), d_mul(" << z << ", " << z << ")), " << kv << "*" << kv
              << "), 1.0)));\n";
            return v;
        }
        case K::TaperY: {
            // Match the CPU path exactly:
            //   u = clamp((y + 0.5h)/h, 0, 1)
            //   s = max(1 + u*(t-1), 1e-3)
            //   xr = x/s, zr = z/s
            //   result = child(xr, y, zr) / max(1, 1/s)   (Lipschitz)
            auto tv = c.fresh("t");
            auto hv = c.fresh("h");
            int tslot = c.bindings ? c.bindings->slot_of(n.id, "t") : -1;
            int hslot = (c.bindings && n.params.count("h"))
                        ? c.bindings->slot_of(n.id, "h") : -1;
            float hdef = n.params.count("h") ? n.params.at("h") : 2.0f;
            auto u  = c.fresh_d(), s = c.fresh_d();
            auto xp = c.fresh_d(), zp = c.fresh_d();
            g << "    float " << tv << " = "
              << (tslot < 0 ? flit(n.params.at("t"))
                            : ("P.v[" + std::to_string(tslot) + "]")) << ";\n"
              << "    float " << hv << " = "
              << (hslot < 0 ? flit(hdef)
                            : ("P.v[" + std::to_string(hslot) + "]")) << ";\n";
            // u = clamp((y + 0.5h)/h, 0, 1); s = max(1 + u*(t-1), 1e-3)
            g << "    Dual " << u << " = d_clamp_s(d_mul_s(d_add_s(" << y << ", "
              << "0.5 * " << hv << "), 1.0 / " << hv << "), 0.0, 1.0);\n"
              << "    Dual " << s << " = d_max_s(d_add_s(d_mul_s(" << u << ", "
              << "(" << tv << " - 1.0)), 1.0), 1e-3);\n"
              << "    Dual " << xp << " = d_div(" << x << ", " << s << ");\n"
              << "    Dual " << zp << " = d_div(" << z << ", " << s << ");\n";
            auto child = emit_node_dual(c, *n.children[0], xp, y, zp);
            if (!child) return child;
            // Lipschitz divisor max(1, 1/s).
            auto v = c.fresh_d();
            g << "    Dual " << v << " = d_div(" << *child
              << ", d_max_s(d_div(d_one(), " << s << "), 1.0));\n";
            return v;
        }
        case K::Union:
        case K::Intersection:
        case K::Difference:
        case K::SmoothUnion: {
            if (n.children.size() < 2)
                return std::unexpected("binary op has < 2 children (dual)");
            auto a = emit_node_dual(c, *n.children[0], x, y, z);
            if (!a) return a;
            auto b = emit_node_dual(c, *n.children[1], x, y, z);
            if (!b) return b;
            auto v = c.fresh_d();
            if (n.kind == K::Union)
                g << "    Dual " << v << " = d_min(" << *a << ", " << *b << ");\n";
            else if (n.kind == K::Intersection)
                g << "    Dual " << v << " = d_max(" << *a << ", " << *b << ");\n";
            else if (n.kind == K::Difference)
                g << "    Dual " << v << " = d_max(" << *a << ", d_neg(" << *b << "));\n";
            else { // SmoothUnion — IQ smin in dual arithmetic
                float kk = 0.1f;
                auto it = n.params.find("k");
                if (it != n.params.end()) kk = it->second;
                int kslot = (c.bindings && it != n.params.end())
                            ? c.bindings->slot_of(n.id, "k") : -1;
                std::string kbase = (kslot < 0)
                    ? flit(kk)
                    : ("P.v[" + std::to_string(kslot) + "]");
                g << "    Dual " << v << " = d_smin(" << *a << ", " << *b
                  << ", " << kbase << ");\n";
            }
            return v;
        }
        case K::Negate: {
            if (n.children.empty())
                return std::unexpected("Negate has no child (dual)");
            auto child = emit_node_dual(c, *n.children[0], x, y, z);
            if (!child) return child;
            auto v = c.fresh_d();
            g << "    Dual " << v << " = d_neg(" << *child << ");\n";
            return v;
        }
        case K::Instance: {
            if (n.children.empty() || !n.children[0])
                return std::unexpected(std::string("Instance is dangling (dual)"));
            const FRepNode* target = n.children[0].get();
            if (c.inst_funcs && c.inst_funcs->shared_targets.count(target)) {
                std::string fn = emit_instance_grad_fn(c, *target);
                if (!fn.empty()) {
                    std::string v = c.fresh_d();
                    c.grad_body << "    Dual " << v << " = " << fn
                                << "(" << x << ", " << y << ", " << z << ");\n";
                    return v;
                }
                // grad fn unavailable (target has a node with no dual emitter):
                // fall through to inline delegation below.
            }
            return emit_node_dual(c, *target, x, y, z);   // delegate to target
        }
        default:
            // BendXY and plugin/mesh/custom nodes: no dual emitter yet.
            // Signal the caller to fall back to finite-difference normals
            // for this object.
            return std::unexpected(
                std::string("no dual AD for node type: ") + n.type_name());
    }
}

std::expected<GlslEmitResult, std::string>
GlslEmitter::emit(const SceneGraph& scene, const TracerConfig& cfg_in,
                  const ParamBindingTable* bindings)
{
    GlslEmitResult res;
    // Publish the runtime-parameter layout so the host can size/seed the
    // buffer; the preamble declares binding 3 only when this is non-empty.
    if (bindings && !bindings->empty()) {
        res.param_bindings = bindings->slots();
        res.placement_hash = bindings->placement_hash();
    }

    // Adaptive raymarch step (mirrors the CPU JIT path). A scene that is
    // a single primitive (optionally affine/twist transformed) is a true
    // SDF and can march at full step (safety_factor 1.0), ~20% faster.
    // Multiple objects are combined with min() in the emitted scene_sdf
    // (a CSG union), and CSG/plugin/mesh/custom-expr nodes break the
    // distance property, so any of those require the conservative step.
    // We copy cfg so the override is local to this emit.
    TracerConfig cfg = cfg_in;
    {
        int visible = 0;
        bool needs_safety = false;
        for (const auto& [id, obj] : scene.objects()) {
            if (!obj.visible || !obj.geometry) continue;
            ++visible;
            if (node_requires_safety_step(*obj.geometry)) needs_safety = true;
        }
        if (visible <= 1 && !needs_safety)
            cfg.safety_factor = 1.0f;
    }

    // ── Resolve which tile-cull method to emit ──────────────────────────────
    // Both paths are in the tree; this only selects between them. Interval is
    // currently available only when the scene is a single CustomExprNode (the
    // interval SDF is emitted from its expression AST); Lipschitz works for any
    // node tree. Auto prefers a metric tree's exact L=1 Lipschitz, and reaches
    // for Interval on a non-metric CustomExpr scene where a coarse probe shows
    // it prunes more; otherwise Lipschitz with the caller's / an estimated L.
    bool use_interval_cull = false;
    const expr::NodePtr* single_ce = nullptr;
    if (cfg.cull_method == TracerConfig::CullMethod::Off) cfg.cull_slabs = 0;
    if (cfg.cull_slabs > 0) {
        int visible = 0; bool all_unit_lip = true;
        const FRepNode* only = nullptr;
        for (const auto& [id, obj] : scene.objects()) {
            if (!obj.visible || !obj.geometry) continue;
            ++visible; only = obj.geometry.get();
            if (!node_is_unit_lipschitz(*obj.geometry)) all_unit_lip = false;
        }
        if (visible == 1 && only)
            single_ce = static_cast<const expr::NodePtr*>(only->custom_expr_ast());

        using CM = TracerConfig::CullMethod;
        if (cfg.cull_method == CM::Interval) {
            use_interval_cull = true;               // node-tree interval now available for any scene
        } else if (cfg.cull_method == CM::Lipschitz) {
            use_interval_cull = false;
        } else if (cfg.cull_method == CM::Auto) {
            if (all_unit_lip) {
                use_interval_cull = false;          // metric tree: L=1 exact + cheapest
                cfg.cull_lipschitz = 1.0f;
            } else {
                use_interval_cull = true;           // non-metric: interval, sound without L
            }
        }
    }

    // We build sdf_body and albedo_body separately. Each object's SDF
    // expression is wrapped in its own `{ ... }` scope so SSA variables
    // never leak between objects (and never leak between the two top-
    // level functions). Each object exposes its result through a
    // function-local variable named "best".
    std::ostringstream sdf_body;
    std::ostringstream albedo_body;
    std::ostringstream pbr_body;
    std::ostringstream grad_body;
    // Analytic normals via dual-number AD. Stays true only if every object
    // has a dual emitter; otherwise we fall back to central differences.
    bool analytic_normals = true;

    sdf_body    << "    float best = 1e9;\n";
    albedo_body << "    float best = 1e9;\n";
    albedo_body << "    vec3  out_alb = vec3(0.5);\n";
    albedo_body << "    vec2  out_rm  = vec2(0.5, 0.0);  // roughness, metallic\n";
    pbr_body    << "    float best = 1e9;\n";
    pbr_body    << "    vec2  out_rm = vec2(0.5, 0.0);\n";
    grad_body   << "    Dual best_grad = Dual(1e9, vec3(0.0, 1.0, 0.0));\n";
    // Per-object reflectivity lookup, mirroring pbr_body. Returns the
    // mirror-reflection strength for the nearest object at a point.
    // Emitted only when reflections are enabled (cfg.max_bounces > 0);
    // otherwise the function is omitted and main() never calls it.
    std::ostringstream refl_body;
    refl_body   << "    float best = 1e9;\n";
    refl_body   << "    float out_refl = 0.0;\n";

    // Aggregate state for MeshSDF voxel uploads. Both the SDF body and
    // the albedo body need to see the same mesh slots (an MeshSDFNode
    // appearing in both should reuse the same sample_mesh function).
    // For simplicity each emit_node call appends; duplicate mesh data
    // is acceptable in this iteration.
    MeshAccum accum;

    // Level 2 instancing pre-pass: find every geometry root referenced by an
    // InstanceNode. Those roots (and only those) become shared GLSL functions,
    // so N instances of a shape emit one function + N calls instead of N copies.
    InstanceFuncs inst_funcs;
    if (cfg.instance_shared_subprograms) {
        // Walk all objects' trees; for each InstanceNode, mark its resolved
        // target pointer as a shared target.
        std::function<void(const FRepNode*)> scan = [&](const FRepNode* nn) {
            if (!nn) return;
            if (nn->kind == NodeKind::Instance && !nn->children.empty() && nn->children[0])
                inst_funcs.shared_targets.insert(nn->children[0].get());
            for (const auto& ch : nn->children)
                if (ch) scan(ch.get());
        };
        for (const auto& [id, obj] : scene.objects())
            if (obj.geometry) scan(obj.geometry.get());
    }

    // Texture aggregation. Each material with pattern == Texture pushes
    // its RGBA8 pixels into `tex_pixels` and records (offset, w, h).
    // The shader then samples via constant offsets baked into the
    // per-object albedo expression.
    struct TexMeta {
        int offset; // byte offset (= 4*pixel offset) into tex_pixels
        int w;
        int h;
    };
    std::vector<TexMeta>      tex_metas;
    std::vector<std::uint8_t> tex_pixels;

    int n_obj = 0;
    int total_ssa = 0;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible) continue;

        // === SDF body: { local Ctx; emit; min into best } =================
        {
            Ctx c;
            c.mesh_accum = &accum;
            c.bindings = bindings;
            c.inst_funcs = &inst_funcs;
            std::expected<std::string, std::string> child;
            // If this object's own geometry root is a shared instance target,
            // emit/reuse its function and call it, so the original and all its
            // instances share one body (the actual memory saving). Otherwise emit
            // inline as before.
            if (inst_funcs.shared_targets.count(obj.geometry.get())) {
                std::string fn = emit_instance_fn(c, *obj.geometry);
                if (fn.empty())
                    return std::unexpected("object '" + id + "' (sdf): instance fn emit failed");
                std::string v = c.fresh();
                c.sdf_body << "    float " << v << " = " << fn << "(x, y, z);\n";
                child = v;
            } else {
                child = emit_node(c, *obj.geometry, "x", "y", "z");
            }
            if (!child) {
                return std::unexpected(
                    "object '" + id + "' (sdf): " + child.error());
            }
            sdf_body << "    {\n"
                     << c.sdf_body.str()
                     << "        best = min(best, " << *child << ");\n"
                     << "    }\n";
            total_ssa += c.next_var;
        }

        // === Gradient body (dual-number AD): build scene_sdf_grad in lock-
        // step with scene_sdf. Each object contributes a Dual; the running
        // result keeps the Dual of the nearest object (matching the min()
        // union — the closest surface owns the normal). If ANY object lacks
        // a dual emitter (e.g. a plugin or BendXY), we abandon the analytic
        // path entirely and scene_normal falls back to central differences,
        // so correctness is preserved.
        if (analytic_normals) {
            Ctx c;
            c.mesh_accum = &accum;
            c.bindings = bindings;
            c.inst_funcs = &inst_funcs;
            std::expected<std::string, std::string> child;
            // If this object's root is a shared instance target, emit/reuse its
            // grad function and call it (so the original and its instances share
            // one dual-AD body — the largest per-object emission). Else inline.
            if (inst_funcs.shared_targets.count(obj.geometry.get())) {
                std::string fn = emit_instance_grad_fn(c, *obj.geometry);
                if (fn.empty()) {
                    analytic_normals = false;   // target has a node with no dual AD
                    child = std::unexpected(std::string("no dual"));
                } else {
                    std::string v = c.fresh_d();
                    c.grad_body << "    Dual x = Dual(p.x, vec3(1.0,0.0,0.0));\n"
                                << "    Dual y = Dual(p.y, vec3(0.0,1.0,0.0));\n"
                                << "    Dual z = Dual(p.z, vec3(0.0,0.0,1.0));\n"
                                << "    Dual " << v << " = " << fn << "(x, y, z);\n";
                    child = v;
                }
            } else {
                // Seed dual coords: value = coord, gradient = basis vector.
                c.grad_body
                    << "    Dual x = Dual(p.x, vec3(1.0, 0.0, 0.0));\n"
                    << "    Dual y = Dual(p.y, vec3(0.0, 1.0, 0.0));\n"
                    << "    Dual z = Dual(p.z, vec3(0.0, 0.0, 1.0));\n";
                child = emit_node_dual(c, *obj.geometry, "x", "y", "z");
            }
            if (!child) {
                // This object can't do analytic AD — disable it globally.
                analytic_normals = false;
            } else {
                grad_body << "    {\n"
                          << c.grad_body.str()
                          << "        if (" << *child << ".v < best_grad.v) "
                          << "best_grad = " << *child << ";\n"
                          << "    }\n";
            }
        }

        // === Albedo body: same SDF expression in a fresh scope, then update
        // out_alb if this object is closer. We need to *temporarily* shadow
        // the outer "best" so the per-object SDF doesn't accidentally
        // reference the running minimum. Use a uniquely-named local.
        {
            Ctx c;
            c.mesh_accum = &accum;
            c.bindings = bindings;
            c.inst_funcs = &inst_funcs;
            std::expected<std::string, std::string> child;
            // Reuse the shared SDF function for the distance test when this
            // object is an instance target (the material colour below stays
            // per-object). Avoids re-emitting the geometry in the albedo body.
            if (inst_funcs.shared_targets.count(obj.geometry.get())) {
                std::string fn = emit_instance_fn(c, *obj.geometry);
                if (fn.empty())
                    return std::unexpected("object '" + id + "' (albedo): instance fn emit failed");
                std::string v = c.fresh();
                c.sdf_body << "    float " << v << " = " << fn << "(x, y, z);\n";
                child = v;
            } else {
                child = emit_node(c, *obj.geometry, "x", "y", "z");
            }
            if (!child) {
                return std::unexpected(
                    "object '" + id + "' (albedo): " + child.error());
            }
            // Emit the albedo expression based on the material's pattern.
            // Solid → constant vec3. Patterned → mix(albedo, albedo2, t)
            // where t comes from a position-dependent function. All math
            // happens in GLSL — identical semantics to the CPU IR emitter
            // in core/compiler/codegen.cpp::emit_pattern lambda.
            std::ostringstream alb_expr;
            const auto& M = obj.material;
            auto vc3 = [&](const std::array<float,3>& a) {
                std::ostringstream o;
                o << "vec3(" << flit(a[0]) << ", "
                              << flit(a[1]) << ", " << flit(a[2]) << ")";
                return o.str();
            };
            switch (M.pattern) {
                case Material::Pattern::Solid:
                    alb_expr << vc3(M.albedo);
                    break;
                case Material::Pattern::Checker: {
                    // parity = (floor(s*x) + floor(s*y) + floor(s*z)) & 1
                    alb_expr
                        << "mix(" << vc3(M.albedo) << ", " << vc3(M.albedo2)
                        << ", float((int(floor(" << flit(M.pattern_scale)
                        << " * x)) + int(floor(" << flit(M.pattern_scale)
                        << " * y)) + int(floor(" << flit(M.pattern_scale)
                        << " * z))) & 1))";
                    break;
                }
                case Material::Pattern::Stripes: {
                    alb_expr
                        << "mix(" << vc3(M.albedo) << ", " << vc3(M.albedo2)
                        << ", float(int(floor(" << flit(M.pattern_scale)
                        << " * y)) & 1))";
                    break;
                }
                case Material::Pattern::GradientY: {
                    // t = clamp((y / scale + 1) / 2, 0, 1)
                    alb_expr
                        << "mix(" << vc3(M.albedo) << ", " << vc3(M.albedo2)
                        << ", clamp((y / " << flit(M.pattern_scale)
                        << " + 1.0) * 0.5, 0.0, 1.0))";
                    break;
                }
                case Material::Pattern::Noise: {
                    // Same Murmur-like integer hash as the CPU IR path:
                    //   h = ix*K1 ^ iy*K2 ^ iz*K3, mix, mask, scale to [0,1)
                    alb_expr
                        << "mix(" << vc3(M.albedo) << ", " << vc3(M.albedo2)
                        << ", float(("
                        << "(int(floor(" << flit(M.pattern_scale) << " * x)) * 0x9E3779B9) ^ "
                        << "(int(floor(" << flit(M.pattern_scale) << " * y)) * 0x85EBCA6B) ^ "
                        << "(int(floor(" << flit(M.pattern_scale) << " * z)) * 0xC2B2AE35)"
                        << ") & 0x7FFFFF) * "
                        << flit(1.0f / 8388608.0f) << ")";
                    break;
                }
                case Material::Pattern::Texture: {
                    // Triplanar projection: sample the texture three times
                    // (against the YZ, XZ, XY planes) and blend by the
                    // surface normal's component weights. The normal is
                    // available globally via the ray-marching code path
                    // (we use scene_normal at the hit point — but that's
                    // only defined inside main(). We *re-derive* it via
                    // central differences inside the albedo function so
                    // scene_albedo can stand alone).
                    if (M.texture_rgba.empty() || M.texture_width <= 0
                        || M.texture_height <= 0)
                    {
                        // Texture mode requested but no pixels supplied —
                        // fall back to the solid albedo.
                        alb_expr << vc3(M.albedo);
                        break;
                    }
                    // Register this texture in the accumulator.
                    TexMeta tm;
                    tm.offset = static_cast<int>(tex_pixels.size());
                    tm.w      = M.texture_width;
                    tm.h      = M.texture_height;
                    tex_pixels.insert(tex_pixels.end(),
                        M.texture_rgba.begin(), M.texture_rgba.end());
                    int tex_idx = static_cast<int>(tex_metas.size());
                    tex_metas.push_back(tm);

                    // The actual call: triplanar_sample_<i>(x, y, z) is
                    // defined later in the shader, after all textures
                    // are known. `pattern_scale` controls world→UV mapping.
                    alb_expr << "triplanar_sample_" << tex_idx
                             << "(vec3(x, y, z), "
                             << flit(M.pattern_scale) << ")";
                    break;
                }
            }

            albedo_body
                << "    {\n"
                << c.sdf_body.str()
                << "        if (" << *child << " < best) {\n"
                << "            best = " << *child << ";\n"
                << "            out_alb = " << alb_expr.str() << ";\n"
                << "            out_rm  = vec2(" << flit(M.roughness) << ", "
                                              << flit(M.metallic)  << ");\n"
                << "        }\n"
                << "    }\n";
            // Same per-object loop body but only tracks roughness/metallic.
            // We can't reuse the SSA `c` from above directly without
            // letting both functions share it, so emit a fresh scope.
            // The SDF computation duplicates — that's fine, the
            // optimizer inlines and folds.
            pbr_body
                << "    {\n"
                << c.sdf_body.str()
                << "        if (" << *child << " < best) {\n"
                << "            best = " << *child << ";\n"
                << "            out_rm = vec2(" << flit(M.roughness) << ", "
                                             << flit(M.metallic)  << ");\n"
                << "        }\n"
                << "    }\n";
            // Reflectivity lookup — only meaningful when reflections are
            // enabled. We always build the body (cheap) but only emit
            // the wrapping function when cfg.max_bounces > 0.
            refl_body
                << "    {\n"
                << c.sdf_body.str()
                << "        if (" << *child << " < best) {\n"
                << "            best = " << *child << ";\n"
                << "            out_refl = " << flit(M.reflectivity) << ";\n"
                << "        }\n"
                << "    }\n";
            total_ssa += c.next_var;
        }
        ++n_obj;
    }

    sdf_body    << "    return best;\n";
    albedo_body << "    return out_alb;\n";
    pbr_body    << "    return out_rm;\n";
    refl_body   << "    return out_refl;\n";

    res.object_count = n_obj;
    res.expr_lines   = total_ssa;
    res.mesh_count   = static_cast<int>(accum.meshes.size());
    res.mesh_voxels  = std::move(accum.mesh_voxels);
    res.texture_count  = static_cast<int>(tex_metas.size());
    res.texture_pixels = std::move(tex_pixels);

    // Assemble the full shader. Keep it close to gpu/sphere_trace.comp so
    // VulkanCtx works unchanged. If there are MeshSDF meshes, splice in
    // the storage buffer declaration + per-mesh sample functions.
    std::ostringstream src;
    src << "#version 450\n"
        << "// AUTO-GENERATED by frep::gpu::GlslEmitter — DO NOT EDIT BY HAND\n"
        << "// objects=" << n_obj << "  ssa_vars=" << res.expr_lines
        << "  meshes=" << res.mesh_count
        << "  textures=" << res.texture_count << "\n"
        << "\n"
        << "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
        << ((cfg.cull_slabs > 0)
            ? "shared float sh_cull_t0;\nshared float sh_cull_t1;\nshared uint sh_occ[" + std::to_string(cfg.cull_slabs) + "];\n"
            : "")
        << "layout(set = 0, binding = 0, rgba8) uniform image2D out_image;\n";

    // If we have MeshSDF nodes, declare the storage buffer.
    if (res.mesh_count > 0) {
        src << "layout(std430, set = 0, binding = 1) readonly buffer MeshData {\n"
            << "    float mesh_voxels[];\n"
            << "} mesh_data;\n";
    }
    // Texture storage buffer (RGBA8 packed as uints for indexed access).
    if (res.texture_count > 0) {
        src << "layout(std430, set = 0, binding = 2) readonly buffer TextureData {\n"
            << "    uint texture_pixels[];\n"
            << "} tex_data;\n";
    }
    // Runtime model-parameter buffer (binding 3). Present only when the caller
    // supplied a ParamBindingTable with runtime parameters; the node emitters
    // read P.v[slot] for those, so editing them needs only a buffer re-upload,
    // not a re-emit/recompile. Identical slot layout to the CPU/GPU-IR buffer.
    if (!res.param_bindings.empty()) {
        src << "layout(std430, set = 0, binding = 3) readonly buffer Params {\n"
            << "    float v[];\n"
            << "} P;\n";
    }
    src << "\n"
        << "layout(push_constant) uniform Push {\n"
        << "    vec3  cam_pos;   float fov_scale;\n"
        << "    vec3  cam_fwd;   float sphere_radius;\n"
        << "    vec3  cam_right; float light_count;\n"
        << "    vec3  cam_up;    float pad_w;\n"
        << "    vec4  lights[4];           // xyz = pos, w = intensity\n"
        << "    vec4  light_colors[4];     // xyz = rgb tint, w unused\n"
        << "    int   width;\n"
        << "    int   height;\n"
        << "    float projection_mode;     // 0 = perspective, 1 = orthographic\n"
        << "    float ortho_size;          // half-height in world units (ortho only)\n"
        << "    float accum_blend;         // temporal denoise: mix(prev, new, accum_blend)\n"
        << "    float frame_seed;          // per-frame jitter offset for accumulation\n"
        << "    int   tile_x0;             // tile origin x (absolute pixel)\n"
        << "    int   tile_y0;             // tile origin y\n"
        << "    int   tile_x1;             // tile end x (exclusive; 0 = width)\n"
        << "    int   tile_y1;             // tile end y (exclusive; 0 = height)\n"
        << "} pc;\n"
        << "\n";

    // Per-mesh trilinear sample functions. One function per MeshSDFNode
    // appearance so the bbox / resolution / offset can be constant-folded
    // by the SPIR-V compiler.
    for (const auto& m : accum.meshes) {
        int N = m.res;
        src << "// MeshSDF #" << m.gpu_index
            << ": " << N << "^3 voxels at offset " << m.offset << "\n"
            << "float sample_mesh_" << m.gpu_index
            << "(float x, float y, float z) {\n"
            // World → voxel coords.
            << "    float fx = (x - " << flit(m.bmin[0]) << ") / "
            << flit(m.cell[0]) << ";\n"
            << "    float fy = (y - " << flit(m.bmin[1]) << ") / "
            << flit(m.cell[1]) << ";\n"
            << "    float fz = (z - " << flit(m.bmin[2]) << ") / "
            << flit(m.cell[2]) << ";\n"
            << "    fx = clamp(fx, 0.0, " << (N - 1) << ".0);\n"
            << "    fy = clamp(fy, 0.0, " << (N - 1) << ".0);\n"
            << "    fz = clamp(fz, 0.0, " << (N - 1) << ".0);\n"
            << "    int ix = int(fx); int iy = int(fy); int iz = int(fz);\n"
            << "    ix = min(ix, " << (N - 2) << "); iy = min(iy, "
            << (N - 2) << "); iz = min(iz, " << (N - 2) << ");\n"
            << "    float tx = fx - float(ix);\n"
            << "    float ty = fy - float(iy);\n"
            << "    float tz = fz - float(iz);\n"
            // Load 8 corners. Index = offset + (k*N + j)*N + i.
            << "    int base = " << m.offset << ";\n"
            << "    int N1 = " << N << ";\n"
            << "    int i0 = ix, j0 = iy, k0 = iz;\n"
            << "    int i1 = ix + 1, j1 = iy + 1, k1 = iz + 1;\n"
            << "    float c000 = mesh_data.mesh_voxels[base + (k0*N1 + j0)*N1 + i0];\n"
            << "    float c100 = mesh_data.mesh_voxels[base + (k0*N1 + j0)*N1 + i1];\n"
            << "    float c010 = mesh_data.mesh_voxels[base + (k0*N1 + j1)*N1 + i0];\n"
            << "    float c110 = mesh_data.mesh_voxels[base + (k0*N1 + j1)*N1 + i1];\n"
            << "    float c001 = mesh_data.mesh_voxels[base + (k1*N1 + j0)*N1 + i0];\n"
            << "    float c101 = mesh_data.mesh_voxels[base + (k1*N1 + j0)*N1 + i1];\n"
            << "    float c011 = mesh_data.mesh_voxels[base + (k1*N1 + j1)*N1 + i0];\n"
            << "    float c111 = mesh_data.mesh_voxels[base + (k1*N1 + j1)*N1 + i1];\n"
            // Trilinear lerp.
            << "    float a00 = mix(c000, c100, tx);\n"
            << "    float a10 = mix(c010, c110, tx);\n"
            << "    float a01 = mix(c001, c101, tx);\n"
            << "    float a11 = mix(c011, c111, tx);\n"
            << "    float b0  = mix(a00,  a10,  ty);\n"
            << "    float b1  = mix(a01,  a11,  ty);\n"
            << "    float sampled = mix(b0, b1, tz);\n"
            // Outside-bbox distance correction (matches CPU codegen path).
            << "    float ex = max(max(" << flit(m.bmin[0]) << " - x, 0.0),"
            <<                   " max(x - " << flit(m.bmax[0]) << ", 0.0));\n"
            << "    float ey = max(max(" << flit(m.bmin[1]) << " - y, 0.0),"
            <<                   " max(y - " << flit(m.bmax[1]) << ", 0.0));\n"
            << "    float ez = max(max(" << flit(m.bmin[2]) << " - z, 0.0),"
            <<                   " max(z - " << flit(m.bmax[2]) << ", 0.0));\n"
            << "    return sampled + sqrt(ex*ex + ey*ey + ez*ez);\n"
            << "}\n\n";
    }

    // Per-texture sample functions. We pack 4 bytes (RGBA) per uint and
    // read with bit shifts. Triplanar samples three times — once per
    // canonical plane — and blends by the normal's axis weights. The
    // normal is computed via central differences over scene_sdf right
    // inside the sampler so scene_albedo() remains self-sufficient.
    if (res.texture_count > 0) {
        // Forward-declare scene_sdf since _tex_normal calls it but
        // scene_sdf's body is emitted later.
        src << "float scene_sdf(float x, float y, float z);\n\n";
        // One generic unpacked-pixel helper.
        src << "vec3 _unpack_rgb(uint packed) {\n"
            << "    float r = float((packed)       & 0xFFu) / 255.0;\n"
            << "    float g = float((packed >>  8) & 0xFFu) / 255.0;\n"
            << "    float b = float((packed >> 16) & 0xFFu) / 255.0;\n"
            << "    return vec3(r, g, b);\n"
            << "}\n\n";
        // Normal helper that re-derives surface normal from scene_sdf.
        src << "vec3 _tex_normal(vec3 p) {\n"
            << "    const float h = 1e-3;\n"
            << "    float dx = scene_sdf(p.x+h, p.y, p.z) - scene_sdf(p.x-h, p.y, p.z);\n"
            << "    float dy = scene_sdf(p.x, p.y+h, p.z) - scene_sdf(p.x, p.y-h, p.z);\n"
            << "    float dz = scene_sdf(p.x, p.y, p.z+h) - scene_sdf(p.x, p.y, p.z-h);\n"
            << "    return normalize(vec3(dx, dy, dz));\n"
            << "}\n\n";
        for (std::size_t i = 0; i < tex_metas.size(); ++i) {
            const auto& tm = tex_metas[i];
            int off_px = tm.offset / 4;  // texture_pixels is byte-array;
                                         // we read it as uint32 -> divide by 4
            src << "// Texture #" << i << ": " << tm.w << "x" << tm.h
                << " at byte offset " << tm.offset << "\n"
                << "vec3 _sample_tex_" << i << "_uv(vec2 uv) {\n"
                << "    uv = fract(uv);\n"
                << "    int u = int(uv.x * " << tm.w << ".0);\n"
                << "    int v = int(uv.y * " << tm.h << ".0);\n"
                << "    u = clamp(u, 0, " << (tm.w - 1) << ");\n"
                << "    v = clamp(v, 0, " << (tm.h - 1) << ");\n"
                // RGBA8 → uint32: 4 channels per pixel; tex_data.texture_pixels
                // is declared as uint[]; we packed by byte index/4.
                << "    int idx = " << off_px << " + v * " << tm.w << " + u;\n"
                << "    return _unpack_rgb(tex_data.texture_pixels[idx]);\n"
                << "}\n\n"
                << "vec3 triplanar_sample_" << i
                << "(vec3 p, float scale) {\n"
                << "    // Raise the per-axis normal weights to the 4th power so\n"
                << "    // axis-aligned regions get sharp, single-plane sampling\n"
                << "    // and diagonals blend over a narrower band. Compare\n"
                << "    // n vs n^4: at 45° the linear weight is 0.5, the\n"
                << "    // pow-4 weight is 0.0625 — much less diamond ghosting.\n"
                << "    vec3 n = abs(_tex_normal(p));\n"
                << "    n = pow(n, vec3(4.0));\n"
                << "    float wsum = n.x + n.y + n.z + 1e-6;\n"
                << "    n /= wsum;\n"
                << "    vec3 q = p * scale;\n"
                << "    vec3 cx = _sample_tex_" << i << "_uv(q.zy);\n"
                << "    vec3 cy = _sample_tex_" << i << "_uv(q.xz);\n"
                << "    vec3 cz = _sample_tex_" << i << "_uv(q.xy);\n"
                << "    return cx * n.x + cy * n.y + cz * n.z;\n"
                << "}\n\n";
        }
    }

    // User template functions: frep_tmpl_<name>(params..., x, y, z), emitted
    // before scene_sdf (whose body calls them) in definition order so a template
    // may call an earlier one. Empty when the scene defines no templates.
    if (!scene.templates().empty())
        CustomExprNode::emit_templates_glsl(src, scene.templates());

    // Level 2 instancing: shared subprogram definitions, emitted before
    // scene_sdf so its body can call them. Empty when no instances are present.
    src << inst_funcs.defs.str();
    src << "float scene_sdf(float x, float y, float z) {\n"
        << sdf_body.str()
        << "}\n"
        << "\n"
        << "float scene_sdf_v(vec3 p) { return scene_sdf(p.x, p.y, p.z); }\n";
    if (use_interval_cull) {
        // Interval SDF for the tile cull: vec2(lo,hi) bound of the field over a
        // box. A single CustomExpr uses the expression interval emitter directly;
        // any other node tree uses the per-node interval emitter, combining the
        // visible objects with min() exactly as scene_sdf does.
        src << gpu::glsl_interval_prelude()
            << "vec2 sdf_ival(vec3 lo, vec3 hi) {\n";
        if (single_ce) {
            gpu::GlslIntervalEmitter ie;
            std::string ivres, ivbody = ie.emit(**single_ce, ivres);
            src << ivbody << "    return " << ivres << ";\n";
        } else {
            src << "    vec2 best = vec2(1e30, 1e30);\n";
            for (const auto& [id, obj] : scene.objects()) {
                if (!obj.visible || !obj.geometry) continue;
                gpu::NodeIntervalEmitter ne;
                std::string res, body = ne.emit(*obj.geometry, res);
                src << "    {\n" << body
                    << "        best = min(best, " << res << ");\n"
                    << "    }\n";
            }
            src << "    return best;\n";
        }
        src << "}\n";
    }
    src << "\n"
        << "vec3 scene_albedo(float x, float y, float z) {\n"
        << albedo_body.str()
        << "}\n"
        << "\n"
        // Per-object material parameters: (roughness, metallic). Mirrors
        // the structure of scene_albedo but returns just the PBR pair.
        // Used by shade() so each object gets its own Cook-Torrance
        // response instead of the project-wide hardcoded defaults.
        << "vec2 scene_pbr(float x, float y, float z) {\n"
        << pbr_body.str()
        << "}\n"
        << "\n";

    // ── Analytic normals via dual-number AD ─────────────────────────────────
    // A Dual carries a value and its gradient. Composing the SDF in dual
    // arithmetic yields the exact analytic gradient, so the normal matches
    // the CPU's forward-mode AD instead of finite differences. Emitted only
    // when every object had a dual emitter (analytic_normals); otherwise we
    // keep the central-difference normal below.
    if (analytic_normals) {
        src << "struct Dual { float v; vec3 g; };\n"
            << "Dual d_add(Dual a, Dual b){ return Dual(a.v+b.v, a.g+b.g); }\n"
            << "Dual d_sub(Dual a, Dual b){ return Dual(a.v-b.v, a.g-b.g); }\n"
            << "Dual d_mul(Dual a, Dual b){ return Dual(a.v*b.v, a.v*b.g + b.v*a.g); }\n"
            << "Dual d_div(Dual a, Dual b){ float inv=1.0/b.v; return Dual(a.v*inv, (a.g*b.v - b.g*a.v)*(inv*inv)); }\n"
            << "Dual d_add_s(Dual a, float s){ return Dual(a.v+s, a.g); }\n"
            << "Dual d_sub_s(Dual a, float s){ return Dual(a.v-s, a.g); }\n"
            << "Dual d_mul_s(Dual a, float s){ return Dual(a.v*s, a.g*s); }\n"
            << "Dual d_neg(Dual a){ return Dual(-a.v, -a.g); }\n"
            << "Dual d_sqrt(Dual a){ float s=sqrt(max(a.v,1e-12)); return Dual(s, a.g*(0.5/s)); }\n"
            << "Dual d_sin(Dual a){ return Dual(sin(a.v), cos(a.v)*a.g); }\n"
            << "Dual d_cos(Dual a){ return Dual(cos(a.v), -sin(a.v)*a.g); }\n"
            // min/max: pick the branch's value AND gradient (subgradient).
            << "Dual d_min(Dual a, Dual b){ return (a.v < b.v) ? a : b; }\n"
            << "Dual d_max(Dual a, Dual b){ return (a.v > b.v) ? a : b; }\n"
            // max/clamp against scalars, and a constant 1 (gradient zero) —
            // used by the taper deformation to mirror the CPU's clamped
            // scale factor and Lipschitz divisor exactly.
            << "Dual d_max_s(Dual a, float s){ return (a.v > s) ? a : Dual(s, vec3(0.0)); }\n"
            << "Dual d_clamp_s(Dual a, float lo, float hi){\n"
            << "    if (a.v < lo) return Dual(lo, vec3(0.0));\n"
            << "    if (a.v > hi) return Dual(hi, vec3(0.0));\n"
            << "    return a;\n"
            << "}\n"
            << "Dual d_one(){ return Dual(1.0, vec3(0.0)); }\n"
            // |p| of dual coords: sqrt(x^2+y^2+z^2), gradient via chain rule.
            << "Dual d_len3(Dual x, Dual y, Dual z){\n"
            << "    Dual s = d_add(d_add(d_mul(x,x), d_mul(y,y)), d_mul(z,z));\n"
            << "    return d_sqrt(s);\n"
            << "}\n"
            // dot(p, const n): linear; value = sum, gradient = sum of g*n.
            << "Dual d_dot3(Dual x, Dual y, Dual z, float nx, float ny, float nz){\n"
            << "    return Dual(x.v*nx + y.v*ny + z.v*nz, x.g*nx + y.g*ny + z.g*nz);\n"
            << "}\n"
            // Box SDF in dual form: q = abs(p)-h; outside + inside terms.
            << "Dual d_box(Dual x, Dual y, Dual z, float hx, float hy, float hz){\n"
            << "    Dual qx = d_sub_s(Dual(abs(x.v), sign(x.v)*x.g), hx);\n"
            << "    Dual qy = d_sub_s(Dual(abs(y.v), sign(y.v)*y.g), hy);\n"
            << "    Dual qz = d_sub_s(Dual(abs(z.v), sign(z.v)*z.g), hz);\n"
            << "    Dual mx = d_max(qx, Dual(0.0, vec3(0.0)));\n"
            << "    Dual my = d_max(qy, Dual(0.0, vec3(0.0)));\n"
            << "    Dual mz = d_max(qz, Dual(0.0, vec3(0.0)));\n"
            << "    Dual outside = d_len3(mx, my, mz);\n"
            << "    Dual inside = d_min(d_max(qx, d_max(qy, qz)), Dual(0.0, vec3(0.0)));\n"
            << "    return d_add(outside, inside);\n"
            << "}\n"
            // IQ smin in dual form (matches the CPU formula incl. the *0.5).
            << "Dual d_smin(Dual a, Dual b, float k){\n"
            << "    if (k <= 0.0) return d_min(a, b);\n"
            << "    float kk = k * 2.0;\n"
            << "    Dual d = d_sub(a, b);\n"
            << "    float adv = abs(d.v);\n"
            << "    vec3  adg = (d.v >= 0.0) ? d.g : -d.g;\n"   // d/dp |a-b|
            << "    Dual ad = Dual(adv, adg);\n"
            << "    float hv = max(kk - adv, 0.0) / kk;\n"
            << "    vec3  hg = (kk - adv > 0.0) ? (-adg / kk) : vec3(0.0);\n"
            << "    Dual h = Dual(hv, hg);\n"
            << "    Dual mn = d_min(a, b);\n"
            << "    Dual h3 = d_mul(d_mul(h, h), h);\n"          // h^3
            << "    Dual corr = d_mul_s(h3, kk / 6.0);\n"
            << "    return d_sub(mn, corr);\n"
            << "}\n"
            << inst_funcs.grad_defs.str()   // Level 2: shared grad subprograms
            << "Dual scene_sdf_grad(vec3 p) {\n"
            << grad_body.str()
            << "    return best_grad;\n"
            << "}\n"
            << "vec3 scene_normal(vec3 p) {\n"
            << "    vec3 g = scene_sdf_grad(p).g;\n"
            // Explicit sqrt(dot) + divide, matching the CPU's normalization
            // (sqrt of len2, then component/len) rather than GLSL length()/
            // normalize(), which may lower to a reduced-precision rsqrt.
            << "    float gl = sqrt(g.x*g.x + g.y*g.y + g.z*g.z);\n"
            << "    if (gl < 1e-6) return vec3(0.0, 1.0, 0.0);\n"
            << "    return g / gl;\n"
            << "}\n\n";
    } else {
        src << "vec3 scene_normal(vec3 p) {\n"
            << "    const float h = 1e-3;\n"
            << "    float dx = scene_sdf_v(p + vec3(h,0,0)) - scene_sdf_v(p - vec3(h,0,0));\n"
            << "    float dy = scene_sdf_v(p + vec3(0,h,0)) - scene_sdf_v(p - vec3(0,h,0));\n"
            << "    float dz = scene_sdf_v(p + vec3(0,0,h)) - scene_sdf_v(p - vec3(0,0,h));\n"
            << "    vec3 g = vec3(dx, dy, dz);\n"
            << "    float gl = length(g);\n"
            << "    if (gl < 1e-6) return vec3(0.0, 1.0, 0.0);\n"
            << "    return g / gl;\n"
            << "}\n\n";
    }

    // Reflectivity lookup — only emitted when reflections are enabled.
    if (cfg.max_bounces > 0) {
        src << "// Per-object mirror reflectivity in [0,1]. Mirrors the\n"
            << "// structure of scene_pbr. Only present when reflections\n"
            << "// are on (TracerConfig.max_bounces > 0).\n"
            << "float scene_reflectivity(float x, float y, float z) {\n"
            << refl_body.str()
            << "}\n"
            << "float scene_reflectivity_v(vec3 p) {\n"
            << "    return scene_reflectivity(p.x, p.y, p.z);\n"
            << "}\n"
            << "\n";
    }

    // Sky/background colour as a function so the reflection ray (and
    // main()'s miss branch) sample the same gradient. Sky constants
    // baked from cfg.
    src << "vec3 sky_color_s(float s) {\n"
        << "    return mix(vec3("
        << flit(cfg.sky_horizon[0]) << ", " << flit(cfg.sky_horizon[1]) << ", "
        << flit(cfg.sky_horizon[2]) << "), vec3("
        << flit(cfg.sky_top[0]) << ", " << flit(cfg.sky_top[1]) << ", "
        << flit(cfg.sky_top[2]) << "), clamp(s, 0.0, 1.0));\n"
        << "}\n"
        // Primary-ray sky uses the SAME gradient input as the CPU JIT path:
        // s = 0.5 + 0.5 * uv_y, where uv_y is the NDC vertical coordinate
        // (range [-1,1]), NOT the normalized ray direction's y. They differ
        // because normalize() shrinks the y component, so a dir.y-based
        // gradient ran at a different rate and made the GPU sky a touch
        // lighter than the IR paths across the whole background — a constant
        // offset over a large area. Reflection rays have no NDC, so they fall
        // back to the direction-based form below.
        << "vec3 sky_color(vec3 dir) {\n"
        << "    return sky_color_s(0.5 + 0.5 * dir.y);\n"
        << "}\n"
        << "\n";

    src // Cook-Torrance microfacet BRDF — same model as the CPU JIT path.
        // Loops over the lights[] array; intensity = 0 means "unused" so
        // empty slots cost almost nothing.
        << "const float PI = 3.14159265359;\n"
        << "\n"
        << "// GGX normal distribution function.\n"
        << "float _D_GGX(float ndoth, float a) {\n"
        << "    float a2 = a * a;\n"
        << "    float d  = (ndoth * ndoth) * (a2 - 1.0) + 1.0;\n"
        << "    return a2 / max(PI * d * d, 1e-6);\n"
        << "}\n"
        << "// Smith / GGX geometry (Schlick approximation, paired).\n"
        << "float _G_Smith(float ndotv, float ndotl, float a) {\n"
        << "    float k = (a + 1.0); k = (k * k) * 0.125;\n"
        << "    float gv = ndotv / (ndotv * (1.0 - k) + k);\n"
        << "    float gl = ndotl / (ndotl * (1.0 - k) + k);\n"
        << "    return gv * gl;\n"
        << "}\n"
        << "// Fresnel (Schlick).\n"
        << "vec3 _F_Schlick(float vdoth, vec3 f0) {\n"
        // Match the CPU JIT path's manual expansion (1-vdoth)^5 =
        // ((1-vdoth)^2)^2 * (1-vdoth), NOT pow(x,5.0). pow() is evaluated as
        // exp(5*log(x)) and rounds differently from the exact chain of
        // multiplies the CPU emits, leaving a small per-pixel offset wherever
        // Fresnel contributes (i.e. all lit pixels, via the kd=(1-F) diffuse
        // weight too — not just specular highlights).
        << "    float m = 1.0 - vdoth;\n"
        << "    float m2 = m * m;\n"
        << "    float p = m2 * m2 * m;\n"
        << "    return f0 + (vec3(1.0) - f0) * p;\n"
        << "}\n"
        << "\n";

    // ── shadow ray emission ─────────────────────────────────────────────────
    // Three modes baked from TracerConfig.enable_shadows:
    //   true  → IQ-style soft shadow accumulator (mirrors CPU JIT's
    //           emit_shadow()), penumbra width driven by cfg.shadow_softness
    //   false → constant-fold-friendly stub returning 1.0 everywhere
    //
    // We bake the choice into the GLSL source rather than gating at
    // runtime on a uniform — uniforms in a hot loop cost more than a
    // recompile, and the higher-level viewport already triggers a
    // pipeline rebuild when the config drifts (see scene-hash in
    // vulkan_viewport.cpp).
    if (cfg.enable_shadows) {
        src << "// Soft shadow ray-march (Inigo Quilez 2010 algorithm).\n"
            << "// Returns 1.0 = fully lit, 0.0 = fully occluded;\n"
            << "// in-between values are penumbra from shadow_softness="
            << flit(cfg.shadow_softness) << ".\n"
            << "//\n"
            << "// Two bugs we deliberately avoid:\n"
            << "//   1) `t += max(d, eps)` floor: forcing a minimum step\n"
            << "//      causes overshoot of nearby surfaces (esp. floor\n"
            << "//      planes seen at grazing angle), producing spurious\n"
            << "//      shadow hits that appear as dark blobs/holes on the\n"
            << "//      ground near the horizon. We use raw `t += d`.\n"
            << "//   2) Too-tight termination eps: a shadow ray that\n"
            << "//      starts near a surface and travels parallel to it\n"
            << "//      keeps reading sub-eps SDF values and reports the\n"
            << "//      same surface as an occluder. We use a slightly\n"
            << "//      relaxed eps and start the ray at t=0.05.\n"
            << "float _shadow_ray(vec3 origin, vec3 dir, float max_t) {\n";
        if (cfg.shadow_samples > 1) {
            // Area-light soft shadows: when multi-sampling, each ray is
            // a HARD binary occlusion test (0 or 1). The softness comes
            // entirely from averaging many rays aimed at jittered points
            // across the light's disk (see _soft_shadow). Mixing the IQ
            // penumbra term in here would double up and wash out the
            // sample-count effect, making "Shadow samples" / "Light
            // radius" look like they do nothing.
            src << "    float t = 0.05;\n"
                << "    for (int i = 0; i < " << cfg.shadow_steps << "; ++i) {\n"
                << "        vec3 q = origin + dir * t;\n"
                << "        float d = scene_sdf_v(q);\n"
                << "        if (d < 0.002) return 0.0;\n"
                << "        t += d * " << flit(cfg.safety_factor) << ";\n"
                << "        if (t > max_t) break;\n"
                << "    }\n"
                << "    return 1.0;\n"
                << "}\n";
        } else {
            // Single-sample: keep the cheap IQ penumbra approximation so
            // a 1-sample shadow still has a soft edge (controlled by the
            // "Soft shadows" softness slider).
            src << "    float t = 0.05;\n"
                << "    float k = 1.0;\n"
                << "    for (int i = 0; i < " << cfg.shadow_steps << "; ++i) {\n"
                << "        vec3 q = origin + dir * t;\n"
                << "        float d = scene_sdf_v(q);\n"
                << "        if (d < 0.002) return 0.0;\n"
                << "        k = min(k, " << flit(cfg.shadow_softness) << " * d / t);\n"
                << "        t += d * " << flit(cfg.safety_factor) << ";\n"
                << "        if (t > max_t) break;\n"
                << "    }\n"
                << "    return clamp(k, 0.0, 1.0);\n"
                << "}\n";
        }
    } else {
        src << "// Shadows disabled via TracerConfig.enable_shadows = false.\n"
            << "float _shadow_ray(vec3 origin, vec3 dir, float max_t) {\n"
            << "    return 1.0;\n"
            << "}\n";
    }

    // ── soft-shadow multi-sample wrapper ────────────────────────────────────
    // Casts `shadow_samples` shadow rays toward jittered points on a
    // virtual spherical area light of half-extent `shadow_light_radius`
    // centred on the light, and averages the occlusion. With
    // shadow_samples == 1 this collapses to a single _shadow_ray call
    // (the loop runs once, jitter is zero) so there's no overhead when
    // the feature is off. A small hash-based per-pixel rotation of the
    // sample pattern decorrelates neighbouring pixels — combined with
    // the real-time temporal accumulation this converges to smooth
    // penumbrae.
    if (cfg.enable_shadows && cfg.shadow_samples > 1) {
        src << "// Cheap hash for per-pixel jitter decorrelation.\n"
            << "float _hash12(vec2 p) {\n"
            << "    vec3 p3 = fract(vec3(p.xyx) * 0.1031);\n"
            << "    p3 += dot(p3, p3.yzx + 33.33);\n"
            << "    return fract((p3.x + p3.y) * p3.z);\n"
            << "}\n"
            << "float _soft_shadow(vec3 origin, vec3 lpos, vec2 seed) {\n"
            << "    float sum = 0.0;\n"
            << "    float base = _hash12(seed) * 6.2831853;\n"
            << "    // Second decorrelated hash to jitter the *radial*\n"
            << "    // position of the samples, not just the angle. Without\n"
            << "    // this the N samples always land on the same N rings,\n"
            << "    // so a 4-sample area light produces visible radial\n"
            << "    // banding along the penumbra that temporal\n"
            << "    // accumulation (which only rotated the spiral) never\n"
            << "    // fully smoothed. The offset is fed the same seed (so\n"
            << "    // it varies per pixel and per frame) plus a constant\n"
            << "    // so it doesn't correlate with `base`.\n"
            << "    float rjit = _hash12(seed + 17.0);\n"
            << "    const int N = " << cfg.shadow_samples << ";\n"
            << "    for (int s = 0; s < N; ++s) {\n"
            << "        // Distribute sample offsets on a disk around the\n"
            << "        // light using a cheap golden-angle spiral, with the\n"
            << "        // stratified radius jittered within its ring.\n"
            << "        float fi = (float(s) + rjit) / float(N);\n"
            << "        float ang = base + float(s) * 2.39996323;\n"
            << "        float rad = sqrt(fi) * " << flit(cfg.shadow_light_radius) << ";\n"
            << "        vec3 off = vec3(cos(ang) * rad, sin(ang) * rad,\n"
            << "                        (fi - 0.5) * rad);\n"
            << "        vec3 jl = lpos + off;\n"
            << "        vec3 ldir = jl - origin;\n"
            << "        float ldist = length(ldir);\n"
            << "        ldir /= max(ldist, 1e-4);\n"
            << "        sum += _shadow_ray(origin, ldir, ldist);\n"
            << "    }\n"
            << "    return sum / float(N);\n"
            << "}\n";
    } else {
        // Single-sample passthrough — keeps shade() call sites uniform.
        src << "float _soft_shadow(vec3 origin, vec3 lpos, vec2 seed) {\n"
            << "    vec3 ldir = lpos - origin;\n"
            << "    float ldist = length(ldir);\n"
            << "    ldir /= max(ldist, 1e-4);\n"
            << "    return _shadow_ray(origin, ldir, ldist);\n"
            << "}\n";
    }

    // ── ambient occlusion emission ──────────────────────────────────────────
    // Mirrors the CPU JIT path's emit_ao(): cfg.ao_samples steps along
    // the surface normal, each of length cfg.ao_step, accumulated with
    // 0.5^i weighting and scaled by cfg.ao_strength. When AO is
    // disabled the stub returns 1.0 and the multiplier optimises away.
    if (cfg.enable_ao) {
        src << "\n// Ambient occlusion accumulator.\n"
            << "// " << cfg.ao_samples << " samples, step="
            << flit(cfg.ao_step) << ", strength="
            << flit(cfg.ao_strength) << ".\n"
            << "float _ao(vec3 p, vec3 n) {\n"
            << "    float total = 0.0;\n"
            << "    float w = 1.0;\n";
        for (int i = 1; i <= cfg.ao_samples; ++i) {
            float dist = i * cfg.ao_step;
            src << "    { vec3 s = p + n * " << flit(dist) << ";\n"
                << "      float h = scene_sdf_v(s);\n"
                << "      total += (" << flit(dist) << " - h) * w;\n"
                << "      w *= 0.5; }\n";
        }
        src << "    float cl = clamp(total, 0.0, 1.0);\n"
            << "    return 1.0 - " << flit(cfg.ao_strength) << " * cl;\n"
            << "}\n\n";
    } else {
        src << "\n// AO disabled via TracerConfig.enable_ao = false.\n"
            << "float _ao(vec3 p, vec3 n) { return 1.0; }\n\n";
    }

    // ── shade() emission ────────────────────────────────────────────────────
    // Branches between CookTorrance (full microfacet BRDF, energy-
    // conserving) and Blinn-Phong (cheaper, classical). Both honour the
    // per-material `roughness` and `metallic` fields, both call _ao()
    // for ambient occlusion and _shadow_ray() for shadow attenuation
    // (their dispatches were already specialised above).
    //
    // The CookTorrance path mirrors the CPU JIT path in codegen.cpp
    // (emit_shading_cook_torrance). The Blinn-Phong path likewise
    // mirrors emit_shading_blinn_phong: a single specular term using
    // half-vector pow(ndoth, shininess) blended by a metallic factor.
    src << "vec3 shade(vec3 p, vec3 n, vec3 view_dir) {\n"
        << "    vec3  albedo = scene_albedo(p.x, p.y, p.z);\n"
        << "    vec2  rm     = scene_pbr(p.x, p.y, p.z);\n"
        << "    float roughness = rm.x;\n"
        << "    float metallic  = rm.y;\n"
        << "    float ao_v = _ao(p, n);\n"
        << "    vec3  Lo = albedo * 0.08 * ao_v;   // ambient × AO\n"
        << "    int   nL = int(pc.light_count + 0.5);\n"
        << "    nL = clamp(nL, 0, 4);\n";

    if (cfg.shading_model == TracerConfig::ShadingModel::CookTorrance) {
        src << "    // Cook-Torrance microfacet BRDF.\n"
            << "    float a  = roughness * roughness;\n"
            << "    vec3  f0 = mix(vec3(0.04), albedo, metallic);\n"
            << "    for (int i = 0; i < nL; ++i) {\n"
            << "        vec3  lpos = pc.lights[i].xyz;\n"
            << "        float lI   = pc.lights[i].w;\n"
            << "        if (lI <= 0.0) continue;\n"
            << "        vec3  ldir = lpos - p;\n"
            << "        float ldist = length(ldir);\n"
            << "        ldir /= max(ldist, 1e-4);\n"
            << "        float ndotl = max(dot(n, ldir), 0.0);\n"
            << "        if (ndotl <= 0.0) continue;\n"
            << "        float shadow = _soft_shadow(p + n * 0.01, lpos, p.xy + p.z + vec2(pc.frame_seed, pc.frame_seed * 1.618));\n"
            << "        vec3  half_v = normalize(ldir + view_dir);\n"
            << "        float ndotv  = max(dot(n, view_dir), 1e-4);\n"
            << "        float ndoth  = max(dot(n, half_v),   0.0);\n"
            << "        float vdoth  = max(dot(view_dir, half_v), 0.0);\n"
            << "        float D = _D_GGX(ndoth, a);\n"
            << "        vec3  F = _F_Schlick(vdoth, f0);\n"
        // Specular visibility, computed in the SAME algebraically-cancelled
        // form as the CPU JIT path (codegen.cpp): the naive
        //   D·G·F / (4·ndotv·ndotl),  G = (ndotv/gv)·(ndotl/gl)
        // and the cancelled
        //   D·F / (4·gv·gl),  gv = ndotv·(1-k)+k,  gl = ndotl·(1-k)+k
        // are algebraically identical, but differ in float: the naive form
        // computes ndotv·ndotl in both numerator and denominator, leaving a
        // small per-pixel rounding offset on every shaded pixel (the constant
        // mean |Δ| ≈ 0.022 vs the CPU). Using the cancelled form here removes
        // that systematic difference and also matches the CPU's clamp point
        // (cap before ×PI). k = (a+1)²/8 with a = roughness² as in _G_Smith.
            << "        float kg = (a + 1.0); kg = (kg * kg) * 0.125;\n"
            << "        float gv = max(ndotv * (1.0 - kg) + kg, 1e-5);\n"
            << "        float gl = max(ndotl * (1.0 - kg) + kg, 1e-5);\n"
            << "        vec3 specular = F * D / max(4.0 * gv * gl, 1e-5);\n"
        // Firefly clamp — mirror the CPU's cap (applied before the ×PI
        // factor, so the value here is capped at 8.0 like the CPU's 8·PI
        // after its own ×PI). Removes the grazing-angle white rim.
        << "        specular = min(specular, vec3(8.0));\n"
            << (cfg.enable_specular ? "" : "        specular = vec3(0.0);\n")
            << "        vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);\n"
            // PBR diffuse normalisation (kd*albedo/PI) gives physically
            // correct but visually dim results when the light intensity is
            // in unitless [0,1] terms — most authored scenes assume light
            // intensities sized for non-energy-conserving Lambert. We
            // compensate by treating `lI` as already containing the PI
            // factor, i.e. effectively making `lI_effective = lI * PI`.
            // See Frostbite's Moving Frostbite to PBR PDF.
            << "        vec3 lcol = pc.light_colors[i].xyz;\n"
            << "        Lo += (kd * albedo + specular * PI)\n"
            << "            * ndotl * shadow * lI * lcol;\n"
            << "    }\n";
    } else {
        // Blinn-Phong. Use a shininess derived from roughness so the
        // user-visible result of dragging the roughness slider matches
        // the CookTorrance intuition (higher roughness → broader
        // highlight). Shininess range chosen to land sane highlights
        // across roughness ∈ [0, 1].
        src << "    // Blinn-Phong specular.\n"
            << "    float shininess = mix(128.0, 4.0, roughness);\n"
            << "    vec3  spec_tint = mix(vec3(1.0), albedo, metallic);\n"
            << "    for (int i = 0; i < nL; ++i) {\n"
            << "        vec3  lpos = pc.lights[i].xyz;\n"
            << "        float lI   = pc.lights[i].w;\n"
            << "        if (lI <= 0.0) continue;\n"
            << "        vec3  ldir = lpos - p;\n"
            << "        float ldist = length(ldir);\n"
            << "        ldir /= max(ldist, 1e-4);\n"
            << "        float ndotl = max(dot(n, ldir), 0.0);\n"
            << "        if (ndotl <= 0.0) continue;\n"
            << "        float shadow = _soft_shadow(p + n * 0.01, lpos, p.xy + p.z + vec2(pc.frame_seed, pc.frame_seed * 1.618));\n"
            << "        vec3  half_v = normalize(ldir + view_dir);\n"
            << "        float ndoth = max(dot(n, half_v), 0.0);\n"
            << "        float spec_pow = pow(ndoth, shininess) * (1.0 - roughness * 0.5);\n"
            << "        vec3  diffuse  = albedo * ndotl * (1.0 - metallic);\n"
            << "        vec3  specular = spec_tint * spec_pow;\n"
            << "        vec3  lcol = pc.light_colors[i].xyz;\n"
            << "        Lo += (diffuse + specular) * shadow * lI * lcol;\n"
            << "    }\n";
    }

    src << "    return Lo;\n"
        << "}\n"
        << "\n";

    src
        << "void main() {\n"
        << "    // Tile bounds: tile_x1/y1 == 0 means whole frame.\n"
        << "    int tx1 = (pc.tile_x1 > 0) ? pc.tile_x1 : pc.width;\n"
        << "    int ty1 = (pc.tile_y1 > 0) ? pc.tile_y1 : pc.height;\n"
        << "    // Local invocation indexes the tile; absolute pixel adds the origin.\n"
        << "    ivec2 local = ivec2(gl_GlobalInvocationID.xy);\n"
        << "    ivec2 px = local + ivec2(pc.tile_x0, pc.tile_y0);\n"
        ;

    // Workgroup tile cull, emitted BEFORE the bounds-check early-returns so the
    // barrier sits in uniform control flow (a barrier after a divergent return
    // is undefined and hangs some drivers). One invocation bounds the
    // workgroup's frustum slabs with the Lipschitz box rule
    //     f(box) subset [f(c) - L*r, f(c) + L*r],  r = circumradius,
    // and publishes the surviving depth span. A fully culled workgroup ends up
    // with t > t_far, so the march body never runs and the existing miss path
    // shades the background.
    if (cfg.cull_slabs > 0) {
        src << "    // Spread the " << cfg.cull_slabs << " box tests across the 64 lanes\n"
            << "    // (lane s handles slabs s, s+64, ...); lane 0 then reduces the\n"
            << "    // per-slab occupancy to a [t0,t1] span. Avoids serialising every\n"
            << "    // slab on one invocation while the rest wait at the barrier.\n"
            << "    {\n"
            << "        uint lid = gl_LocalInvocationIndex;\n"
            << "        ivec2 wg0 = ivec2(gl_WorkGroupID.xy) * ivec2(8) + ivec2(pc.tile_x0, pc.tile_y0);\n"
            << "        float st = (" << flit(cfg.max_dist) << " - 0.001) / float("
            <<          cfg.cull_slabs << ");\n"
            << "        for (uint s = lid; s < " << cfg.cull_slabs << "u; s += 64u) {\n"
            << "            float ta = 0.001 + st * float(s);\n"
            << "            float tb = ta + st;\n"
            << "            vec3 clo = vec3(1e30), chi = vec3(-1e30);\n"
            << "            for (int c = 0; c < 4; ++c) {\n"
            << "                ivec2 q = wg0 + ivec2((c & 1) * 8, ((c >> 1) & 1) * 8);\n"
            << "                float cu = (2.0 * float(q.x) / float(pc.width) - 1.0)\n"
            << "                         * (float(pc.width) / float(pc.height));\n"
            << "                float cv = 1.0 - 2.0 * float(q.y) / float(pc.height);\n"
            << "                vec3 co, cd;\n"
            << "                if (pc.projection_mode > 0.5) {\n"
            << "                    co = pc.cam_pos + pc.cam_right * (cu * pc.ortho_size)\n"
            << "                                    + pc.cam_up    * (cv * pc.ortho_size);\n"
            << "                    cd = normalize(pc.cam_fwd);\n"
            << "                } else {\n"
            << "                    co = pc.cam_pos;\n"
            << "                    cd = normalize(pc.cam_fwd + pc.cam_right * (cu * pc.fov_scale)\n"
            << "                                             + pc.cam_up    * (cv * pc.fov_scale));\n"
            << "                }\n"
            << "                clo = min(clo, min(co + cd * ta, co + cd * tb));\n"
            << "                chi = max(chi, max(co + cd * ta, co + cd * tb));\n"
            << "            }\n"
            << (use_interval_cull
                ? std::string(
                  "            vec2 fi = sdf_ival(clo, chi);\n"
                  "            sh_occ[s] = (fi.x > 0.0 || fi.y < 0.0) ? 0u : 1u;\n")
                : std::string(
                  "            vec3 cc = 0.5 * (clo + chi);\n"
                  "            float r = ") + flit(cfg.cull_lipschitz) +
                  " * length(0.5 * (chi - clo));\n"
                  "            float fc = scene_sdf_v(cc);\n"
                  "            sh_occ[s] = (fc - r > 0.0 || fc + r < 0.0) ? 0u : 1u;\n")
            << "        }\n"
            << "    }\n"
            << "    barrier();\n"
            << "    memoryBarrierShared();\n"
            << "    if (gl_LocalInvocationIndex == 0u) {\n"
            << "        float st = (" << flit(cfg.max_dist) << " - 0.001) / float("
            <<          cfg.cull_slabs << ");\n"
            << "        float b0 = 1e30, b1 = -1e30;\n"
            << "        for (int s = 0; s < " << cfg.cull_slabs << "; ++s) {\n"
            << "            if (sh_occ[s] == 0u) continue;\n"
            << "            float ta = 0.001 + st * float(s);\n"
            << "            b0 = min(b0, ta); b1 = max(b1, ta + st);\n"
            << "        }\n"
            << "        float mrg = 4.0 * (" << flit(cfg.max_dist) << " - 0.001) / 32.0;\n"
            << "        sh_cull_t0 = b0 - mrg; sh_cull_t1 = b1 + mrg;\n"
            << "    }\n"
            << "    barrier();\n"
            << "    memoryBarrierShared();\n";
    }
    src
        << "    if (px.x >= tx1 || px.y >= ty1) return;\n"
        << "    if (px.x >= pc.width || px.y >= pc.height) return;\n"
        << "    // Ray is computed from the FULL frame so the tile matches the\n"
        << "    // corresponding region of a whole-frame render exactly.\n"
        << "    float u = (2.0 * float(px.x) / float(pc.width)  - 1.0)\n"
        << "            * (float(pc.width) / float(pc.height));\n"
        << "    float v = 1.0 - 2.0 * float(px.y) / float(pc.height);\n"
        << "    // Build the ray. Two projection modes:\n"
        << "    //   - Perspective: all rays share the same origin\n"
        << "    //     (cam_pos), direction diverges through fov_scale.\n"
        << "    //   - Orthographic: rays parallel along cam_fwd,\n"
        << "    //     origin offset within the cam_right / cam_up\n"
        << "    //     plane by ortho_size half-extents.\n"
        << "    vec3 ray_origin;\n"
        << "    vec3 ray_dir;\n"
        << "    if (pc.projection_mode > 0.5) {\n"
        << "        ray_origin = pc.cam_pos\n"
        << "                   + pc.cam_right * (u * pc.ortho_size)\n"
        << "                   + pc.cam_up    * (v * pc.ortho_size);\n"
        << "        ray_dir    = normalize(pc.cam_fwd);\n"
        << "    } else {\n"
        << "        ray_origin = pc.cam_pos;\n"
        << "        ray_dir = normalize(\n"
        << "            pc.cam_fwd\n"
        << "            + pc.cam_right * (u * pc.fov_scale)\n"
        << "            + pc.cam_up    * (v * pc.fov_scale));\n"
        << "    }\n"
        << "    float t = 0.001;\n"
        << "    bool  hit = false;\n"
        << "    vec3  p   = ray_origin;\n"
        << "    float last_d = 1e30;\n";

    if (cfg.cull_slabs > 0) {
        src << "    float t_far = min(" << flit(cfg.max_dist) << ", sh_cull_t1);\n"
            << "    t = max(t, sh_cull_t0);\n";
    } else {
        src << "    float t_far = " << flit(cfg.max_dist) << ";\n";
    }
    src
        << "    // Tracer parameters baked from TracerConfig: max_steps,\n"
        << "    // max_dist, epsilon. Matches the CPU JIT path's tracer\n"
        << "    // for visual parity — including the loop structure: t starts\n"
        << "    // at 0.001 (not 0) and the max_dist bound is checked at the\n"
        << "    // TOP of each iteration (before the SDF eval), exactly as the\n"
        << "    // CPU PHI loop does. These two details shifted the march\n"
        << "    // sequence by a step on silhouettes and were the dominant\n"
        << "    // CPU↔GLSL divergence (a bright fringe on object edges).\n"
        << "    // Enhanced sphere tracing: over-relax the step by omega with an\n"
        << "    // overshoot guard that backs up and latches omega to 1 when a\n"
        << "    // step jumps past a surface (tunnel-proof). Mirrors the CPU JIT\n"
        << "    // march exactly for parity; cuts step-exhausting silhouette rays.\n"
        << "    float step_len = 0.0;\n"
        << "    float omega = " << flit(cfg.over_relax) << ";\n"
        << "    int dbg_steps = 0;\n"
        << "    for (int i = 0; i < " << cfg.max_steps << "; ++i) {\n"
        << "        dbg_steps = i;\n"
        << "        if (t > t_far) break;\n"
        << "        p = ray_origin + ray_dir * t;\n"
        << "        float d = scene_sdf_v(p);\n"
        << "        float radius = d * " << flit(cfg.safety_factor) << ";\n"
        << "        bool sor_fail = (omega > 1.0) && ((radius + last_d) < step_len);\n"
        << "        step_len = sor_fail ? (step_len * (1.0 - omega)) : (radius * omega);\n"
        << "        omega = sor_fail ? 1.0 : omega;\n"
        << "        if (d < " << flit(cfg.epsilon) << " && !sor_fail) { hit = true; last_d = d; break; }\n"
        << "        last_d = d;\n"
        << "        t += step_len;\n"
        << "    }\n"
        << "    // Grazing-ray rescue. Along a silhouette the ray creeps\n"
        << "    // through the thin valley just outside the surface, taking\n"
        << "    // ever-smaller steps, and can exhaust max_steps while still\n"
        << "    // a hair above epsilon. Treating that as a miss returns the\n"
        << "    // bright sky colour, which appears as a light fringe\n"
        << "    // tracing the silhouette wherever the background behind it\n"
        << "    // is dark. If the march stopped close to the surface and\n"
        << "    // within range, count it as a hit — the shaded surface is\n"
        << "    // far closer to the truth than the sky.\n"
        << "    if (!hit && t <= " << flit(cfg.max_dist)
        <<            " && last_d < " << flit(cfg.epsilon * 80.0f) << ") {\n"
        << "        hit = true;\n"
        << "    }\n"
        << "    vec3 color;\n"
        << "    if (hit) {\n"
        << "        vec3 n = scene_normal(p);\n"
        << "        color = shade(p, n, -ray_dir);\n";
    if (cfg.max_bounces > 0) {
        src << "        // ── Reflection bounces ──────────────────────────────\n"
            << "        // Cast a mirror ray per bounce, blending the\n"
            << "        // reflected colour in by the surface's\n"
            << "        // reflectivity, modulated by a Schlick Fresnel\n"
            << "        // term so grazing angles reflect more. `throughput`\n"
            << "        // accumulates the product of reflectivities so a\n"
            << "        // dim mirror seen in another dim mirror contributes\n"
            << "        // proportionally less.\n"
            << "        vec3  refl_origin = p;\n"
            << "        vec3  refl_n      = n;\n"
            << "        vec3  refl_dir    = ray_dir;\n"
            << "        float throughput  = 1.0;\n"
            << "        for (int b = 0; b < " << cfg.max_bounces << "; ++b) {\n"
            << "            float refl = scene_reflectivity_v(refl_origin);\n"
            << "            if (refl <= 0.001) break;\n"
            << "            // Fresnel-Schlick (scalar, F0 = refl): more\n"
            << "            // reflection at grazing angles.\n"
            << "            float cosi = clamp(dot(-refl_dir, refl_n), 0.0, 1.0);\n"
            << "            float fres = refl + (1.0 - refl) * pow(1.0 - cosi, 5.0);\n"
            << "            throughput *= fres;\n"
            << "            if (throughput < 0.01) break;\n"
            << "            // Reflect and march again, offset off the\n"
            << "            // surface to avoid self-intersection.\n"
            << "            refl_dir = reflect(refl_dir, refl_n);\n"
            << "            vec3 ro  = refl_origin + refl_n * 0.01;\n"
            << "            float rt = 0.0;\n"
            << "            bool  rhit = false;\n"
            << "            vec3  rp = ro;\n"
            << "            for (int i = 0; i < " << cfg.max_steps << "; ++i) {\n"
            << "                rp = ro + refl_dir * rt;\n"
            << "                float d = scene_sdf_v(rp);\n"
            << "                if (d < " << flit(cfg.epsilon) << ") { rhit = true; break; }\n"
            << "                if (rt > " << flit(cfg.max_dist) << ") break;\n"
            << "                rt += d * " << flit(cfg.safety_factor) << ";\n"
            << "            }\n"
            << "            vec3 rcol;\n"
            << "            if (rhit) {\n"
            << "                vec3 rn = scene_normal(rp);\n"
            << "                rcol = shade(rp, rn, -refl_dir);\n"
            << "                color = mix(color, rcol, throughput);\n"
            << "                // Continue bouncing from this hit.\n"
            << "                refl_origin = rp;\n"
            << "                refl_n      = rn;\n"
            << "            } else {\n"
            << "                rcol = sky_color(refl_dir);\n"
            << "                color = mix(color, rcol, throughput);\n"
            << "                break;  // ray escaped to sky — no more bounces\n"
            << "            }\n"
            << "        }\n";
    }
    src << "    } else {\n"
        << "        color = sky_color_s(0.5 + 0.5 * v);\n"
        << "    }\n"
        << "    // Defense in depth — clamp first, then explicitly\n"
        << "    // sanitise NaNs (which can survive clamp on some\n"
        << "    // drivers) by replacing them with zero. This is the\n"
        << "    // belt-and-braces fix for the white-pixel artefacts\n"
        << "    // that occasionally appeared at silhouette edges.\n"
        << "    color = clamp(color, 0.0, 1.0);\n"
        << "    if (any(isnan(color))) color = vec3(0.0);\n";
    if (cfg.debug_view == TracerConfig::DebugView::StepHeatmap) {
        // Colour by march-iteration count: blue (few) -> green -> red (many),
        // normalised to max_steps. Reveals where the march is expensive.
        src << "    {\n"
            << "        float frac = clamp(float(dbg_steps) / float(" << cfg.max_steps << "), 0.0, 1.0);\n"
            << "        vec3 cool = vec3(0.0, 0.1, 0.6);\n"
            << "        vec3 mid  = vec3(0.0, 0.8, 0.2);\n"
            << "        vec3 hot  = vec3(0.9, 0.1, 0.0);\n"
            << "        color = frac < 0.5 ? mix(cool, mid, frac * 2.0)\n"
            << "                           : mix(mid, hot, (frac - 0.5) * 2.0);\n"
            << "    }\n";
    } else if (cfg.debug_view == TracerConfig::DebugView::CullSpan) {
        // Colour by the fraction of [0,max_dist] the ray actually had to march
        // (t_far / max_dist): green = most of the ray was culled away, red =
        // little culled. Shows where the tile cull is effective.
        src << "    {\n"
            << "        float kept = clamp(t_far / " << flit(cfg.max_dist) << ", 0.0, 1.0);\n"
            << "        color = mix(vec3(0.0, 0.7, 0.1), vec3(0.9, 0.2, 0.0), kept);\n"
            << "    }\n";
    }
    src << "    // Temporal denoise accumulation. When accum_blend < 1 the\n"
        << "    // camera/scene is static and we blend this frame into the\n"
        << "    // running average already sitting in the storage image:\n"
        << "    //   out = mix(previous, new, accum_blend),  accum_blend = 1/n.\n"
        << "    // accum_blend == 1 (reset / moving / offscreen) writes the\n"
        << "    // new frame outright. Each invocation reads and writes only\n"
        << "    // its own texel, so the in-place read-modify-write is race-\n"
        << "    // free without a second image.\n"
        << "    if (pc.accum_blend < 0.999) {\n"
        << "        vec3 prev = imageLoad(out_image, px).rgb;\n"
        << "        color = mix(prev, color, pc.accum_blend);\n"
        << "    }\n"
        << "    imageStore(out_image, px, vec4(color, 1.0));\n"
        << "}\n";
    res.source = src.str();
    return res;
}

} // namespace frep::gpu
