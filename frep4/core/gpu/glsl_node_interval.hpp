// Interval GLSL codegen for a whole node tree, so the tile cull can use interval
// arithmetic on any scene — not only a single CustomExpr. It mirrors
// GlslEmitter::emit_node: same node kinds, same formulas, but every value is a
// vec2(lo,hi) bound over an input coordinate box, and coordinates flow as six
// scalars (xlo,xhi, ylo,yhi, zlo,zhi) that transforms rewrite as intervals.
//
// The emitted body is a sequence of `vec2 tN = ...;` lines ending in the scene
// bound; it is dropped into `vec2 scene_sdf_ival(vec3 lo, vec3 hi)`. The interval
// helpers (ival_mul/sqr/sqrt/abs/sin/cos/...) come from glsl_interval_prelude().
//
// Metric primitives and rigid transforms are exact; non-metric deformations
// (Twist/Bend/Taper) use interval trig on the rotated coordinates, which is
// sound but loose. CustomExpr nodes defer to GlslIntervalEmitter on their AST.
#pragma once
#include "core/frep/node.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/gpu/glsl_interval.hpp"
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace frep::gpu {

class NodeIntervalEmitter {
public:
    // Coordinate interval carried down the tree: each axis is [lo,hi] as a GLSL
    // vec2 expression (usually a temporary name, sometimes an inline expr).
    struct Box { std::string x, y, z; };   // each is a vec2 expression

    // Emit the interval SDF body for `root`; `result` gets the final vec2 temp.
    // `param` resolves a node parameter to a GLSL constant/expression (same
    // callback the scalar emitter uses, so runtime-param scenes stay in sync).
    std::string emit(const FRepNode& root, std::string& result) {
        out_.str({}); n_ = 0;
        Box b{"vec2(lo.x, hi.x)", "vec2(lo.y, hi.y)", "vec2(lo.z, hi.z)"};
        result = go(root, b);
        return out_.str();
    }

    // Parameter resolver: default reads the node's static param value. A caller
    // with runtime bindings can override before emit().
    std::function<std::string(const FRepNode&, const std::string&, float)> param =
        [](const FRepNode& n, const std::string& name, float def) {
            auto it = n.params.find(name);
            std::ostringstream s; s.precision(9);
            s << std::fixed << (it != n.params.end() ? it->second : def);
            return s.str();
        };

private:
    std::ostringstream out_;
    int n_ = 0;

    std::string tmp(const std::string& rhs) {
        std::string v = "iv" + std::to_string(n_++);
        out_ << "    vec2 " << v << " = " << rhs << ";\n";
        return v;
    }
    std::string p(const FRepNode& n, const std::string& k, float d) { return param(n, k, d); }

    std::string go(const FRepNode& n, const Box& b) {
        using K = NodeKind;
        switch (n.kind) {
            case K::Sphere:       return prim_sphere(n, b);
            case K::Box:          return prim_box(n, b);
            case K::Plane:        return prim_plane(n, b);
            case K::Union:        return binary(n, b, "min");
            case K::Intersection: return binary(n, b, "max");
            case K::Difference:   return difference(n, b);
            case K::SmoothUnion:  return smooth_union(n, b);
            case K::Negate:       return negate(n, b);
            case K::Translate:    return translate(n, b);
            case K::Scale:        return scale(n, b);
            case K::RotateY:      return rotate_y(n, b);
            case K::RotateX:      return rotate_x(n, b);
            case K::RotateZ:      return rotate_z(n, b);
            case K::TwistY:       return twist_y(n, b);
            case K::BendXY:       return bend_xy(n, b);
            case K::TaperY:       return taper_y(n, b);
            case K::Instance:     return (n.children.empty() || !n.children[0])
                                         ? tmp("vec2(1e30)") : go(*n.children[0], b);
            case K::Scene:        return n.children.empty() ? tmp("vec2(1e30)")
                                                            : go(*n.children[0], b);
            default:              break;
        }
        // CustomExpr / Plugin: if it exposes an expr AST, emit interval from it
        // with the current coordinate box substituted for x/y/z.
        if (const void* a = n.custom_expr_ast()) {
            const auto& ast = *static_cast<const expr::NodePtr*>(a);
            GlslIntervalEmitter ie;
            // Reuse the expression interval emitter, but feed it our coordinate
            // box by wrapping: it reads lo/hi.{x,y,z}; we can't rebind those
            // names locally, so inline a helper scope.
            std::string res;
            std::string body = ie.emit(*ast, res);
            // Substitute the fixed lo/hi coordinate reads with our box exprs.
            body = subst_coords(body, b);
            res  = subst_coords(res, b);
            out_ << body;
            return res.empty() ? tmp("vec2(-1e30,1e30)") : res;
        }
        return tmp("vec2(-1e30, 1e30)");   // unknown node: no information
    }

    // Replace the expression emitter's coordinate reads with our box exprs.
    static std::string subst_coords(std::string s, const Box& b) {
        auto rep = [&](const std::string& from, const std::string& to){
            for (size_t i; (i = s.find(from)) != std::string::npos; )
                s.replace(i, from.size(), to);
        };
        rep("vec2(lo.x, hi.x)", b.x);
        rep("vec2(lo.y, hi.y)", b.y);
        rep("vec2(lo.z, hi.z)", b.z);
        return s;
    }

    // ── primitives ──────────────────────────────────────────────────────────
    std::string prim_sphere(const FRepNode& n, const Box& b) {
        auto x2 = tmp("ival_sqr(" + b.x + ")");
        auto y2 = tmp("ival_sqr(" + b.y + ")");
        auto z2 = tmp("ival_sqr(" + b.z + ")");
        auto s  = tmp(x2 + " + " + y2 + " + " + z2);
        auto r  = tmp("ival_sqrt(" + s + ")");
        return tmp("vec2(" + r + ".x - " + p(n,"r",1.0f) + ", " + r + ".y - " + p(n,"r",1.0f) + ")");
    }
    std::string prim_box(const FRepNode& n, const Box& b) {
        // q = abs(p) - h; d = length(max(q,0)) + min(max(q.x,q.y,q.z),0)
        // Interval form: qi = ival_abs(coord) - h; outside = sqrt(sum sqr(max(qi,0)));
        // inside = min(max over axes of qi.hi, 0). Bound d in [inside_lo, out_hi].
        std::string hx=p(n,"hx",0.5f), hy=p(n,"hy",0.5f), hz=p(n,"hz",0.5f);
        auto qx = tmp("ival_abs(" + b.x + ") - vec2(" + hx + ")");
        auto qy = tmp("ival_abs(" + b.y + ") - vec2(" + hy + ")");
        auto qz = tmp("ival_abs(" + b.z + ") - vec2(" + hz + ")");
        // max(q,0) per axis, then squared and summed for the exterior length
        auto mx = tmp("max(" + qx + ", vec2(0.0))");
        auto my = tmp("max(" + qy + ", vec2(0.0))");
        auto mz = tmp("max(" + qz + ", vec2(0.0))");
        auto sum= tmp("ival_sqr(" + mx + ") + ival_sqr(" + my + ") + ival_sqr(" + mz + ")");
        auto out= tmp("ival_sqrt(" + sum + ")");
        // interior term: min(max(qx,qy,qz), 0), computed on the interval
        auto mxa= tmp("max(" + qx + ", max(" + qy + ", " + qz + "))");
        auto ins= tmp("min(" + mxa + ", vec2(0.0))");
        return tmp(out + " + " + ins);
    }
    std::string prim_plane(const FRepNode& n, const Box& b) {
        // signed distance to plane y = 0 by default: just the (transformed) y.
        // frep4's PlaneNode stores a normal; approximate with n·p form.
        std::string nx=p(n,"nx",0.0f), ny=p(n,"ny",1.0f), nz=p(n,"nz",0.0f), d=p(n,"d",0.0f);
        auto tx = tmp("ival_mul(vec2(" + nx + "), " + b.x + ")");
        auto ty = tmp("ival_mul(vec2(" + ny + "), " + b.y + ")");
        auto tz = tmp("ival_mul(vec2(" + nz + "), " + b.z + ")");
        return tmp(tx + " + " + ty + " + " + tz + " - vec2(" + d + ")");
    }

    // ── operations ──────────────────────────────────────────────────────────
    std::string binary(const FRepNode& n, const Box& b, const char* op) {
        auto a = go(*n.children[0], b);
        auto c = go(*n.children[1], b);
        return tmp(std::string(op) + "(" + a + ", " + c + ")");
    }
    std::string difference(const FRepNode& n, const Box& b) {
        auto a = go(*n.children[0], b);
        auto c = go(*n.children[1], b);
        auto nc = tmp("vec2(-" + c + ".y, -" + c + ".x)");   // negate interval
        return tmp("max(" + a + ", " + nc + ")");
    }
    std::string smooth_union(const FRepNode& n, const Box& b) {
        // Interval bound of the C2 smooth min: it lies between the hard min and
        // that min minus k/6 (the max blend depth). Sound and simple; tightness
        // is secondary for a cull.
        auto a = go(*n.children[0], b);
        auto c = go(*n.children[1], b);
        std::string k = p(n, "k", 0.1f);
        auto hard = tmp("min(" + a + ", " + c + ")");
        return tmp("vec2(" + hard + ".x - " + k + ", " + hard + ".y)");
    }
    std::string negate(const FRepNode& n, const Box& b) {
        auto a = go(*n.children[0], b);
        return tmp("vec2(-" + a + ".y, -" + a + ".x)");
    }

    // ── transforms (rewrite the coordinate box) ─────────────────────────────
    std::string translate(const FRepNode& n, const Box& b) {
        Box t{ b.x + " - vec2(" + p(n,"tx",0.0f) + ")",
               b.y + " - vec2(" + p(n,"ty",0.0f) + ")",
               b.z + " - vec2(" + p(n,"tz",0.0f) + ")" };
        return go(*n.children[0], t);
    }
    std::string scale(const FRepNode& n, const Box& b) {
        std::string sx=p(n,"sx",1.0f), sy=p(n,"sy",1.0f), sz=p(n,"sz",1.0f);
        Box t{ "ival_div(" + b.x + ", vec2(" + sx + "))",
               "ival_div(" + b.y + ", vec2(" + sy + "))",
               "ival_div(" + b.z + ", vec2(" + sz + "))" };
        auto child = go(*n.children[0], t);
        // scale distance by the smallest |factor|; ival_mul keeps endpoints ordered
        std::string mn = "min(abs(" + sx + "), min(abs(" + sy + "), abs(" + sz + ")))";
        return tmp("ival_mul(" + child + ", vec2(" + mn + "))");
    }
    std::string rotate_y(const FRepNode& n, const Box& b) {
        std::string ang = p(n, "a", 0.0f);
        auto cs = "cos(" + ang + ")"; auto sn = "sin(" + ang + ")";
        Box t{ "ival_mul(vec2(" + cs + "), " + b.x + ") + ival_mul(vec2(" + sn + "), " + b.z + ")",
               b.y,
               "ival_mul(vec2(" + std::string("-") + sn + "), " + b.x + ") + ival_mul(vec2(" + cs + "), " + b.z + ")" };
        return go(*n.children[0], t);
    }
    std::string rotate_x(const FRepNode& n, const Box& b) {
        std::string ang = p(n, "a", 0.0f);
        auto cs = "cos(" + ang + ")"; auto sn = "sin(" + ang + ")";
        Box t{ b.x,
               "ival_mul(vec2(" + cs + "), " + b.y + ") + ival_mul(vec2(" + sn + "), " + b.z + ")",
               "ival_mul(vec2(" + std::string("-") + sn + "), " + b.y + ") + ival_mul(vec2(" + cs + "), " + b.z + ")" };
        return go(*n.children[0], t);
    }
    std::string rotate_z(const FRepNode& n, const Box& b) {
        std::string ang = p(n, "a", 0.0f);
        auto cs = "cos(" + ang + ")"; auto sn = "sin(" + ang + ")";
        Box t{ "ival_mul(vec2(" + cs + "), " + b.x + ") + ival_mul(vec2(" + sn + "), " + b.y + ")",
               "ival_mul(vec2(" + std::string("-") + sn + "), " + b.x + ") + ival_mul(vec2(" + cs + "), " + b.y + ")",
               b.z };
        return go(*n.children[0], t);
    }
    std::string twist_y(const FRepNode& n, const Box& b) {
        // Rotate (x,z) by angle k*y. Over a box, k*y is an interval, so the
        // rotation coefficients are cos/sin of an interval -> loose but sound.
        std::string k = p(n, "k", 1.0f);
        auto ang = tmp("ival_mul(vec2(" + k + "), " + b.y + ")");
        auto ca  = tmp("ival_cos(" + ang + ")");
        auto sa  = tmp("ival_sin(" + ang + ")");
        Box t{ "ival_mul(" + ca + ", " + b.x + ") + ival_mul(" + sa + ", " + b.z + ")",
               b.y,
               "ival_mul(vec2(-1.0,1.0), ival_mul(" + sa + ", " + b.x + ")) + ival_mul(" + ca + ", " + b.z + ")" };
        // (the x/z bounds widen because cos/sin of an interval spans a range)
        return go(*n.children[0], t);
    }
    std::string bend_xy(const FRepNode& n, const Box& b) {
        // ks=k (clamped); th=ks*x; r=1/ks + y; x'=r*sin(th); y'=r*cos(th)-1/ks;
        // recurse on (x',y',z); result = child / max(1, |ks*r|). cos/sin of an
        // interval -> sound but loose, like the twist.
        float kc = 1.0f; { auto it=n.params.find("k"); if (it!=n.params.end()) kc=it->second; }
        float ks = std::fabs(kc) < 1e-6f ? 1e-6f : kc;
        std::ostringstream ss; ss.precision(9); ss << std::fixed << ks;
        std::string ksS = ss.str();
        std::ostringstream is; is.precision(9); is << std::fixed << (1.0f/ks);
        std::string invk = is.str();
        auto th = tmp("ival_mul(vec2(" + ksS + "), " + b.x + ")");
        auto r  = tmp("vec2(" + invk + ") + " + b.y);
        auto sn = tmp("ival_sin(" + th + ")");
        auto cs = tmp("ival_cos(" + th + ")");
        Box t{ "ival_mul(" + r + ", " + sn + ")",
               "ival_mul(" + r + ", " + cs + ") - vec2(" + invk + ")",
               b.z };
        auto child = go(*n.children[0], t);
        auto ksr = tmp("ival_abs(ival_mul(vec2(" + ksS + "), " + r + "))");
        auto lip = tmp("max(vec2(1.0), " + ksr + ")");
        return tmp("ival_div(" + child + ", " + lip + ")");
    }
    std::string taper_y(const FRepNode& n, const Box& b) {
        // u=clamp((y+h/2)/h,0,1); s=max(1+u*(t-1),1e-3); is=1/s;
        // x'=x*is; z'=z*is; recurse on (x',y,z'); result=child/max(1,is).
        std::string t = p(n, "t", 1.0f), h = p(n, "h", 1.0f);
        auto u0 = tmp("ival_mul(" + b.y + " + vec2(0.5 * " + h + "), vec2(1.0 / (" + h + ")))");
        auto u  = tmp("clamp(" + u0 + ", vec2(0.0), vec2(1.0))");
        auto s0 = tmp("vec2(1.0) + ival_mul(" + u + ", vec2((" + t + ") - 1.0))");
        auto s  = tmp("max(" + s0 + ", vec2(1.0e-3))");
        auto isv= tmp("ival_div(vec2(1.0), " + s + ")");
        Box tb{ "ival_mul(" + b.x + ", " + isv + ")", b.y, "ival_mul(" + b.z + ", " + isv + ")" };
        auto child = go(*n.children[0], tb);
        auto lip = tmp("max(vec2(1.0), " + isv + ")");
        return tmp("ival_div(" + child + ", " + lip + ")");
    }
};

} // namespace frep::gpu
