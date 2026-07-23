// GLSL interval arithmetic for GPU tile culling.
//
// Emits `vec2 sdf_ival(vec3 lo, vec3 hi)` — component .x = lower bound,
// .y = upper bound of the scene SDF over the axis-aligned box [lo,hi]. The
// algebra mirrors CustomExprCompiler::gen_ival (core/frep/custom_expr.cpp); the
// two must stay in step, so the rules are written out in the same order here.
//
// A tile-cull compute shader uses it to reject screen tiles whose whole frustum
// slab is strictly inside or strictly outside the surface, the way MPR prunes
// its tile tree. The render kernel then skips culled tiles entirely.
//
// This header only *generates source*; compiling and dispatching it is the
// caller's job (see glsl_compile.hpp).
#pragma once
#include "core/frep/expr_ast.hpp"
#include <sstream>
#include <string>
#include <unordered_map>

namespace frep::gpu {

// ── Interval expression emitter ─────────────────────────────────────────────
// Every subexpression becomes `vec2 tN = ...;` so DAG sharing survives (each
// node is emitted once, keyed by address).
class GlslIntervalEmitter {
public:
    // Returns the body of sdf_ival: a sequence of `vec2 tN = ...;` lines whose
    // last temporary is the result. `result` receives that temporary's name.
    std::string emit(const expr::Node& root, std::string& result) {
        out_.str({}); memo_.clear(); n_ = 0;
        result = go(root);
        return out_.str();
    }

    // Complete shader source: interval helpers + sdf_ival + tile-cull main.
    // tile  = pixels per tile edge.
    // slabs = depth subdivisions of [near,far] tested per tile.
    // The shader writes, per tile, the active depth range as vec2(t0,t1); a
    // fully culled tile gets vec2(1e30,-1e30) and the render kernel skips it.
    // One slab per tile would bound the whole view ray and never cull, so the
    // depth subdivision is what makes the pass useful.
    static std::string tile_cull_shader(const expr::Node& root, int tile = 8,
                                        int slabs = 32);

private:
    std::ostringstream out_;
    std::unordered_map<const expr::Node*, std::string> memo_;
    int n_ = 0;

    std::string tmp(const std::string& rhs) {
        std::string v = "t" + std::to_string(n_++);
        out_ << "    vec2 " << v << " = " << rhs << ";\n";
        return v;
    }

    std::string go(const expr::Node& n) {
        if (auto it = memo_.find(&n); it != memo_.end()) return it->second;
        using K = expr::Node::Kind;
        std::string v;
        switch (n.kind) {
            case K::Number: v = tmp("vec2(" + fmt(n.num) + ")"); break;
            case K::Const:
                v = tmp("vec2(" + fmt(n.ident == "pi" ? 3.14159265359f
                                                      : 2.71828182846f) + ")");
                break;
            case K::Var:
                v = tmp(n.ident == "x" ? "vec2(lo.x, hi.x)"
                      : n.ident == "y" ? "vec2(lo.y, hi.y)"
                                       : "vec2(lo.z, hi.z)");
                break;
            case K::UnaryNeg: {
                auto a = go(*n.children[0]);
                v = tmp("vec2(-" + a + ".y, -" + a + ".x)");
                break;
            }
            case K::BinOp:   v = binop(n); break;
            case K::Call:    v = call(n);  break;
        }
        memo_[&n] = v;
        return v;
    }

    std::string binop(const expr::Node& n) {
        auto a = go(*n.children[0]);
        auto b = go(*n.children[1]);
        switch (n.bop) {
            case expr::Op::Add: return tmp(a + " + " + b);
            case expr::Op::Sub: return tmp("vec2(" + a + ".x - " + b + ".y, "
                                                   + a + ".y - " + b + ".x)");
            case expr::Op::Mul: {
                // Same-variable product is a square: its minimum is 0 when the
                // interval straddles 0, not the endpoint product.
                const auto& c0 = *n.children[0];
                const auto& c1 = *n.children[1];
                bool sq = (&c0 == &c1) ||
                          (c0.kind == expr::Node::Kind::Var &&
                           c1.kind == expr::Node::Kind::Var && c0.ident == c1.ident);
                if (sq)
                    return tmp("ival_sqr(" + a + ")");
                return tmp("ival_mul(" + a + ", " + b + ")");
            }
            case expr::Op::Div: return tmp("ival_div(" + a + ", " + b + ")");
        }
        return "vec2(0.0)";
    }

    std::string call(const expr::Node& n) {
        std::vector<std::string> a;
        for (const auto& c : n.children) a.push_back(go(*c));
        const auto& f = n.ident;
        if (f == "sqrt")  return tmp("ival_sqrt(" + a[0] + ")");
        if (f == "abs")   return tmp("ival_abs(" + a[0] + ")");
        if (f == "min")   return tmp("min(" + a[0] + ", " + a[1] + ")");
        if (f == "max")   return tmp("max(" + a[0] + ", " + a[1] + ")");
        if (f == "pow")   return tmp("ival_pow(" + a[0] + ", " + a[1] + ")");
        // nth_root(a,b) == a^(1/b); reciprocal of a positive interval flips ends.
        if (f == "nth_root")
            return tmp("ival_pow(" + a[0] + ", vec2(1.0 / " + a[1] + ".y, 1.0 / "
                                   + a[1] + ".x))");
        if (f == "sin")   return tmp("ival_sin(" + a[0] + ")");
        if (f == "cos")   return tmp("ival_cos(" + a[0] + ")");
        if (f == "tan")   return tmp("ival_tan(" + a[0] + ")");
        if (f == "asin")  return tmp("vec2(asin(clamp(" + a[0] + ".x,-1.0,1.0)), asin(clamp(" + a[0] + ".y,-1.0,1.0)))");
        if (f == "acos")  return tmp("vec2(acos(clamp(" + a[0] + ".y,-1.0,1.0)), acos(clamp(" + a[0] + ".x,-1.0,1.0)))");
        if (f == "atan")  return tmp("vec2(atan(" + a[0] + ".x), atan(" + a[0] + ".y))");
        if (f == "atan2") return tmp("ival_atan2(" + a[0] + ", " + a[1] + ")");
        if (f == "mod")   return tmp("ival_mod(" + a[0] + ", " + a[1] + ")");
        if (f == "exp")   return tmp("vec2(exp(" + a[0] + ".x), exp(" + a[0] + ".y))");
        if (f == "log")   return tmp("vec2(log(max(" + a[0] + ".x,1e-8)), log(max(" + a[0] + ".y,1e-8)))");
        if (f == "floor") return tmp("floor(" + a[0] + ")");
        if (f == "ceil")  return tmp("ceil(" + a[0] + ")");
        return tmp("vec2(-1e30, 1e30)");   // unknown -> no information
    }

    static std::string fmt(float v) {
        std::ostringstream s; s.precision(9); s << std::fixed << v; return s.str();
    }
};

// Interval helper functions, shared by every generated shader.
inline const char* glsl_interval_prelude() {
    return R"(
// ---- interval helpers: .x = lo, .y = hi ------------------------------------
vec2 ival_mul(vec2 a, vec2 b) {
    vec4 p = vec4(a.x*b.x, a.x*b.y, a.y*b.x, a.y*b.y);
    return vec2(min(min(p.x,p.y), min(p.z,p.w)), max(max(p.x,p.y), max(p.z,p.w)));
}
vec2 ival_sqr(vec2 a) {
    float l = a.x*a.x, h = a.y*a.y;
    bool spans0 = (a.x <= 0.0) && (a.y >= 0.0);
    return vec2(spans0 ? 0.0 : min(l,h), max(l,h));
}
vec2 ival_div(vec2 a, vec2 b) {
    if (b.x <= 0.0 && b.y >= 0.0) return vec2(-1e30, 1e30);   // divisor straddles 0
    vec4 q = vec4(a.x/b.x, a.x/b.y, a.y/b.x, a.y/b.y);
    return vec2(min(min(q.x,q.y), min(q.z,q.w)), max(max(q.x,q.y), max(q.z,q.w)));
}
vec2 ival_sqrt(vec2 a) { return vec2(sqrt(max(a.x,0.0)), sqrt(max(a.y,0.0))); }
vec2 ival_abs(vec2 a) {
    float l = abs(a.x), h = abs(a.y);
    bool spans0 = (a.x <= 0.0) && (a.y >= 0.0);
    return vec2(spans0 ? 0.0 : min(l,h), max(l,h));
}
vec2 ival_pow(vec2 a, vec2 e) {   // constant exponent, non-negative base
    return vec2(pow(max(a.x,0.0), e.x), pow(max(a.y,0.0), e.y));
}
vec2 ival_mod(vec2 a, vec2 m) {   // conservative: any value in [0,m)
    if (m.x != m.y || m.x <= 0.0) return vec2(-1e30, 1e30);
    if (a.y - a.x >= m.x) return vec2(0.0, m.x);
    float la = a.x - floor(a.x/m.x)*m.x;
    float lb = a.y - floor(a.y/m.x)*m.x;
    return (la <= lb) ? vec2(la, lb) : vec2(0.0, m.x);
}
// exists k with lo <= phase + k*PI <= hi
bool crosses_pi(float lo, float hi, float phase) {
    const float PI = 3.14159265359;
    return floor((hi - phase)/PI) >= ceil((lo - phase)/PI);
}
vec2 ival_sin(vec2 a) {
    float e0 = sin(a.x), e1 = sin(a.y);
    float lo = min(e0,e1), hi = max(e0,e1);
    if (crosses_pi(a.x, a.y,  1.57079632679)) hi =  1.0;
    if (crosses_pi(a.x, a.y, -1.57079632679)) lo = -1.0;
    return vec2(lo, hi);
}
vec2 ival_cos(vec2 a) {
    float e0 = cos(a.x), e1 = cos(a.y);
    float lo = min(e0,e1), hi = max(e0,e1);
    if (crosses_pi(a.x, a.y, 0.0))           hi =  1.0;
    if (crosses_pi(a.x, a.y, 3.14159265359)) lo = -1.0;
    return vec2(lo, hi);
}
vec2 ival_tan(vec2 a) {
    if (crosses_pi(a.x, a.y, 1.57079632679)) return vec2(-1e30, 1e30);  // pole
    float t0 = tan(a.x), t1 = tan(a.y);
    return vec2(min(t0,t1), max(t0,t1));
}
vec2 ival_atan2(vec2 y, vec2 x) {
    const float PI = 3.14159265359;
    if (x.x <= 0.0 && y.x <= 0.0 && y.y >= 0.0) return vec2(-PI, PI);  // origin/cut
    vec4 c = vec4(atan(y.x, x.x), atan(y.x, x.y), atan(y.y, x.x), atan(y.y, x.y));
    return vec2(min(min(c.x,c.y), min(c.z,c.w)), max(max(c.x,c.y), max(c.z,c.w)));
}
)";
}

inline std::string GlslIntervalEmitter::tile_cull_shader(const expr::Node& root,
                                                         int tile, int slabs) {
    GlslIntervalEmitter e;
    std::string res;
    std::string body = e.emit(root, res);

    std::ostringstream s;
    s << "#version 450\n"
      << "layout(local_size_x = 8, local_size_y = 8) in;\n"
      << glsl_interval_prelude()
      << "\n"
         "layout(std430, binding = 0) buffer Ranges { vec2 ranges[]; };\n"
         "layout(push_constant) uniform Push {\n"
         "    vec3 origin;\n"
         "    vec3 forward;\n"
         "    vec3 right;\n"
         "    vec3 up;\n"
         "    vec2 fov_scale;   // (aspect*tan(fov/2), tan(fov/2))\n"
         "    vec2 near_far;\n"
         "    ivec2 tiles;\n"
         "    ivec2 image;\n"
         "} pc;\n"
         "\n"
         "vec2 sdf_ival(vec3 lo, vec3 hi) {\n"
      << body
      << "    return " << res << ";\n"
         "}\n"
         "\n"
         "// AABB of the frustum slab between depths t0 and t1 for one tile.\n"
      << "const int TILE  = " << tile  << ";\n"
      << "const int SLABS = " << slabs << ";\n"
      << "void slab_box(ivec2 t, float t0, float t1, out vec3 lo, out vec3 hi) {\n"
         "    vec2 p0 = vec2(t * TILE) / vec2(pc.image) * 2.0 - 1.0;\n"
         "    vec2 p1 = vec2((t + ivec2(1)) * TILE) / vec2(pc.image) * 2.0 - 1.0;\n"
         "    lo = vec3( 1e30);\n"
         "    hi = vec3(-1e30);\n"
         "    for (int c = 0; c < 4; ++c) {\n"
         "        vec2 p = vec2((c & 1) == 0 ? p0.x : p1.x, (c & 2) == 0 ? p0.y : p1.y);\n"
         "        vec3 d = normalize(pc.forward + p.x * pc.fov_scale.x * pc.right\n"
         "                                      + p.y * pc.fov_scale.y * pc.up);\n"
         "        lo = min(lo, min(pc.origin + d*t0, pc.origin + d*t1));\n"
         "        hi = max(hi, max(pc.origin + d*t0, pc.origin + d*t1));\n"
         "    }\n"
         "}\n"
         "\n"
         "void main() {\n"
         "    ivec2 t = ivec2(gl_GlobalInvocationID.xy);\n"
         "    if (t.x >= pc.tiles.x || t.y >= pc.tiles.y) return;\n"
         "    float step = (pc.near_far.y - pc.near_far.x) / float(SLABS);\n"
         "    float t0 = 1e30, t1 = -1e30;\n"
         "    for (int i = 0; i < SLABS; ++i) {\n"
         "        float a = pc.near_far.x + step * float(i);\n"
         "        float b = a + step;\n"
         "        vec3 lo, hi;\n"
         "        slab_box(t, a, b, lo, hi);\n"
         "        vec2 f = sdf_ival(lo, hi);\n"
         "        if (f.x > 0.0 || f.y < 0.0) continue;   // slab has no surface\n"
         "        t0 = min(t0, a);\n"
         "        t1 = max(t1, b);\n"
         "    }\n"
         "    ranges[t.y * pc.tiles.x + t.x] = vec2(t0, t1);\n"
         "}\n";
    return s.str();
}

} // namespace frep::gpu
