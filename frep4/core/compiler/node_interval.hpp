// CPU interval bound for a whole node tree — the numeric twin of the GLSL
// NodeIntervalEmitter (core/gpu/glsl_node_interval.hpp). Same node kinds, same
// interval rules, but evaluated directly on floats so the Auto cull resolver can
// probe a node tree's interval cull rate without a GPU. Kept rule-for-rule in
// step with the GLSL emitter; if one changes, change both.
#pragma once
#include "core/frep/node.hpp"
#include "core/frep/custom_expr.hpp"
#include <algorithm>
#include <cmath>

namespace frep::jit {

struct Iv { float lo, hi; };

namespace ivdetail {
inline Iv mul(Iv a, Iv b) {
    float p0=a.lo*b.lo, p1=a.lo*b.hi, p2=a.hi*b.lo, p3=a.hi*b.hi;
    return { std::min(std::min(p0,p1), std::min(p2,p3)),
             std::max(std::max(p0,p1), std::max(p2,p3)) };
}
inline Iv sqr(Iv a) {
    float l=a.lo*a.lo, h=a.hi*a.hi;
    bool s0 = a.lo <= 0 && a.hi >= 0;
    return { s0 ? 0.0f : std::min(l,h), std::max(l,h) };
}
inline Iv add(Iv a, Iv b){ return {a.lo+b.lo, a.hi+b.hi}; }
inline Iv sub(Iv a, Iv b){ return {a.lo-b.hi, a.hi-b.lo}; }
inline Iv neg(Iv a){ return {-a.hi, -a.lo}; }
inline Iv sqrt_(Iv a){ return { std::sqrt(std::max(a.lo,0.0f)), std::sqrt(std::max(a.hi,0.0f)) }; }
inline Iv abs_(Iv a){
    float l=std::fabs(a.lo), h=std::fabs(a.hi);
    bool s0 = a.lo <= 0 && a.hi >= 0;
    return { s0 ? 0.0f : std::min(l,h), std::max(l,h) };
}
inline Iv div(Iv a, Iv b){
    if (b.lo <= 0 && b.hi >= 0) return {-1e30f, 1e30f};
    float q0=a.lo/b.lo,q1=a.lo/b.hi,q2=a.hi/b.lo,q3=a.hi/b.hi;
    return { std::min(std::min(q0,q1),std::min(q2,q3)), std::max(std::max(q0,q1),std::max(q2,q3)) };
}
inline Iv imin(Iv a, Iv b){ return {std::min(a.lo,b.lo), std::min(a.hi,b.hi)}; }
inline Iv imax(Iv a, Iv b){ return {std::max(a.lo,b.lo), std::max(a.hi,b.hi)}; }
inline bool crosses_pi(float lo, float hi, float ph){
    const float PI=3.14159265359f;
    return std::floor((hi-ph)/PI) >= std::ceil((lo-ph)/PI);
}
inline Iv isin(Iv a){
    float e0=std::sin(a.lo), e1=std::sin(a.hi);
    float lo=std::min(e0,e1), hi=std::max(e0,e1);
    if (crosses_pi(a.lo,a.hi, 1.57079632679f)) hi=1.0f;
    if (crosses_pi(a.lo,a.hi,-1.57079632679f)) lo=-1.0f;
    return {lo,hi};
}
inline Iv icos(Iv a){
    float e0=std::cos(a.lo), e1=std::cos(a.hi);
    float lo=std::min(e0,e1), hi=std::max(e0,e1);
    if (crosses_pi(a.lo,a.hi,0.0f))           hi=1.0f;
    if (crosses_pi(a.lo,a.hi,3.14159265359f)) lo=-1.0f;
    return {lo,hi};
}
} // namespace ivdetail

// Interval of an expression AST over a coordinate box (mirrors gen_ival / the
// GLSL expression interval emitter). Used for CustomExpr sub-nodes.
inline Iv expr_interval(const expr::Node& n, Iv X, Iv Y, Iv Z) {
    using namespace ivdetail;
    using K = expr::Node::Kind;
    switch (n.kind) {
        case K::Number: return {n.num, n.num};
        case K::Const:  { float v = n.ident=="pi"?3.14159265359f:2.71828182846f; return {v,v}; }
        case K::Var:    return n.ident=="x"?X : n.ident=="y"?Y : Z;
        case K::UnaryNeg: return neg(expr_interval(*n.children[0],X,Y,Z));
        case K::BinOp: {
            Iv a=expr_interval(*n.children[0],X,Y,Z), b=expr_interval(*n.children[1],X,Y,Z);
            switch (n.bop) {
                case expr::Op::Add: return add(a,b);
                case expr::Op::Sub: return sub(a,b);
                case expr::Op::Mul: {
                    auto& c0=*n.children[0]; auto& c1=*n.children[1];
                    bool s = (&c0==&c1) || (c0.kind==K::Var && c1.kind==K::Var && c0.ident==c1.ident);
                    return s ? sqr(a) : mul(a,b);
                }
                case expr::Op::Div: return div(a,b);
            }
            return {-1e30f,1e30f};
        }
        case K::Call: {
            const auto& f=n.ident;
            Iv a=expr_interval(*n.children[0],X,Y,Z);
            if (f=="sqrt") return sqrt_(a);
            if (f=="abs")  return abs_(a);
            if (f=="sin")  return isin(a);
            if (f=="cos")  return icos(a);
            if (f=="min")  return imin(a, expr_interval(*n.children[1],X,Y,Z));
            if (f=="max")  return imax(a, expr_interval(*n.children[1],X,Y,Z));
            return {-1e30f,1e30f};
        }
    }
    return {-1e30f,1e30f};
}

// Interval bound of a node tree over box [lo,hi] (mirrors NodeIntervalEmitter).
inline Iv node_interval(const FRepNode& n, Iv X, Iv Y, Iv Z) {
    using namespace ivdetail;
    auto pf = [&](const char* k, float d){ auto it=n.params.find(k); return it!=n.params.end()?it->second:d; };
    switch (n.kind) {
        case NodeKind::Sphere: {
            Iv s = add(add(sqr(X), sqr(Y)), sqr(Z));
            Iv r = sqrt_(s); float rad = pf("r",1.0f);
            return {r.lo-rad, r.hi-rad};
        }
        case NodeKind::Box: {
            Iv qx=sub(abs_(X),{pf("hx",0.5f),pf("hx",0.5f)});
            Iv qy=sub(abs_(Y),{pf("hy",0.5f),pf("hy",0.5f)});
            Iv qz=sub(abs_(Z),{pf("hz",0.5f),pf("hz",0.5f)});
            Iv mx={std::max(qx.lo,0.0f),std::max(qx.hi,0.0f)};
            Iv my={std::max(qy.lo,0.0f),std::max(qy.hi,0.0f)};
            Iv mz={std::max(qz.lo,0.0f),std::max(qz.hi,0.0f)};
            Iv out=sqrt_(add(add(sqr(mx),sqr(my)),sqr(mz)));
            Iv mxa=imax(qx,imax(qy,qz));
            Iv ins={std::min(mxa.lo,0.0f),std::min(mxa.hi,0.0f)};
            return add(out,ins);
        }
        case NodeKind::Plane: {
            Iv tx=mul({pf("nx",0.0f),pf("nx",0.0f)},X);
            Iv ty=mul({pf("ny",1.0f),pf("ny",1.0f)},Y);
            Iv tz=mul({pf("nz",0.0f),pf("nz",0.0f)},Z);
            float d=pf("d",0.0f);
            Iv s=add(add(tx,ty),tz); return {s.lo-d,s.hi-d};
        }
        case NodeKind::Union:        return imin(node_interval(*n.children[0],X,Y,Z), node_interval(*n.children[1],X,Y,Z));
        case NodeKind::Intersection: return imax(node_interval(*n.children[0],X,Y,Z), node_interval(*n.children[1],X,Y,Z));
        case NodeKind::Difference:   return imax(node_interval(*n.children[0],X,Y,Z), neg(node_interval(*n.children[1],X,Y,Z)));
        case NodeKind::SmoothUnion: {
            Iv h=imin(node_interval(*n.children[0],X,Y,Z), node_interval(*n.children[1],X,Y,Z));
            float k=pf("k",0.1f); return {h.lo-k, h.hi};
        }
        case NodeKind::Negate:       return neg(node_interval(*n.children[0],X,Y,Z));
        case NodeKind::Translate:    return node_interval(*n.children[0],
                                        {X.lo-pf("tx",0.0f),X.hi-pf("tx",0.0f)},
                                        {Y.lo-pf("ty",0.0f),Y.hi-pf("ty",0.0f)},
                                        {Z.lo-pf("tz",0.0f),Z.hi-pf("tz",0.0f)});
        case NodeKind::Scale: {
            float sx=pf("sx",1.0f), sy=pf("sy",1.0f), sz=pf("sz",1.0f);
            float mn=std::min(std::abs(sx),std::min(std::abs(sy),std::abs(sz)));
            Iv c=node_interval(*n.children[0], div(X,{sx,sx}), div(Y,{sy,sy}), div(Z,{sz,sz}));
            // scale distance by smallest |factor|; result endpoints stay ordered
            return {std::min(c.lo*mn,c.hi*mn), std::max(c.lo*mn,c.hi*mn)};
        }
        case NodeKind::RotateY: {
            float ang=pf("a",0.0f), cs=std::cos(ang), sn=std::sin(ang);
            Iv nx=add(mul({cs,cs},X), mul({sn,sn},Z));
            Iv nz=add(mul({-sn,-sn},X), mul({cs,cs},Z));
            return node_interval(*n.children[0], nx, Y, nz);
        }
        case NodeKind::RotateX: {
            float ang=pf("a",0.0f), cs=std::cos(ang), sn=std::sin(ang);
            Iv ny=add(mul({cs,cs},Y), mul({sn,sn},Z));
            Iv nz=add(mul({-sn,-sn},Y), mul({cs,cs},Z));
            return node_interval(*n.children[0], X, ny, nz);
        }
        case NodeKind::RotateZ: {
            float ang=pf("a",0.0f), cs=std::cos(ang), sn=std::sin(ang);
            Iv nx=add(mul({cs,cs},X), mul({sn,sn},Y));
            Iv ny=add(mul({-sn,-sn},X), mul({cs,cs},Y));
            return node_interval(*n.children[0], nx, ny, Z);
        }
        case NodeKind::TwistY: {
            float k=pf("k",1.0f);
            Iv ang=mul({k,k},Y), ca=icos(ang), sa=isin(ang);
            Iv nx=add(mul(ca,X), mul(sa,Z));
            Iv nz=add(mul({-1,1}, mul(sa,X)), mul(ca,Z));
            return node_interval(*n.children[0], nx, Y, nz);
        }
        case NodeKind::BendXY: {
            // ks=k (clamped away from 0); th=ks*x; r=1/ks + y;
            // x' = r*sin(th); y' = r*cos(th) - 1/ks; recurse on (x',y',z);
            // result = child / max(1, |ks*r|)  (Lipschitz correction).
            float k=pf("k",1.0f); float ks = std::fabs(k)<1e-6f ? 1e-6f : k;
            float invk = 1.0f/ks;
            Iv th=mul({ks,ks},X), r=add({invk,invk},Y);
            Iv nx=mul(r, isin(th));
            Iv ny=sub(mul(r, icos(th)), {invk,invk});
            Iv child=node_interval(*n.children[0], nx, ny, Z);
            Iv ksr=abs_(mul({ks,ks}, r));
            Iv lip={std::max(1.0f,ksr.lo), std::max(1.0f,ksr.hi)};
            return div(child, lip);
        }
        case NodeKind::TaperY: {
            // u=clamp((y+h/2)/h,0,1); s=max(1+u*(t-1),1e-3); is=1/s;
            // x'=x*is; z'=z*is; recurse on (x',y,z'); result=child/max(1,is).
            float t=pf("t",1.0f), h=pf("h",1.0f);
            Iv u = mul(add(Y,{0.5f*h,0.5f*h}), {1.0f/h,1.0f/h});
            u = {std::clamp(u.lo,0.0f,1.0f), std::clamp(u.hi,0.0f,1.0f)};
            Iv s = add({1.0f,1.0f}, mul(u,{t-1.0f,t-1.0f}));
            s = {std::max(s.lo,1e-3f), std::max(s.hi,1e-3f)};
            Iv is = div({1.0f,1.0f}, s);
            Iv child = node_interval(*n.children[0], mul(X,is), Y, mul(Z,is));
            Iv lip = {std::max(1.0f,is.lo), std::max(1.0f,is.hi)};
            return div(child, lip);
        }
        case NodeKind::Scene: return n.children.empty() ? Iv{1e30f,1e30f} : node_interval(*n.children[0],X,Y,Z);
        case NodeKind::Instance: return (n.children.empty() || !n.children[0])
                                        ? Iv{1e30f,1e30f} : node_interval(*n.children[0],X,Y,Z);
        default: break;
    }
    if (const void* a = n.custom_expr_ast())
        return expr_interval(**static_cast<const expr::NodePtr*>(a), X, Y, Z);
    return {-1e30f, 1e30f};
}

} // namespace frep::jit
