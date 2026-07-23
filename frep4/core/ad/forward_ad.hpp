#pragma once
// core/ad/forward_ad.hpp
//
// Forward-mode automatic differentiation (dual numbers).
// Used for:
//   1. Computing grad(f)(x,y,z) = the normal vector of the SDF surface.
//   2. Verification in tests (against finite-diff).
// The LLVM IR version of the normal is in codegen.cpp (finite-diff for simplicity).

#include <array>
#include <cmath>

namespace frep::ad {

template<typename T>
struct Dual {
    T val{};
    T dot{};

    static Dual cst(T v)  { return {v, T{0}}; }
    static Dual var(T v)  { return {v, T{1}}; }

    Dual operator+(Dual o) const noexcept { return {val+o.val, dot+o.dot}; }
    Dual operator-(Dual o) const noexcept { return {val-o.val, dot-o.dot}; }
    Dual operator*(Dual o) const noexcept { return {val*o.val, val*o.dot+dot*o.val}; }
    Dual operator/(Dual o) const noexcept {
        return {val/o.val, (dot*o.val - val*o.dot)/(o.val*o.val)};
    }
    Dual operator-()        const noexcept { return {-val, -dot}; }
    Dual operator+(T s)     const noexcept { return {val+s, dot}; }
    Dual operator-(T s)     const noexcept { return {val-s, dot}; }
    Dual operator*(T s)     const noexcept { return {val*s, dot*s}; }
    bool operator<(Dual o)  const noexcept { return val < o.val; }
    bool operator>(Dual o)  const noexcept { return val > o.val; }
    bool operator<=(Dual o) const noexcept { return val <= o.val; }
    bool operator>=(Dual o) const noexcept { return val >= o.val; }
};

template<typename T> Dual<T> operator+(T s, Dual<T> d) { return {s+d.val, d.dot}; }
template<typename T> Dual<T> operator-(T s, Dual<T> d) { return {s-d.val, -d.dot}; }
template<typename T> Dual<T> operator*(T s, Dual<T> d) { return {s*d.val, s*d.dot}; }

template<typename T>
Dual<T> sqrt(Dual<T> d) {
    T sv = std::sqrt(d.val);
    return {sv, d.dot / (T{2} * sv)};
}

template<typename T>
Dual<T> abs(Dual<T> d) {
    return {std::abs(d.val), d.dot * (d.val >= T{0} ? T{1} : T{-1})};
}

template<typename T>
Dual<T> min(Dual<T> a, Dual<T> b) { return a.val < b.val ? a : b; }

template<typename T>
Dual<T> max(Dual<T> a, Dual<T> b) { return a.val > b.val ? a : b; }

template<typename T>
Dual<T> clamp(Dual<T> x, T lo, T hi) { return min(max(x, Dual<T>::cst(lo)), Dual<T>::cst(hi)); }

// gradient — computes the SDF normal vector at point (x,y,z).
// The SDF must accept Dual<float> arguments.
template<typename SDF>
std::array<float, 3> gradient(SDF&& f, float x, float y, float z) {
    using D = Dual<float>;
    float nx = f(D::var(x), D::cst(y), D::cst(z)).dot;
    float ny = f(D::cst(x), D::var(y), D::cst(z)).dot;
    float nz = f(D::cst(x), D::cst(y), D::var(z)).dot;
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len < 1e-6f) return {0.0f, 1.0f, 0.0f};
    return {nx/len, ny/len, nz/len};
}

// Finite-difference normal (for comparison/fallback).
template<typename SDF>
std::array<float, 3> normal_fd(SDF&& f, float x, float y, float z, float eps = 0.001f) {
    float nx = f(x+eps,y,z) - f(x-eps,y,z);
    float ny = f(x,y+eps,z) - f(x,y-eps,z);
    float nz = f(x,y,z+eps) - f(x,y,z-eps);
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len < 1e-6f) return {0.0f, 1.0f, 0.0f};
    return {nx/len, ny/len, nz/len};
}

} // namespace frep::ad
