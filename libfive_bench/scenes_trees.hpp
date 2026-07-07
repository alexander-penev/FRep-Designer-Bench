// Canonical scene set as libfive trees (see scenes/MATH.md). f<=0 inside.
#pragma once
#include <libfive/tree/tree.hpp>
#include <string>
#include <vector>
using libfive::Tree;
inline Tree len(Tree x, Tree y, Tree z) { return sqrt(x*x + y*y + z*z); }
inline std::vector<std::pair<std::string, Tree>> canonical_scenes() {
    Tree x = Tree::X(), y = Tree::Y(), z = Tree::Z();
    std::vector<std::pair<std::string, Tree>> v;
    v.emplace_back("s1_sphere", len(x, y, z) - 1.0);
    { // cube minus sphere
        Tree box = max(max(abs(x), abs(y)), abs(z)) - 0.9;
        v.emplace_back("s2_csg", max(box, -(len(x, y, z) - 1.1)));
    }
    { // C2 cubic smooth-union of two spheres
        Tree a = len(x - 0.45, y, z) - 0.7, b = len(x + 0.45, y, z) - 0.7;
        Tree k(0.25);
        Tree h = max(k - abs(a - b), Tree(0.0)) / k;
        v.emplace_back("s3_blend", min(a, b) - h * h * h * k / 6.0);
    }
    { // gyroid clipped by sphere
        Tree g = sin(3*x)*cos(3*y) + sin(3*y)*cos(3*z) + sin(3*z)*cos(3*x) - 0.2;
        v.emplace_back("s4_gyroid", max(g, len(x, y, z) - 1.4));
    }
    { // twisted bar
        Tree w(1.2), c = cos(w * z), s = sin(w * z);
        Tree xt = x * c + y * s, yt = -x * s + y * c;
        v.emplace_back("s5_twist", max(max(abs(xt) - 0.35, abs(yt) - 0.35), abs(z) - 1.1));
    }
    return v;
}
