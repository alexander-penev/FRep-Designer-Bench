// Canonical scene set as libfive trees (see scenes/MATH.md). f<=0 inside.
#pragma once
#include <libfive/tree/tree.hpp>
#include <cmath>
#include <string>
#include <vector>
using libfive::Tree;
inline Tree len(Tree x, Tree y, Tree z) { return sqrt(x*x + y*y + z*z); }

// Complex scenes to replace the broken imported .frep archives (hello_world /
// architecture / involute_gear), whose roots were atan2/division — not signed
// distance fields, so every evaluator (libfive interval render included) draws
// them empty. These are built purely from min/max/sqrt/abs/rotation, so they
// are valid SDFs by construction: finite everywhere, negative inside, and they
// convert and evaluate identically across every backend.
inline std::vector<std::pair<std::string, Tree>> complex_scenes() {
    Tree x = Tree::X(), y = Tree::Y(), z = Tree::Z();
    Tree r2 = sqrt(x*x + y*y);                       // 2D radius
    std::vector<std::pair<std::string, Tree>> v;

    { // c1_gear: hobbed disc — body minus hub bore, union of N radial teeth
        Tree body = max(r2 - Tree(0.85), abs(z) - Tree(0.22));
        Tree bore = max(r2 - Tree(0.28), abs(z) - Tree(0.5));
        Tree g = max(body, -bore);
        const int N = 18;
        for (int i = 0; i < N; ++i) {
            double a = 2.0 * M_PI * i / N, c = std::cos(a), s = std::sin(a);
            Tree xt = x * Tree(c) + y * Tree(s);
            Tree yt = x * Tree(-s) + y * Tree(c);
            Tree tooth = max(max(abs(xt - Tree(0.92)) - Tree(0.11),
                                 abs(yt) - Tree(0.09)), abs(z) - Tree(0.22));
            g = min(g, tooth);
        }
        v.emplace_back("c1_gear", g);
    }
    { // c2_colonnade: floor + roof slabs joined by a 5x5 grid of columns
        Tree g = abs(z + Tree(0.85)) - Tree(0.12);             // floor slab
        for (int i = -2; i <= 2; ++i)
            for (int j = -2; j <= 2; ++j) {
                Tree cx(0.55 * i), cy(0.55 * j);
                Tree col = max(sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy))
                               - Tree(0.11), abs(z) - Tree(0.78));
                g = min(g, col);
            }
        g = min(g, abs(z - Tree(0.85)) - Tree(0.12));          // roof slab
        v.emplace_back("c2_colonnade", g);
    }
    return v;
}
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
