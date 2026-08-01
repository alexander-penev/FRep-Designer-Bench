// Re-import the repetitive benchmark scenes with structure: build the colonnade
// and the gear as a node-graph union of instanced parts (one shared prototype +
// N placements) instead of one flattened CustomExpr. factor_instances then wires
// the copies to a shared prototype, so partition_csg_groups sees N separable
// groups -> the RTX path gets a real multi-BLAS TLAS. Verifies the structured
// field is identical to the monolithic .json, and writes *_struct.json.
#include "core/frep/scene.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/smart_import.hpp"
#include "core/io/scene_io.hpp"
#include "core/compiler/compile_sdf.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace frep;

// Union of a base slab + 5x5 capped-cylinder columns (radius 0.11, z-cap 0.78),
// matching c2_colonnade.let.
static SceneGraph build_colonnade() {
    std::vector<FRepNode::Ptr> parts;
    parts.push_back(std::make_shared<CustomExprNode>("abs(z+0.85)-0.12", "floor"));
    parts.push_back(std::make_shared<CustomExprNode>("abs(z-0.85)-0.12", "ceil"));
    int k = 0;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            float px = -1.1f + 0.55f * i, py = -1.1f + 0.55f * j;
            auto col = std::make_shared<CustomExprNode>(
                "max(sqrt(x*x+y*y)-0.11, abs(z)-0.78)", "col" + std::to_string(k));
            col->set_bounds({-0.11f, -0.11f, -0.78f, 0.11f, 0.11f, 0.78f});
            parts.push_back(std::make_shared<TranslateNode>(col, px, py, 0.f,
                                                            "tc" + std::to_string(k)));
            ++k;
        }
    auto root = union_all(std::move(parts));
    root->id = "colonnade";
    SceneGraph s; s.add_object(root);
    return s;
}

// Union of a gear body (annulus) + 18 rotated teeth (tooth on +x, 20 deg apart),
// matching c1_gear.let.
static SceneGraph build_gear() {
    std::vector<FRepNode::Ptr> parts;
    {
        auto body = std::make_shared<CustomExprNode>(
            "max(max(sqrt(x*x+y*y)-0.85, abs(z)-0.22),"
            " 0.0-(max(sqrt(x*x+y*y)-0.28, abs(z)-0.5)))", "body");
        body->set_bounds({-0.85f, -0.85f, -0.22f, 0.85f, 0.85f, 0.22f});  // compact annulus
        parts.push_back(body);
    }
    for (int k = 0; k < 18; ++k) {
        auto tooth = std::make_shared<CustomExprNode>(
            "max(max(abs(x-0.92)-0.11, abs(y)-0.09), abs(z)-0.22)",
            "tooth" + std::to_string(k));
        tooth->set_bounds({0.81f, -0.09f, -0.22f, 1.03f, 0.09f, 0.22f});
        float a = k * 2.0f * 3.14159265358979f / 18.0f;
        parts.push_back(std::make_shared<RotateZNode>(tooth, a, "rz" + std::to_string(k)));
    }
    auto root = union_all(std::move(parts));
    root->id = "gear";
    SceneGraph s; s.add_object(root);
    return s;
}

// Max |structured - monolithic| over a grid; 0 (to float eps) means identical.
static double parity(const SceneGraph& a, const std::string& mono_json) {
    auto ca = jit::compile_scene_sdf(a);
    SceneGraph b = io::load_scene(mono_json);
    auto cb = jit::compile_scene_sdf(b);
    if (!ca || !cb) { std::printf("  compile failed\n"); return 1e9; }
    double md = 0;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            for (int k = 0; k < 20; ++k) {
                float x = -1.6f + 3.2f * i / 19, y = -1.6f + 3.2f * j / 19,
                      z = -1.3f + 2.6f * k / 19;
                double d = std::fabs((double)ca->fn(x, y, z) - cb->fn(x, y, z));
                if (d > md) md = d;
            }
    return md;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "scenes/frep4";

    struct { const char* name; SceneGraph (*build)(); } scenes[] = {
        {"c2_colonnade", build_colonnade},
        {"c1_gear",      build_gear},
    };
    for (auto& sc : scenes) {
        SceneGraph s = sc.build();
        int n = factor_instances(s);
        double md = parity(s, dir + "/" + sc.name + ".json");
        std::printf("%-14s -> %2d instances of 1 prototype; parity vs monolithic: max %.6f  %s\n",
                    sc.name, n, md, md < 1e-3 ? "IDENTICAL" : "DIFFERS");
        std::string out = dir + "/" + sc.name + "_struct.json";
        if (io::save_scene(s, out)) std::printf("               wrote %s\n", out.c_str());
    }
    return 0;
}
