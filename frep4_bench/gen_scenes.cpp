// Emit the canonical scenes as frep4 JSON (one CustomExprNode per scene; math = scenes/MATH.md).
#include "core/frep/scene.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/io/scene_io.hpp"
#include <cstdio>
#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
using namespace frep;
static const char* EXPR[5][2] = {
 {"s1_sphere","sqrt(x*x+y*y+z*z)-1.0"},
 {"s2_csg","max(max(max(abs(x),abs(y)),abs(z))-0.9, 0.0-(sqrt(x*x+y*y+z*z)-1.1))"},
 {"s3_blend","min(sqrt((x-0.45)*(x-0.45)+y*y+z*z)-0.7, sqrt((x+0.45)*(x+0.45)+y*y+z*z)-0.7)"
             " - pow(max(0.25-abs((sqrt((x-0.45)*(x-0.45)+y*y+z*z)-0.7)-(sqrt((x+0.45)*(x+0.45)+y*y+z*z)-0.7)),0.0)/0.25,3.0)*0.25/6.0"},
 {"s4_gyroid","max(sin(3.0*x)*cos(3.0*y)+sin(3.0*y)*cos(3.0*z)+sin(3.0*z)*cos(3.0*x)-0.2, sqrt(x*x+y*y+z*z)-1.4)"},
 {"s5_twist","max(max(abs(x*cos(1.2*z)+y*sin(1.2*z))-0.35, abs(0.0-x*sin(1.2*z)+y*cos(1.2*z))-0.35), abs(z)-1.1)"}
};
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    for (auto& e : EXPR) {
        SceneGraph s;
        s.add_object(std::make_shared<CustomExprNode>(e[1], e[0]));
        std::string p = dir + "/" + std::string(e[0]) + ".json";
        if (io::save_scene(s, p)) std::printf("wrote %s\n", p.c_str());
        else { std::fprintf(stderr, "save failed: %s\n", p.c_str()); return 1; }
    }
    // Wrap ported expressions (scenes/frep4/*.let from libfive/mpr scenes) as well.
    for (auto& p : std::filesystem::directory_iterator(dir))
        if (p.path().extension() == ".let") {
            std::ifstream in(p.path()); std::stringstream ss; ss << in.rdbuf();
            std::string ex = ss.str(); if (!ex.empty() && ex.back()=='\n') ex.pop_back();
            SceneGraph s;
            s.add_object(std::make_shared<CustomExprNode>(ex, p.path().stem().string()));
            io::save_scene(s, (p.path().parent_path() / (p.path().stem().string() + ".json")).string());
            std::printf("wrapped %s\n", p.path().stem().string().c_str());
        }
    return 0;
}
