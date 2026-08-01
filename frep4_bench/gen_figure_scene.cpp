// gen_figure_scene — build the scene used for the paper's interface figure:
// the default five-object demo (CSG box, sphere, rounded box, smooth-union
// "blob", floor) plus four twist-deformed columns across the background.
//
// Writing it as a file rather than clicking it together in the GUI keeps the
// figure reproducible: regenerate, load with
//   frep_designer --scene scenes/frep4/fig_interface.json
// and the same picture comes back.
#include "core/frep/scene.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/deformations.hpp"
#include "core/io/scene_io.hpp"

#include <cstdio>
#include <memory>
#include <string>

using namespace frep;

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "scenes/frep4/fig_interface.json";
    SceneGraph s;

    // ── Foreground: the five demo objects (as in the GUI's default scene) ────
    s.add_object(std::make_shared<SphereNode>(1.0f, "sphere"),
                 Material{{0.9f, 0.3f, 0.2f}});
    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "box"),
            -2.5f, 0.0f, 0.0f, "tr_box"),
        Material{{0.2f, 0.6f, 0.9f}});
    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<IntersectionNode>(
                std::make_shared<BoxNode>(0.9f, 0.9f, 0.9f, "rb_box"),
                std::make_shared<SphereNode>(1.15f, "rb_s"),
                "rounded"),
            2.5f, 0.0f, 0.0f, "tr_rb"),
        Material{{0.3f, 0.9f, 0.4f}});

    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.55f, "ba"),  0.4f, -1.2f, 1.0f, "tba");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.5f, "bb"), -0.4f, -1.2f, 1.0f, "tbb");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.4f, "blobs"),
                 Material{{0.9f, 0.7f, 0.2f}});

    s.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 1.7f, "floor"),
                 Material{{0.75f, 0.75f, 0.77f}});

    // ── Background: four twisted columns ────────────────────────────────────
    // A tall box twisted about Y, then translated into place. The twist is the
    // non-linear deformation of Section 4 applied to a built-in primitive, so
    // the figure shows a deformation and a CSG operator in the same frame.
    const float xs[4] = {-3.6f, -1.2f, 1.2f, 3.6f};
    const Material col{{0.62f, 0.58f, 0.78f}};
    for (int i = 0; i < 4; ++i) {
        auto shaft = std::make_shared<BoxNode>(0.40f, 2.6f, 0.40f,
                                               "col_box" + std::to_string(i));
        auto tw = std::make_shared<TwistYNode>(shaft, 0.45f,
                                               "twist" + std::to_string(i));
        s.add_object(std::make_shared<TranslateNode>(tw, xs[i], 0.9f, -4.6f,
                                                     "col" + std::to_string(i)),
                     col);
    }

    s.camera().fov_deg = 55.0f;

    if (!io::save_scene(s, out)) {
        std::fprintf(stderr, "failed to write %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s (%zu objects)\n", out.c_str(), s.objects().size());
    return 0;
}
