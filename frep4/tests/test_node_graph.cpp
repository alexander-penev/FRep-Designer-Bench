// Headless test of NodeGraphScene::build_tree() — without an X server.
// Builds a graph programmatically, connects ports, calls build_tree(), checks the tree.
#include "gui/node_graph.hpp"
#include "core/frep/node.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/plugin/plugin_api.hpp"
#include "plugins/extra_primitives.hpp"
#include <QApplication>
#include <cstdio>
#include <cassert>

using namespace frep;
using namespace frep::gui;

int main(int argc, char** argv) {
    // QGraphicsScene requires QApplication (Widgets), not QCoreApplication.
    QApplication app(argc, argv);
    NodeGraphScene scene;

    // Build: Translate(Union(Sphere, Box))
    auto* sphere = scene.add_node("Sphere", QPointF(-400, -50));
    auto* box    = scene.add_node("Box",    QPointF(-400,  100));
    auto* uni    = scene.add_node("Union",  QPointF(-200,   25));
    auto* trans  = scene.add_node("Translate", QPointF(0,   25));
    assert(sphere && box && uni && trans);

    // Connect: Sphere -> Union.in[0], Box -> Union.in[1]
    bool c1 = scene.connect_ports(sphere->output_port(), uni->input_port(0));
    bool c2 = scene.connect_ports(box->output_port(),    uni->input_port(1));
    printf("Sphere->Union[0]: %s, Box->Union[1]: %s\n",
           c1 ? "OK" : "FAIL", c2 ? "OK" : "FAIL");

    // Union → Translate.in[0]
    bool c3 = scene.connect_ports(uni->output_port(), trans->input_port(0));
    printf("Union->Translate[0]: %s\n", c3 ? "OK" : "FAIL");

    // Translate → Output
    bool c4 = scene.connect_ports(trans->output_port(),
                                  scene.output_node()->input_port(0));
    printf("Translate->Output: %s\n", c4 ? "OK" : "FAIL");

    // Build tree
    auto tree = scene.build_tree();
    if (!tree) { printf("FAIL: build_tree returned nullptr\n"); return 1; }
    printf("\nBuilt tree:\n");
    printf("  root: %s\n", tree->type_name());
    assert(std::string(tree->type_name()) == "Translate");
    assert(tree->children.size() == 1);
    auto* child = tree->children[0].get();
    printf("  child: %s\n", child->type_name());
    assert(std::string(child->type_name()) == "Union");
    assert(child->children.size() == 2);
    printf("  grandchild[0]: %s\n", child->children[0]->type_name());
    printf("  grandchild[1]: %s\n", child->children[1]->type_name());
    assert(std::string(child->children[0]->type_name()) == "Sphere");
    assert(std::string(child->children[1]->type_name()) == "Box");

    // Cycle-detection test: trans->uni should fail, because uni is upstream of trans.
    bool cycle = scene.connect_ports(trans->output_port(), uni->input_port(0));
    printf("\nCycle detection (trans->union, must FAIL): %s\n",
           cycle ? "FAIL (allowed a cycle!)" : "OK (rejected the cycle)");
    assert(!cycle);

    // Test: load_from_tree round-trip
    auto manual = std::make_shared<DifferenceNode>(
        std::make_shared<SphereNode>(2.0f, "s"),
        std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "b"), "diff");
    scene.load_from_tree(manual);
    auto rebuilt = scene.build_tree();
    if (!rebuilt) { printf("FAIL: load_from_tree round-trip\n"); return 1; }
    printf("\nload_from_tree round-trip: root=%s\n", rebuilt->type_name());
    assert(std::string(rebuilt->type_name()) == "Difference");
    assert(rebuilt->children.size() == 2);

    // ── Plugin nodes in the graph ────────────────────────────────────────────
    // Register a plugin primitive (Torus), add it via add_node, set a custom
    // parameter, build_tree, and verify the produced FRepNode reflects the
    // edited values.
    {
        plugin::PluginRegistry reg;
        register_extra_primitives_into(reg);

        NodeGraphScene plug_scene;
        plug_scene.set_registry(&reg);

        auto* torus = plug_scene.add_node("Torus", QPointF(-200, 0));
        if (!torus) {
            printf("FAIL: plugin Torus node could not be added\n");
            return 1;
        }
        // Edit one parameter — emulates the user double-clicking and typing
        // a new value.
        torus->param_values()["R"] = 1.75f;

        // Connect to output.
        bool connected = plug_scene.connect_ports(
            torus->output_port(),
            plug_scene.output_node()->input_port(0));
        assert(connected);

        auto plug_tree = plug_scene.build_tree();
        if (!plug_tree) {
            printf("FAIL: build_tree with plugin node returned nullptr\n");
            return 1;
        }
        printf("Plugin tree: root=%s\n", plug_tree->type_name());
        assert(std::string(plug_tree->type_name()) == "Torus");
        // The user-edited value of R must have flowed through.
        assert(plug_tree->params.at("R") == 1.75f);
        // The unedited r took its default.
        assert(plug_tree->params.count("r") == 1);
        printf("Plugin tree params: R=%g r=%g\n",
               plug_tree->params.at("R"), plug_tree->params.at("r"));
    }

    printf("\n=== All checks passed ===\n");
    return 0;
}
