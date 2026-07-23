// gui/main.cpp
//
// F-Rep Designer 4.0 GUI entry point.

#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/transforms.hpp"
#include "core/io/scene_io.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/plugin/plugin_loader.hpp"
#include "gui/main_window.hpp"
#include "gui/vulkan_viewport.hpp"
#include "plugins/extra_primitives.hpp"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>

#include <QApplication>
#include <QTimer>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <iostream>
#include <memory>

namespace {

frep::SceneGraph build_default_scene() {
    using namespace frep;
    SceneGraph s;

    // Demo scene for 4.0: a mix of built-in primitives and plugin-based ones
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

    // SmoothUnion in front
    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.55f, "ba"),  0.4f, -1.2f, 1.0f, "tba");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.5f, "bb"), -0.4f, -1.2f, 1.0f, "tbb");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.4f, "blobs"),
                 Material{{0.9f, 0.7f, 0.2f}});

    s.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 1.7f, "floor"),
                 Material{{0.75f, 0.75f, 0.77f}});

    s.camera().fov_deg = 55.0f;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    // Handle --version and --help BEFORE any expensive initialisation
    // (LLVM, Qt, plugin scan). Users expect these flags to be fast and
    // not produce side-effect output from the loader.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--version") {
            std::cout << "FRep Designer 4.0.0\n";
            return 0;
        }
        if (a == "-h" || a == "--help") {
            std::cout
                << "frep_designer — FRep modeling GUI (v4.0.0)\n\n"
                << "Usage:\n"
                << "  frep_designer [--empty | --scene <file.json>]\n\n"
                << "Options:\n"
                << "  --empty           Start with no objects in the scene\n"
                << "  --scene FILE      Load FILE on startup\n"
                << "  --realtime        Use QVulkanWindow-based real-time\n"
                << "                    viewport when hardware Vulkan is\n"
                << "                    available (falls back to offscreen\n"
                << "                    rendering otherwise)\n"
                << "  -v, --version     Print version and exit\n"
                << "  -h, --help        This message\n";
            return 0;
        }
    }

    llvm::InitLLVM init_llvm(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    QApplication app(argc, argv);

    // Plugin registry — register the static plugins
    auto& reg = frep::plugin::PluginRegistry::instance();
    frep::register_extra_primitives();

    // Try to load dynamic plugins. We search several candidate
    // directories so the GUI works both when launched from the build
    // tree (development) and from an installed location (production).
    // Search order:
    //   1. ./plugins/         (relative to current working dir)
    //   2. ./build/plugins/   (typical CMake build layout)
    //   3. $(dirname argv[0])/../lib/frep/plugins/  (Unix install layout)
    //   4. $(dirname argv[0])/plugins/  (next-to-executable layout)
    std::vector<frep::plugin::LoadedPlugin> dyn_plugins;
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
        fs::current_path() / "plugins",
        fs::current_path() / "build" / "plugins",
    };
    if (argc >= 1) {
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        if (!exe_dir.empty()) {
            candidates.push_back(exe_dir / "plugins");
            candidates.push_back(exe_dir / ".." / "lib" / "frep" / "plugins");
        }
    }
    for (const auto& dir : candidates) {
        if (!fs::is_directory(dir)) continue;
        auto found = frep::plugin::LoadedPlugin::load_directory(dir, reg);
        if (!found.empty()) {
            std::cout << "Loaded " << found.size() << " plugin(s) from "
                      << dir << ":\n";
            for (const auto& p : found)
                std::cout << "  • " << fs::path(p.path()).filename().string() << "\n";
            std::move(found.begin(), found.end(),
                      std::back_inserter(dyn_plugins));
            break;  // first directory that worked wins
        }
    }
    if (dyn_plugins.empty())
        std::cout << "No dynamic plugins found in any standard location.\n";

    // Parse command-line options. Defaults to the build_default_scene
    // unless `--empty` is passed, or `--scene path.json` is given.
    bool start_empty = false;
    bool realtime    = false;
    std::string scene_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--empty")       start_empty = true;
        else if (a == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (a == "--realtime") realtime = true;
        // --version and --help are handled in the fast-path at the top
        // of main() so they don't trigger plugin loading or LLVM init.
    }

    // Build the scene.
    frep::SceneGraph scene;
    if (!scene_path.empty()) {
        try {
            scene = frep::io::load_scene(scene_path, &reg);
            std::cout << "Loaded scene: " << scene_path << " ("
                      << scene.objects().size() << " objects)\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load " << scene_path << ": "
                      << e.what() << "\n";
            return 1;
        }
    } else if (!start_empty) {
        scene = build_default_scene();
    } else {
        // Even "empty" scenes get a floor + key light so the viewport
        // isn't pitch black, which looks like a broken renderer.
        using namespace frep;
        scene.add_object(
            std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"),
            Material{{0.65f, 0.65f, 0.7f}});
        scene.lights().push_back({{5, 7, 4}, {1, 1, 0.95f}, 1.0f});
        scene.camera().position = {0, 2, 5};
        scene.camera().target   = {0, 0, 0};
    }

    // Main window
    if (realtime) {
        if (frep::gui::vulkan_viewport_available()) {
            std::cout << "Real-time GPU viewport: ENABLED "
                         "(--realtime requested, hardware Vulkan detected)\n";
        } else {
            std::cout << "Real-time GPU viewport: NOT AVAILABLE "
                         "(--realtime requested but only software Vulkan "
                         "detected). Falling back to the offscreen path.\n";
            realtime = false;
        }
    }
    frep::gui::MainWindow win(&scene, &reg, realtime);
    win.show();

    // Smoke hook: when FREP_SMOKE_CYCLE_MODES is set, cycle through the
    // three render modes after the event loop starts (each switch tears
    // down the previous backend and builds the next), then quit. Used to
    // exercise the switch_render_mode teardown/rebuild path headlessly
    // without manual interaction. Real-time is skipped automatically if
    // hardware Vulkan is unavailable (switch_render_mode guards it).
    if (std::getenv("FREP_SMOKE_CYCLE_MODES")) {
        using frep::gui::RenderMode;
        QTimer::singleShot(300, &win, [&win] {
            win.request_render_mode(RenderMode::OffscreenGPU);
        });
        QTimer::singleShot(600, &win, [&win] {
            win.request_render_mode(RenderMode::Realtime);
        });
        QTimer::singleShot(900, &win, [&win] {
            win.request_render_mode(RenderMode::OffscreenCPU);
        });
        QTimer::singleShot(1200, &win, [] {
            std::printf("FREP_SMOKE_CYCLE_MODES: cycled OK\n");
            qApp->quit();
        });
    }

    return app.exec();
}
