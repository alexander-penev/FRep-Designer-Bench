// tools/render_advanced.cpp
//
// Demo: Plugin API, CustomExprNode, SPIR-V retargeting.
//
// Usage:
//   ./frep_advanced demo.ppm 800 600           # CPU render with plugins
//   ./frep_advanced demo.ppm 800 600 --spirv   # + SPIR-V .spv file

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/retarget_spirv.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/transforms.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/tracer/tile_scheduler.hpp"
#include "plugins/extra_primitives.hpp"

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils/Cloning.h>   // llvm::CloneModule

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void write_ppm(const std::string& path, const float* rgba, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f << "P6\n" << w << " " << h << "\n255\n";
    auto cv = [](float v) -> std::uint8_t {
        v = std::max(0.0f, std::min(1.0f, v));
        return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
    };
    for (int i = 0; i < w * h; ++i) {
        std::uint8_t c[3] = { cv(rgba[i*4]), cv(rgba[i*4+1]), cv(rgba[i*4+2]) };
        f.write(reinterpret_cast<const char*>(c), 3);
    }
    std::cout << "Wrote " << path << " (" << w << "x" << h << ")\n";
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    std::cout << "Wrote " << path << " (" << data.size() << " bytes)\n";
}

void write_text(const std::string& path, const std::string& data) {
    std::ofstream f(path);
    f << data;
    std::cout << "Wrote " << path << " (" << data.size() << " bytes of text)\n";
}

frep::SceneGraph build_phase4_scene() {
    using namespace frep;
    SceneGraph s;

    // 1. CustomExpr: true SDF of a sphere (text expression → LLVM IR)
    //    Demonstrates that the user VPL program may contain a
    //    compilable text fragment.
    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<CustomExprNode>(
                "sqrt(x*x + y*y + z*z) - 1.0",
                "user_sphere"),
            -3.5f, 0.5f, 0.0f, "tr_us"),
        Material{{0.6f, 0.4f, 0.9f}});

    // 2. CustomExpr: hard octahedron, L1 norm
    //    f = (|x| + |y| + |z| - size) / sqrt(3)
    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<CustomExprNode>(
                "(abs(x) + abs(y) + abs(z) - 1.0) * 0.57735",
                "user_oct"),
            0.0f, 0.5f, 0.0f, "tr_uo"),
        Material{{0.9f, 0.5f, 0.3f}});

    // 3. Plugin Torus — TorusPlugin registered via PluginRegistry
    auto& reg = plugin::PluginRegistry::instance();
    bool found_torus = false;
    for (auto& slot : reg.primitives()) {
        if (slot.info.name == "torus") {
            std::array<float, 2> p = {1.0f, 0.35f};
            auto torus = slot.create(p, "torus_inst");
            s.add_object(
                std::make_shared<TranslateNode>(torus, 3.5f, 0.0f, 0.0f, "tr_tor"),
                Material{{0.2f, 0.8f, 0.4f}});
            found_torus = true;
            break;
        }
    }
    if (!found_torus) std::cerr << "WARNING: torus plugin not found\n";

    // 4. Plugin Octahedron
    for (auto& slot : reg.primitives()) {
        if (slot.info.name == "octahedron") {
            std::array<float, 1> p = {0.8f};
            auto oct = slot.create(p, "oct_inst");
            s.add_object(
                std::make_shared<TranslateNode>(oct, 0.0f, -1.5f, 1.5f, "tr_oct"),
                Material{{0.95f, 0.85f, 0.1f}});
            break;
        }
    }

    // Floor
    s.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 2.0f, "floor"),
                 Material{{0.7f, 0.7f, 0.72f}});

    s.camera().position = { 0.0f, 3.0f, 8.0f };
    s.camera().target   = { 0.0f, 0.0f, 0.0f };
    s.camera().up       = { 0.0f, 1.0f, 0.0f };
    s.camera().fov_deg  = 55.0f;
    return s;
}

void print_plugins() {
    auto& reg = frep::plugin::PluginRegistry::instance();
    std::cout << "-- Registered Primitive Plugins --\n";
    for (auto& p : reg.primitives()) {
        std::cout << "  • " << p.info.name << " v" << p.info.version
                  << " — " << p.info.description << "\n";
        std::cout << "    params:";
        for (auto& n : p.param_names) std::cout << " " << n;
        std::cout << "\n";
    }
    std::cout << "-- Registered Retarget Plugins --\n";
    for (auto& r : reg.retargets()) {
        std::cout << "  • " << r.info.name << " v" << r.info.version
                  << " — " << r.info.description << "\n";
        std::cout << "    triple: " << r.triple << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init_llvm(argc, argv);

    std::string out = "phase4.ppm";
    int width = 800, height = 600;
    bool spirv = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::printf(
                "usage: %s [--out FILE] [--width N] [--height N] [--emit spirv]\n"
                "  --out FILE   output image path (default: phase4.ppm)\n"
                "  --width N    image width in pixels (default: 800)\n"
                "  --height N   image height in pixels (default: 600)\n"
                "  --emit spirv also emit SPIR-V from the scene module\n"
                "\nDemonstrates the plugin system on a built-in scene. For\n"
                "general multi-path rendering use frep_multipath; for emitting\n"
                "analysis artifacts (IR/GLSL/SPIR-V) use frep_dump.\n",
                argv[0]);
            return 0;
        }
        else if (a == "--out"    && i + 1 < argc) out = argv[++i];
        else if (a == "--width"  && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (a == "--emit"   && i + 1 < argc) { if (std::string(argv[++i]) == "spirv") spirv = true; }
    }

    std::cout << "F-Rep Designer — plugins + custom expressions + SPIR-V\n"
              << "Resolution: " << width << "x" << height << "\n";

    // Register the plugins
    frep::register_extra_primitives();
    frep::plugin::PluginRegistry::instance()
        .register_retarget(frep::SPIRVRetarget{});

    print_plugins();
    std::cout << "\n";

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    auto scene = build_phase4_scene();
    std::cout << "Objects: " << scene.objects().size() << "\n";

    // ── CPU render path ──────────────────────────────────────────────────────
    auto ctx = std::make_unique<llvm::LLVMContext>();
    frep::SceneCodegen cg(*ctx);
    try {
        cg.emit_render_tile(scene);
    } catch (const std::exception& e) {
        std::cerr << "Codegen error: " << e.what() << "\n";
        return 1;
    }
    auto t1 = clk::now();

    // ── Optional: SPIR-V emit (before handing the module to the JIT) ─────────
    if (spirv) {
        auto t_spirv_start = clk::now();
        auto* mod_ptr = cg.module();
        frep::SPIRVRetarget r;
        // Clone the module — otherwise it would change the triple in the original.
        auto cloned = llvm::CloneModule(*mod_ptr);
        auto result = r.retarget(*cloned);
        if (result) {
            if (!result->assembly.empty())
                write_text(out + ".spirv.ll", result->assembly);
            if (!result->bytes.empty())
                write_bytes(out + ".spv", result->bytes);
            auto t_spirv_end = clk::now();
            using ms = std::chrono::duration<double, std::milli>;
            std::cout << "SPIR-V emit: "
                      << ms(t_spirv_end - t_spirv_start).count() << " ms\n";
            std::cout << "  -> " << out << ".spirv.ll  ("
                      << result->assembly.size() << " B of text IR)\n";
            if (!result->bytes.empty()) {
                std::cout << "  -> " << out << ".spv  ("
                          << result->bytes.size()
                          << " B of binary SPIR-V via "
                          << r.last_translator << ")\n";
                if (!r.last_validator.empty()) {
                    std::cout << "  -> spirv-val: "
                              << (r.last_validated ? "OK" : "FAIL");
                    if (!r.last_validated && !r.last_validator_message.empty())
                        std::cout << " — " << r.last_validator_message;
                    std::cout << "\n";
                }
            } else {
                std::cout << "  -> binary SPIR-V skipped — "
                          << r.last_validator_message << "\n";
                std::cout << "     manual: llvm-spirv " << out << ".spirv.ll\n";
            }
        } else {
            std::cerr << "SPIR-V error: " << result.error() << "\n";
        }
    }

    // ── JIT + render ─────────────────────────────────────────────────────────
    frep::JitEngine jit;
    auto mod = cg.take_module();
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    if (!fn_or) {
        std::cerr << "JIT error: " << fn_or.error() << "\n";
        return 1;
    }
    auto fn = *fn_or;
    auto t2 = clk::now();

    std::vector<float> pixels(static_cast<std::size_t>(width) * height * 4, 0.0f);
    frep::RenderParams rp;
    rp.width = width; rp.height = height; rp.tile_size = 64;
    frep::TileScheduler::render(fn, pixels.data(), scene.camera(), rp);
    auto t3 = clk::now();

    write_ppm(out, pixels.data(), width, height);

    using ms = std::chrono::duration<double, std::milli>;
    std::cout << "Codegen: " << ms(t1 - t0).count() << " ms\n"
              << "JIT O3:  " << ms(t2 - t1).count() << " ms\n"
              << "Render:  " << ms(t3 - t2).count() << " ms\n";
    return 0;
}
