// tools/render_ppm.cpp
//
// PoC: scene → SceneCodegen → LLVM IR → O3 → LLJIT → TileScheduler → PPM.

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/transforms.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <llvm/Support/InitLLVM.h>

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
    auto clamp = [](float v) -> std::uint8_t {
        v = std::max(0.0f, std::min(1.0f, v));
        return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
    };
    for (int i = 0; i < w * h; ++i) {
        std::uint8_t c[3] = {
            clamp(rgba[i*4 + 0]),
            clamp(rgba[i*4 + 1]),
            clamp(rgba[i*4 + 2])
        };
        f.write(reinterpret_cast<const char*>(c), 3);
    }
    std::cout << "Wrote " << path << " (" << w << "x" << h << ")\n";
}

frep::SceneGraph build_demo_scene() {
    using namespace frep;
    SceneGraph s;

    s.add_object(std::make_shared<SphereNode>(1.2f, "sphere_c"),
                 Material{{0.9f, 0.3f, 0.2f}});

    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.8f, "sphere_l"),
            -2.5f, 0.0f, 0.0f, "tr_l"),
        Material{{0.2f, 0.6f, 0.9f}});

    // Rounded cube (Sphere ∩ Box), translated to the right
    s.add_object(
        std::make_shared<TranslateNode>(
            std::make_shared<IntersectionNode>(
                std::make_shared<BoxNode>(1.0f, 1.0f, 1.0f, "cube"),
                std::make_shared<SphereNode>(1.3f, "cube_s"),
                "rcube"),
            2.5f, 0.0f, 0.0f, "tr_r"),
        Material{{0.3f, 0.9f, 0.4f}});

    // SmoothUnion blobs in front
    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.6f, "ba"),  0.4f, -1.5f, 1.0f, "tba");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.5f, "bb"), -0.4f, -1.5f, 1.0f, "tbb");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.4f, "blobs"),
                 Material{{0.9f, 0.7f, 0.2f}});

    // Floor
    s.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 2.0f, "floor"),
                 Material{{0.7f, 0.7f, 0.7f}});

    s.camera().position = { 0.0f, 2.5f, 7.0f };
    s.camera().target   = { 0.0f, 0.0f, 0.0f };
    s.camera().up       = { 0.0f, 1.0f, 0.0f };
    s.camera().fov_deg  = 55.0f;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init_llvm(argc, argv);

    std::string outpath = "out.ppm";
    int width = 800, height = 600;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::printf(
                "usage: %s [--out FILE] [--width N] [--height N] [--dump]\n"
                "  --out FILE   output image path (default: out.ppm)\n"
                "  --width N    image width in pixels (default: 800)\n"
                "  --height N   image height in pixels (default: 600)\n"
                "  --dump       also print timing / debug info\n"
                "\nRenders a built-in demo scene (CPU_IR path). For multi-path\n"
                "rendering and comparison use frep_multipath.\n",
                argv[0]);
            return 0;
        }
        else if (a == "--out"    && i + 1 < argc) outpath = argv[++i];
        else if (a == "--width"  && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (a == "--dump")  dump = true;
    }

    std::cout << "F-Rep Designer 4.0 — PoC\n"
              << "Resolution: " << width << "x" << height << "\n";

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    auto scene = build_demo_scene();
    std::cout << "Objects: " << scene.objects().size() << "\n";

    // Codegen — the context and module are owned here, then handed to the JIT
    auto ctx = std::make_unique<llvm::LLVMContext>();
    frep::SceneCodegen cg(*ctx);
    try {
        cg.emit_render_tile(scene);
    } catch (const std::exception& e) {
        std::cerr << "Codegen error: " << e.what() << "\n";
        return 1;
    }
    auto t1 = clk::now();

    auto mod = cg.take_module();
    if (dump) {
        std::cout << "\n=== LLVM IR (before O3) ===\n";
        mod->print(llvm::errs(), nullptr);
        std::cout << "=== /IR ===\n\n";
    }

    frep::JitEngine jit;
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

    write_ppm(outpath, pixels.data(), width, height);

    using ms = std::chrono::duration<double, std::milli>;
    std::cout << "Codegen: " << ms(t1 - t0).count() << " ms\n"
              << "JIT O3:  " << ms(t2 - t1).count() << " ms\n"
              << "Render:  " << ms(t3 - t2).count() << " ms\n"
              << "Total:   " << ms(t3 - t0).count() << " ms\n";
    return 0;
}
