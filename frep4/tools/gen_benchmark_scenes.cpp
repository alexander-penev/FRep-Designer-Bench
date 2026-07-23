// tools/gen_benchmark_scenes.cpp
//
// Generates a set of benchmark_*.json scenes into examples/ for manual
// performance testing, so testers don't have to hand-build complex scenes
// or remember parameters. Two families:
//
//   simple  — cheap per-object SDFs (spheres/boxes). The spatial-guard
//             heuristic should leave these Inlined (guarding doesn't help).
//   heavy   — expensive per-object SDFs (twist + smooth-union + CSG). The
//             heuristic should switch these to Guarded once calibrated.
//
// Built through the real SceneGraph API + io::save_scene, so the files are
// always schema-valid. Re-run to regenerate.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/deformations.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/node.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

using namespace frep;

namespace {

Material mat(float r, float g, float b, float rough = 0.5f) {
    Material m;
    m.albedo = {r, g, b};
    m.roughness = rough;
    return m;
}

void add_floor(SceneGraph& s) {
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"),
                 mat(0.55f, 0.55f, 0.55f));
}

// Place a camera that frames a cube of objects of the given half-extent.
void frame(SceneGraph& s, float extent) {
    float d = extent * 2.2f + 3.0f;
    s.camera().position = {d, d * 0.6f, d};
    s.camera().target   = {0, extent * 0.3f, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{d, d * 1.5f, d * 0.5f}, {1.0f, 0.97f, 0.9f}, 1.0f});
}

// ── one expensive object: twisted box smooth-unioned with a sphere ──────────
FRepNode::Ptr heavy_object(const std::string& id) {
    auto twisted = std::make_shared<TwistYNode>(
        std::make_shared<BoxNode>(0.35f, 0.55f, 0.35f, "b" + id), 2.5f, "tw" + id);
    return std::make_shared<SmoothUnionNode>(
        twisted,
        std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.3f, "s" + id), 0.3f, 0.2f, 0, "st" + id),
        0.25f, "su" + id);
}

// ── one moderately expensive object: a CSG difference (box minus sphere) ────
FRepNode::Ptr csg_object(const std::string& id) {
    return std::make_shared<DifferenceNode>(
        std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "b" + id),
        std::make_shared<SphereNode>(0.62f, "s" + id),
        "d" + id);
}

// Grid helper: calls make(id) for each cell, translating onto a grid.
template <class MakeFn>
void grid(SceneGraph& s, int count, float spacing, MakeFn make,
          float r, float g, float b) {
    int side = std::max(1, (int)std::ceil(std::cbrt((double)count)));
    float off = (side - 1) * spacing * 0.5f;
    int m = 0;
    for (int i = 0; i < side && m < count; ++i)
      for (int j = 0; j < side && m < count; ++j)
        for (int k = 0; k < side && m < count; ++k, ++m) {
            std::string id = std::to_string(m);
            s.add_object(std::make_shared<TranslateNode>(
                make(id), i*spacing-off, j*spacing-off + spacing*0.5f, k*spacing-off,
                "t" + id), mat(r, g, b));
        }
    frame(s, off + spacing);
}

std::string outdir;

void write(SceneGraph& s, const std::string& name, const char* desc) {
    std::string path = outdir + "/" + name;
    if (io::save_scene(s, path)) {
        // Report object count + average node count so the file's intent is
        // clear (which heuristic branch it exercises).
        long total = 0; int n = 0;
        for (const auto& [id, obj] : s.objects()) {
            if (!obj.geometry) continue;
            total += node_count(*obj.geometry); ++n;
        }
        std::printf("  %-32s  %3d objects, avg %.1f nodes/obj  — %s\n",
                    name.c_str(), n, n ? (double)total/n : 0.0, desc);
    } else {
        std::printf("  FAILED to write %s\n", name.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    outdir = (argc > 1) ? argv[1] : "examples";
    std::printf("Generating benchmark scenes into %s/\n\n", outdir.c_str());

    // ── SIMPLE: cheap primitives, guard should stay off ─────────────────────
    {
        SceneGraph s;
        grid(s, 27, 1.6f, [](const std::string& id) {
            return std::make_shared<SphereNode>(0.5f, "s" + id);
        }, 0.4f, 0.7f, 0.95f);
        add_floor(s);
        write(s, "benchmark_simple_spheres_27.json",
              "cheap SDFs; expect Inlined (guarding doesn't help)");
    }
    {
        SceneGraph s;
        grid(s, 64, 1.5f, [](const std::string& id) {
            return std::make_shared<BoxNode>(0.4f, 0.4f, 0.4f, "b" + id);
        }, 0.9f, 0.6f, 0.3f);
        add_floor(s);
        write(s, "benchmark_simple_boxes_64.json",
              "cheap SDFs; expect Inlined");
    }
    {
        SceneGraph s;
        grid(s, 125, 1.4f, [](const std::string& id) {
            return std::make_shared<SphereNode>(0.45f, "s" + id);
        }, 0.5f, 0.85f, 0.55f);
        add_floor(s);
        write(s, "benchmark_simple_spheres_125.json",
              "many cheap SDFs; stress count, still Inlined");
    }

    // ── HEAVY: expensive per-object SDFs, guard should kick in ──────────────
    {
        SceneGraph s;
        grid(s, 27, 1.7f, heavy_object, 0.8f, 0.4f, 0.6f);
        add_floor(s);
        write(s, "benchmark_heavy_twist_27.json",
              "twist+smooth-union per object; expect Guarded (~3x)");
    }
    {
        SceneGraph s;
        grid(s, 64, 1.7f, heavy_object, 0.7f, 0.5f, 0.85f);
        add_floor(s);
        write(s, "benchmark_heavy_twist_64.json",
              "twist+smooth-union; expect Guarded (~6x)");
    }
    {
        SceneGraph s;
        grid(s, 125, 1.7f, heavy_object, 0.85f, 0.45f, 0.4f);
        add_floor(s);
        write(s, "benchmark_heavy_twist_125.json",
              "large heavy scene; expect Guarded, slow inline baseline");
    }
    {
        SceneGraph s;
        grid(s, 48, 1.6f, csg_object, 0.6f, 0.7f, 0.9f);
        add_floor(s);
        write(s, "benchmark_heavy_csg_48.json",
              "CSG difference per object; moderate cost, expect Guarded");
    }
    // Mixed: half cheap, half heavy — exercises averaging in the heuristic.
    {
        SceneGraph s;
        int count = 48, side = 4; float sp = 1.7f;
        float off = (side - 1) * sp * 0.5f;
        int m = 0;
        for (int i = 0; i < side && m < count; ++i)
          for (int j = 0; j < side && m < count; ++j)
            for (int k = 0; k < side && m < count; ++k, ++m) {
                std::string id = std::to_string(m);
                FRepNode::Ptr g = (m % 2 == 0)
                    ? heavy_object(id)
                    : FRepNode::Ptr(std::make_shared<SphereNode>(0.5f, "s" + id));
                s.add_object(std::make_shared<TranslateNode>(
                    g, i*sp-off, j*sp-off + sp*0.5f, k*sp-off, "t" + id),
                    mat(0.7f, 0.7f, 0.5f));
            }
        frame(s, off + sp);
        add_floor(s);
        write(s, "benchmark_mixed_48.json",
              "half cheap/half heavy; tests average-cost heuristic");
    }

    std::printf("\nDone. Load any in the app, or render headless to compare.\n");
    return 0;
}
