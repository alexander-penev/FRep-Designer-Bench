// tools/gallery.cpp
//
// Generates the project's examples gallery. Each scene exercises a
// different combination of features and writes a PPM (and stats line)
// next to it. The intention is that the gallery doubles as a regression
// reference — if any of these images change unexpectedly, something
// regressed.
//
// Scenes:
//   01_csg            — pure CSG (Union/Intersection/Difference/SmoothUnion)
//   02_deformations   — twisted box, bent capsule-like shape, tapered cylinder
//   03_mesh_carve     — imported sphere mesh, carved with a box (MeshSDF + Diff)
//   04_patterns       — five spheres, one per procedural pattern
//   05_cpu_vs_gpu     — two side-by-side renders of the same scene
//   06_texture        — triplanar texture mapping on a sphere and a cube
//   07_hero           — busy scene combining CSG + deformations + patterns
//
// Each scene fits in ~30-50 lines of C++ — handy as copy-paste starting
// points for new users.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/deformations.hpp"
#include "core/frep/instance.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/frep/mesh_sdf.hpp"
#include "core/mesh/marching_cubes.hpp"
#include "core/io/bmp_loader.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/tracer/tile_scheduler.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/postprocess/post_process.hpp"

#include <llvm/Support/TargetSelect.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace frep;

// ── Common helpers ─────────────────────────────────────────────────────────

static void save_ppm(const std::string& path,
                     const std::vector<std::uint8_t>& rgba, int W, int H)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i) {
        f.put(static_cast<char>(rgba[i*4 + 0]));
        f.put(static_cast<char>(rgba[i*4 + 1]));
        f.put(static_cast<char>(rgba[i*4 + 2]));
    }
}

static void save_ppm_float(const std::string& path,
                           const std::vector<float>& px, int W, int H)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i)
        for (int c = 0; c < 3; ++c)
            f.put(static_cast<char>(std::clamp(px[i*4 + c], 0.0f, 1.0f) * 255.0f));
}

static void normalize3(float v[3]) {
    float L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (L > 0) { v[0]/=L; v[1]/=L; v[2]/=L; }
}

// Set up camera + lights with a consistent look across all gallery scenes.
static void setup_lighting(SceneGraph& s,
                           std::array<float,3> cam_pos = {0, 1.8f, 5.5f},
                           std::array<float,3> cam_tgt = {0, 0, 0})
{
    s.camera().position = cam_pos;
    s.camera().target   = cam_tgt;
    auto& L = s.lights(); L.clear();
    L.push_back({{ 5,  7,  4}, {1.0f, 0.97f, 0.9f}, 1.0f});
    L.push_back({{-4,  3, -2}, {0.4f, 0.5f,  0.7f}, 0.4f});  // cool fill
}

// Build push constants matching the scene camera. Delegates to the
// shared builder so we get multi-light support and the right field
// layout automatically.
static gpu::ShaderPush build_push(const SceneGraph& s, int W, int H) {
    return gpu::build_push_from_scene(s, W, H);
}

// Render a scene on the GPU, save PPM, log timing.
static bool render_gpu(const SceneGraph& s, const std::string& out_path,
                       int W = 800, int H = 500)
{
    auto t0 = std::chrono::steady_clock::now();
    auto e = gpu::GlslEmitter::emit(s);
    if (!e) {
        std::fprintf(stderr, "  emit: %s\n", e.error().c_str());
        return false;
    }
    auto spv = gpu::compile_glsl_to_spv_managed(e->source);
    if (!spv) {
        std::fprintf(stderr, "  compile: %s\n", spv.error().c_str());
        return false;
    }
    auto ctx_or = gpu::VulkanCtx::create(spv->path(), e->mesh_voxels, e->texture_pixels);
    if (!ctx_or) {
        std::fprintf(stderr, "  vk: %s\n", ctx_or.error().c_str());
        return false;
    }
    auto& ctx = **ctx_or;
    std::vector<std::uint8_t> px;
    auto rr = ctx.render(build_push(s, W, H), px);
    if (!rr) {
        std::fprintf(stderr, "  render: %s\n", rr.error().c_str());
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    save_ppm(out_path, px, W, H);
    std::printf("  ✓ %s (%dx%d, %d objs, render %.1f ms, total %.1f ms)\n",
                out_path.c_str(), W, H, e->object_count,
                ctx.stats().render_ms, total_ms);
    return true;
}

// Supersampling factor for gallery images — render at SS× and box-downsample
// so demo images have clean silhouettes with no stray edge pixels.
static int g_ssaa = 2;

// Render a scene on the CPU JIT path, save PPM, log timing.
static bool render_cpu(const SceneGraph& s, const std::string& out_path,
                       int W = 800, int H = 500)
{
    auto t0 = std::chrono::steady_clock::now();
    const int rw = W * g_ssaa, rh = H * g_ssaa;
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg; cfg.enable_shadows = true; cfg.enable_ao = false;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(s);
    auto mod = cg.take_module();
    JitEngine jit;
    auto fn_or = jit.load(std::move(mod), std::move(ctx));
    if (!fn_or) {
        std::fprintf(stderr, "  JIT failed\n");
        return false;
    }
    RenderParams rp; rp.width = rw; rp.height = rh;
    std::vector<float> px(rw * rh * 4, 0);
    TileScheduler::render(*fn_or, px.data(), s.camera(), rp);
    // Box-downsample SS× → target resolution.
    std::vector<float> out = px;
    int ow = rw, oh = rh;
    if (g_ssaa > 1) {
        post::BoxDownsampleSSAA ss(g_ssaa);
        post::Frame o = ss.apply(post::Frame(std::move(px), rw, rh));
        out = std::move(o.rgba); ow = o.w; oh = o.h;
    }
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    save_ppm_float(out_path, out, ow, oh);
    std::printf("  ✓ %s (%dx%d, CPU JIT, SSAA %dx, %.1f ms)\n",
                out_path.c_str(), ow, oh, g_ssaa, total_ms);
    return true;
}

// ── Scene 01: pure CSG ─────────────────────────────────────────────────────

static SceneGraph scene_csg() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // 1. Difference: sphere - box (left)
    Material m1{{0.95f, 0.5f, 0.2f}};
    auto sphA = std::make_shared<SphereNode>(0.95f, "sA");
    auto cubeA = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.55f, 0.55f, 0.55f, "bA"),
        0.55f, 0.45f, 0.45f, "btA");
    auto carved = std::make_shared<DifferenceNode>(sphA, cubeA, "diffA");
    s.add_object(std::make_shared<TranslateNode>(carved, -2.0f, 0, 0, "ltA"), m1);

    // 2. SmoothUnion (center)
    Material m2{{0.4f, 0.95f, 0.5f}};
    auto sa = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.6f, "sa"), -0.35f, 0, 0, "sat");
    auto sb = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.55f, "sb"), 0.35f, 0.15f, 0, "sbt");
    s.add_object(std::make_shared<SmoothUnionNode>(sa, sb, 0.3f, "blob"), m2);

    // 3. Intersection: sphere ∩ box (right)
    Material m3{{0.3f, 0.5f, 0.95f}};
    auto si = std::make_shared<SphereNode>(0.9f, "si");
    auto bi = std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "bi");
    auto lens = std::make_shared<IntersectionNode>(si, bi, "lens");
    s.add_object(std::make_shared<TranslateNode>(lens, 2.0f, 0, 0, "ltC"), m3);
    setup_lighting(s);
    return s;
}

// ── Scene 02: non-linear deformations ──────────────────────────────────────

static SceneGraph scene_deformations() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Twisted box (left)
    Material m1{{0.95f, 0.4f, 0.3f}};
    auto bx = std::make_shared<BoxNode>(0.35f, 1.0f, 0.35f, "b");
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<TwistYNode>(bx, 2.0f, "tw"),
        -2.0f, 0, 0, "twt"), m1);

    // Bent thin box (center)
    Material m2{{0.4f, 0.95f, 0.6f}};
    auto bx2 = std::make_shared<BoxNode>(0.25f, 0.7f, 0.4f, "b2");
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BendXYNode>(bx2, 0.4f, "bd"),
        0, 0, 0, "bdt"), m2);

    // Tapered cylinder-like (right)
    Material m3{{0.3f, 0.5f, 0.95f}};
    auto bx3 = std::make_shared<BoxNode>(0.5f, 0.8f, 0.5f, "b3");
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<TaperYNode>(bx3, 0.25f, 1.6f, "tp"),
        2.0f, 0, 0, "tpt"), m3);
    setup_lighting(s);
    return s;
}

// ── Scene 03: imported mesh + CSG carve ────────────────────────────────────

static SceneGraph scene_mesh_carve() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Generate a sphere mesh, then voxelize.
    SceneGraph ref;
    ref.add_object(std::make_shared<SphereNode>(1.0f));
    mesh::MarchingCubesParams mcp; mcp.rx=mcp.ry=mcp.rz=24;
    auto mesh = mesh::extract_iso_mesh(ref, mcp);
    auto voxel_sphere = std::make_shared<MeshSDFNode>(mesh, 48, "msph");

    // Difference vs a box.
    Material m1{{0.9f, 0.6f, 0.2f}};
    auto box = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b"),
        0.55f, 0.55f, 0.55f, "bt");
    auto carved = std::make_shared<DifferenceNode>(voxel_sphere, box, "carve");
    s.add_object(carved, m1);
    setup_lighting(s, {0, 1.5f, 4}, {0, 0, 0});
    return s;
}

// ── Scene 04: procedural patterns gallery ──────────────────────────────────

static SceneGraph scene_patterns() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 0.7f, "floor"), mp);

    auto place = [&](float x, Material m, const std::string& id) {
        s.add_object(std::make_shared<TranslateNode>(
            std::make_shared<SphereNode>(0.55f, id), x, 0, 0, id+"_t"), m);
    };

    Material m1{{0.9f, 0.3f, 0.3f}};               place(-3.0f, m1, "solid");
    Material m2;
    m2.pattern=Material::Pattern::Checker;
    m2.albedo={0.95f,0.95f,0.95f}; m2.albedo2={0.1f,0.1f,0.1f};
    m2.pattern_scale=6.0f;                          place(-1.5f, m2, "checker");
    Material m3;
    m3.pattern=Material::Pattern::Stripes;
    m3.albedo={0.9f,0.9f,0.4f}; m3.albedo2={0.6f,0.3f,0.7f};
    m3.pattern_scale=8.0f;                          place(0.0f, m3, "stripes");
    Material m4;
    m4.pattern=Material::Pattern::GradientY;
    m4.albedo={0.2f,0.4f,0.9f}; m4.albedo2={0.95f,0.6f,0.2f};
    m4.pattern_scale=0.6f;                          place(1.5f, m4, "gradient");
    Material m5;
    m5.pattern=Material::Pattern::Noise;
    m5.albedo={0.3f,0.85f,0.4f}; m5.albedo2={0.95f,0.95f,0.6f};
    m5.pattern_scale=25.0f;                         place(3.0f, m5, "noise");

    setup_lighting(s, {0, 1.5f, 5});
    return s;
}

// ── Scene 05: same scene CPU and GPU ───────────────────────────────────────

static SceneGraph scene_compare() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    Material m1{{0.95f, 0.4f, 0.3f}};
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.9f, "s1"),
        -1.4f, 0, 0, "s1t"), m1);

    Material m2{{0.3f, 0.7f, 0.95f}};
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b1"),
        1.4f, 0, 0, "b1t"), m2);

    setup_lighting(s);
    return s;
}

// ── Scene 06: triplanar texture mapping ───────────────────────────────────

static SceneGraph scene_texture() {
    SceneGraph s;
    Material mp{{0.5f, 0.5f, 0.5f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Build a small procedural texture: bullseye + radial color shift.
    io::Image tex;
    tex.width = 128; tex.height = 128;
    tex.rgba.assign(128 * 128 * 4, 0);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
            std::size_t i = (y * 128 + x) * 4;
            float dx = (x - 64) / 64.0f, dy = (y - 64) / 64.0f;
            float r = std::sqrt(dx*dx + dy*dy);
            float ring = std::sin(r * 12) * 0.5f + 0.5f;
            tex.rgba[i+0] = static_cast<std::uint8_t>(255 * (1 - r) * ring);
            tex.rgba[i+1] = static_cast<std::uint8_t>(200 * (x / 128.0f));
            tex.rgba[i+2] = static_cast<std::uint8_t>(255 * (1 - r) * (1 - ring));
            tex.rgba[i+3] = 255;
        }

    Material m1;
    m1.pattern = Material::Pattern::Texture;
    m1.texture_rgba   = tex.rgba;
    m1.texture_width  = tex.width;
    m1.texture_height = tex.height;
    m1.pattern_scale  = 1.0f;
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "s"), -1.4f, 0, 0, "st"), m1);

    Material m2;
    m2.pattern = Material::Pattern::Texture;
    m2.texture_rgba   = tex.rgba;
    m2.texture_width  = tex.width;
    m2.texture_height = tex.height;
    m2.pattern_scale  = 3.0f;
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.7f, 0.7f, 0.7f, "b"), 1.4f, 0, 0, "bt"), m2);
    setup_lighting(s);
    return s;
}

// ── Scene 07: hero showcase ───────────────────────────────────────────────

static SceneGraph scene_hero() {
    SceneGraph s;
    Material mp;
    mp.pattern = Material::Pattern::Checker;
    mp.albedo  = {0.65f, 0.65f, 0.7f};
    mp.albedo2 = {0.3f, 0.3f, 0.35f};
    mp.pattern_scale = 1.5f;
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Twisted column with stripes (left)
    Material m1;
    m1.pattern = Material::Pattern::Stripes;
    m1.albedo  = {0.95f, 0.6f, 0.2f};
    m1.albedo2 = {0.5f, 0.2f, 0.6f};
    m1.pattern_scale = 6.0f;
    auto col = std::make_shared<BoxNode>(0.3f, 1.0f, 0.3f, "col");
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<TwistYNode>(col, 2.5f, "tw"),
        -1.8f, 0, 0, "twt"), m1);

    // SmoothUnion blob with noise (center)
    Material m2;
    m2.pattern = Material::Pattern::Noise;
    m2.albedo  = {0.3f, 0.85f, 0.95f};
    m2.albedo2 = {0.1f, 0.4f, 0.6f};
    m2.pattern_scale = 12.0f;
    auto a = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.55f, "a"), -0.3f, 0.1f, 0, "at");
    auto b = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.5f, "b"),  0.3f, -0.1f, 0, "bt");
    s.add_object(std::make_shared<SmoothUnionNode>(a, b, 0.25f, "blob"), m2);

    // Difference: sphere - box, gradient material (right)
    Material m3;
    m3.pattern = Material::Pattern::GradientY;
    m3.albedo  = {0.95f, 0.4f, 0.4f};
    m3.albedo2 = {1.0f, 0.95f, 0.6f};
    m3.pattern_scale = 0.9f;
    auto sp = std::make_shared<SphereNode>(0.85f, "sp");
    auto cu = std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.55f, 0.55f, 0.55f, "cu"),
        0.5f, 0.5f, 0.5f, "cut");
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<DifferenceNode>(sp, cu, "diff"),
        1.9f, 0, 0, "dt"), m3);

    setup_lighting(s, {0, 2.0f, 5.5f});
    return s;
}

// ── Scene 08: gyroid via a custom expression ───────────────────────────────
// A gyroid is a triply-periodic minimal surface; its implicit form is a
// plain trig expression, so it shows off the CustomExprNode path. Intersected
// with a sphere so it reads as a solid blob rather than tiling to infinity.

static SceneGraph scene_gyroid() {
    SceneGraph s;
    Material mp{{0.5f, 0.5f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Gyroid surface (scaled by 2 for a tighter period), thickened to a slab
    // by |f| - t, then clipped to a sphere so it forms a bounded chunk.
    auto gyroid = std::make_shared<CustomExprNode>(
        "abs( sin(2*x)*cos(2*y) + sin(2*y)*cos(2*z) + sin(2*z)*cos(2*x) ) - 0.4",
        "gyroid");
    auto ball = std::make_shared<SphereNode>(1.6f, "clip");
    auto chunk = std::make_shared<IntersectionNode>(gyroid, ball, "gx");

    Material m{{0.45f, 0.15f, 0.5f}};   // purple, matching the showcase
    m.roughness = 0.4f;
    s.add_object(std::make_shared<TranslateNode>(chunk, 0, 0.2f, 0, "gt"), m);
    setup_lighting(s);
    return s;
}

// ── Scene 09: capsule primitive ────────────────────────────────────────────
// The capsule SDF — distance to a vertical segment, minus a radius — written
// as a custom expression: sqrt(x^2 + max(0,|y|-h)^2 + z^2) - r. (The same
// shape the dynamic capsule plugin provides; expressed inline here so the
// gallery has no plugin-load dependency.)

static SceneGraph scene_capsule() {
    SceneGraph s;
    Material mp{{0.5f, 0.5f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    auto cap = std::make_shared<CustomExprNode>(
        "sqrt( x*x + max(0, abs(y) - 0.8)*max(0, abs(y) - 0.8) + z*z ) - 0.45",
        "capsule");

    Material m{{0.6f, 0.18f, 0.14f}};   // deep red
    m.roughness = 0.5f;
    s.add_object(std::make_shared<TranslateNode>(cap, 0, 0.0f, 0, "ct"), m);
    setup_lighting(s);
    return s;
}

// ── Scene 10: textured scene (wood sphere + marble cube) ───────────────────
// Two procedural textures sampled through the Texture material pattern: a
// wood-grain ring pattern on a sphere and a veined marble on a cube.

static SceneGraph scene_textured() {
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);

    // Wood: concentric rings with a warm brown gradient.
    io::Image wood;
    wood.width = 128; wood.height = 128;
    wood.rgba.assign(128 * 128 * 4, 0);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
            std::size_t i = (y * 128 + x) * 4;
            float dx = (x - 64) / 64.0f, dy = (y - 64) / 64.0f;
            float r = std::sqrt(dx*dx + dy*dy);
            float ring = std::sin(r * 28.0f) * 0.5f + 0.5f;
            wood.rgba[i+0] = (std::uint8_t)(150 + 80 * ring);
            wood.rgba[i+1] = (std::uint8_t)(90  + 50 * ring);
            wood.rgba[i+2] = (std::uint8_t)(40  + 25 * ring);
            wood.rgba[i+3] = 255;
        }
    Material mw;
    mw.pattern = Material::Pattern::Texture;
    mw.texture_rgba = wood.rgba; mw.texture_width = 128; mw.texture_height = 128;
    mw.pattern_scale = 1.0f;
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "ws"), -1.4f, 0, 0, "wst"), mw);

    // Marble: grey base with sinusoidal veins broken up by a little noise.
    io::Image marble;
    marble.width = 128; marble.height = 128;
    marble.rgba.assign(128 * 128 * 4, 0);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
            std::size_t i = (y * 128 + x) * 4;
            float fx = x / 128.0f, fy = y / 128.0f;
            float n = std::sin((fx + fy) * 18.0f + std::sin(fy * 30.0f) * 2.0f);
            float v = 0.7f + 0.25f * n;
            std::uint8_t g = (std::uint8_t)(std::min(255.0f, std::max(0.0f, v * 230.0f)));
            marble.rgba[i+0] = g; marble.rgba[i+1] = g;
            marble.rgba[i+2] = (std::uint8_t)std::min(255, g + 8); marble.rgba[i+3] = 255;
        }
    Material mm;
    mm.pattern = Material::Pattern::Texture;
    mm.texture_rgba = marble.rgba; mm.texture_width = 128; mm.texture_height = 128;
    mm.pattern_scale = 2.0f;
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.8f, 0.8f, 0.8f, "mb"), 1.4f, 0, 0, "mbt"), mm);

    setup_lighting(s);
    return s;
}

// ── Main ──────────────────────────────────────────────────────────────────

// 11 — Instancing + non-uniform scale + multi-axis rotation. One twisted shape
// is authored once and instanced across a row (all instances share its geometry,
// so editing the source would move them all and the emitted GLSL carries the
// shape once). Behind it: three non-uniformly scaled ellipsoids and a box tilted
// on all three axes, to exercise RotateX/Y/Z and per-axis Scale together.
static SceneGraph scene_instances() {
    SceneGraph s;
    Material mp; mp.albedo = {0.55f, 0.57f, 0.6f};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);
    const float pi = 3.14159265f;

    // The source shape: a twisted rounded box. Authored once with id "src".
    Material msrc; msrc.albedo = {0.85f, 0.35f, 0.30f};
    auto src_geom = std::make_shared<TwistYNode>(
        std::make_shared<BoxNode>(0.28f, 0.5f, 0.28f, "src_box"), 1.6f, "src");
    s.add_object(std::make_shared<TranslateNode>(src_geom, -2.6f, 0.0f, 0.0f, "src_t"), msrc);

    // Five instances of "src" marching to the right — same geometry, own places.
    Material minst; minst.albedo = {0.30f, 0.55f, 0.85f};
    for (int i = 1; i <= 5; ++i) {
        auto inst = std::make_shared<InstanceNode>(src_geom, "src_t",
            "inst" + std::to_string(i));
        s.add_object(std::make_shared<TranslateNode>(
            inst, -2.6f + 1.05f * i, 0.0f, 0.0f, "inst_t" + std::to_string(i)), minst);
    }

    // Back row: three non-uniformly scaled spheres (ellipsoids).
    Material mell; mell.albedo = {0.9f, 0.75f, 0.25f};
    auto ell = [&](float sx, float sy, float sz, float x, const char* id) {
        auto e = std::make_shared<ScaleNode>(
            std::make_shared<SphereNode>(0.4f, std::string(id) + "s"), sx, sy, sz,
            std::string(id) + "sc");
        s.add_object(std::make_shared<TranslateNode>(e, x, 0.6f, -2.2f,
            std::string(id) + "t"), mell);
    };
    ell(2.0f, 0.7f, 0.7f, -2.2f, "e1");
    ell(0.7f, 2.0f, 0.7f,  0.0f, "e2");
    ell(0.7f, 0.7f, 2.0f,  2.2f, "e3");

    // A box tilted on all three axes (RotateX∘RotateY∘RotateZ).
    Material mrot; mrot.albedo = {0.55f, 0.8f, 0.55f};
    auto tilted = std::make_shared<RotateXNode>(
        std::make_shared<RotateYNode>(
            std::make_shared<RotateZNode>(
                std::make_shared<BoxNode>(0.5f, 0.5f, 0.5f, "tb"), pi/5, "tz"),
            pi/6, "ty"), pi/7, "tx");
    s.add_object(std::make_shared<TranslateNode>(tilted, 2.6f, 0.65f, 0.0f, "tt"), mrot);

    // resolve instance references so they render.
    io::resolve_instances(s, nullptr);
    return s;
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::string out_dir = (argc >= 2) ? argv[1] : "gallery";
    fs::create_directories(out_dir);

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    bool gpu_ok = gpu::VulkanCtx::available();
    if (!gpu_ok)
        std::fprintf(stderr,
            "[info] Vulkan unavailable — rendering the gallery on the CPU_IR\n"
            "       path instead. Since CPU_IR and GPU_GLSL are visually\n"
            "       equivalent (mean |Δ| ~0.0008), the images match what the\n"
            "       GPU path produces; on real hardware the GPU path is used.\n");

    // Render a scene with the GPU path if available, else the CPU path. Both
    // paths share the same corrected shading (light colour, sky, specular,
    // analytic normals), so the gallery looks the same either way.
    auto render = [&](const SceneGraph& s, const std::string& out,
                      int w = 800, int h = 500) {
        if (gpu_ok) render_gpu(s, out, w, h);
        else        render_cpu(s, out, w, h);
    };

    std::printf("\n01_csg\n");
    render(scene_csg(), out_dir + "/01_csg.ppm");

    std::printf("\n02_deformations\n");
    render(scene_deformations(), out_dir + "/02_deformations.ppm");

    std::printf("\n03_mesh_carve\n");
    render(scene_mesh_carve(), out_dir + "/03_mesh_carve.ppm");

    std::printf("\n04_patterns\n");
    render(scene_patterns(), out_dir + "/04_patterns.ppm", 1000, 400);

    std::printf("\n05_cpu_vs_gpu\n");
    if (gpu_ok) render_gpu(scene_compare(), out_dir + "/05_gpu.ppm");
    render_cpu(scene_compare(), out_dir + "/05_cpu.ppm");

    // Texture scenes need the GPU path — the CPU JIT can't sample image
    // textures (a documented GPU-only material feature), so on CPU they would
    // render untextured. Render them only when the GPU is present.
    auto render_textured = [&](const SceneGraph& s, const std::string& out,
                               int w = 800, int h = 500) {
        if (gpu_ok) render_gpu(s, out, w, h);
        else std::fprintf(stderr,
            "  [skip] %s needs the GPU path (textures are GPU-only); "
            "kept the committed image.\n", out.c_str());
    };

    std::printf("\n06_texture\n");
    render_textured(scene_texture(), out_dir + "/06_texture.ppm");

    std::printf("\n07_hero\n");
    render(scene_hero(), out_dir + "/07_hero.ppm", 1024, 640);

    std::printf("\n08_gyroid_customexpr\n");
    render(scene_gyroid(), out_dir + "/08_gyroid_customexpr.ppm", 600, 400);

    std::printf("\n09_plugin_capsule\n");
    render(scene_capsule(), out_dir + "/09_plugin_capsule.ppm", 400, 300);

    std::printf("\n10_textured_scene\n");
    render_textured(scene_textured(), out_dir + "/10_textured_scene.ppm", 600, 400);

    std::printf("\n11_instances\n");
    render(scene_instances(), out_dir + "/11_instances.ppm", 1000, 420);

    std::printf("\nGallery written to: %s\n", out_dir.c_str());
    return 0;
}
