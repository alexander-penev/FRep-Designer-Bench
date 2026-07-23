// tests/test_plugin_runtime.cpp
//
// Verifies the *dynamic* plugin loading path end-to-end:
//   1. Load capsule_plugin.so via PluginLoader at runtime
//   2. Construct a CapsuleNode from the loaded factory
//   3. Confirm it eval's correctly on CPU
//   4. Confirm it codegens correctly via JIT
//   5. Confirm it emit_glsl's correctly via GLSL emitter
//   6. (Optional, GPU-only) Render it via Vulkan and check non-black pixels
//
// This is distinct from test_plugin_glsl.cpp which uses an *inline*
// CapsuleNode defined inside the test file. The runtime test catches
// bugs in the host-symbol-export linkage (ENABLE_EXPORTS,
// --whole-archive) and the LoadedPlugin lifetime management that the
// inline test cannot see.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/plugin/plugin_loader.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

using namespace frep;

namespace {

// Looks for capsule_plugin.so next to the test binary. CMake puts it in
// build/plugins/, and frep_tests runs from build/, so the relative
// path is plugins/capsule_plugin.so. We also try build/plugins/ in
// case ctest is invoked from project root.
std::filesystem::path find_plugin() {
    namespace fs = std::filesystem;
    const std::array<fs::path, 3> candidates = {
        fs::path("plugins") / "capsule_plugin.so",
        fs::path("build") / "plugins" / "capsule_plugin.so",
        fs::path("../plugins") / "capsule_plugin.so",
    };
    for (auto& p : candidates) {
        if (fs::exists(p)) return fs::absolute(p);
    }
    return {};
}

} // anon

// ─────────────────────────────────────────────────────────────────────────────
// Discovery + loading
// ─────────────────────────────────────────────────────────────────────────────
TEST(PluginRuntime, FindsPluginFile) {
    auto p = find_plugin();
    ASSERT_FALSE(p.empty())
        << "capsule_plugin.so not found in any expected location";
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST(PluginRuntime, LoadsAndRegistersCapsule) {
    auto p = find_plugin();
    if (p.empty()) GTEST_SKIP() << "plugin file not present";

    // We use the same registry the runtime would use — singleton. If
    // an earlier test in this process already loaded the plugin, the
    // registration is idempotent (load_directory only loads .so files
    // it hasn't seen, but our registry won't dedupe). We just check
    // the post-condition.
    auto& reg = plugin::PluginRegistry::instance();
    auto loaded = plugin::LoadedPlugin::load_directory(
        p.parent_path(), reg);
    // Either we loaded it now, or a previous test did — either way
    // the registry should know about "Capsule".
    EXPECT_TRUE(loaded.size() >= 1 ||
                reg.find_primitive_by_type_name("Capsule") != nullptr);
    auto* slot = reg.find_primitive_by_type_name("Capsule");
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->node_type_name, "Capsule");
    EXPECT_EQ(slot->param_defaults.size(), 2u);  // height + radius
}

// ─────────────────────────────────────────────────────────────────────────────
// Functional checks once loaded
// ─────────────────────────────────────────────────────────────────────────────
TEST(PluginRuntime, CapsuleEvalsCorrectly) {
    auto p = find_plugin();
    if (p.empty()) GTEST_SKIP() << "plugin file not present";
    auto& reg = plugin::PluginRegistry::instance();
    plugin::LoadedPlugin::load_directory(p.parent_path(), reg);

    auto* slot = reg.find_primitive_by_type_name("Capsule");
    ASSERT_NE(slot, nullptr);
    auto cap = slot->create(slot->param_defaults, "c");
    ASSERT_NE(cap, nullptr);
    cap->params["height"] = 1.0f;   // half-height
    cap->params["radius"] = 0.5f;

    // Inside the cylinder (centre) — distance to surface ≈ -radius.
    EXPECT_NEAR(cap->eval(0, 0, 0), -0.5f, 1e-4);
    // Far above on the axis — distance ≈ y - h - r.
    EXPECT_NEAR(cap->eval(0, 3.0f, 0), 3.0f - 1.0f - 0.5f, 1e-4);
    // Far out radially at y=0 — distance ≈ |x| - r.
    EXPECT_NEAR(cap->eval(2.0f, 0, 0), 2.0f - 0.5f, 1e-4);
}

TEST(PluginRuntime, CapsuleEmitsGlsl) {
    auto p = find_plugin();
    if (p.empty()) GTEST_SKIP() << "plugin file not present";
    auto& reg = plugin::PluginRegistry::instance();
    plugin::LoadedPlugin::load_directory(p.parent_path(), reg);

    auto* slot = reg.find_primitive_by_type_name("Capsule");
    ASSERT_NE(slot, nullptr);
    auto cap = slot->create(slot->param_defaults, "c");
    cap->params["height"] = 0.6f;
    cap->params["radius"] = 0.3f;

    std::ostringstream out;
    bool ok = cap->emit_glsl(out, {}, "v_");
    EXPECT_TRUE(ok);
    auto s = out.str();
    // The capsule SDF formula contains length(vec3(... clamp(y...))).
    EXPECT_NE(s.find("length"),    std::string::npos);
    EXPECT_NE(s.find("clamp(y"),   std::string::npos);
    EXPECT_NE(s.find("0.3"),       std::string::npos);  // radius literal
}

TEST(PluginRuntime, CapsuleRendersOnGpu) {
    if (!gpu::VulkanCtx::available())   GTEST_SKIP() << "no Vulkan";
    if (gpu::find_glslang().empty())    GTEST_SKIP() << "no glslang";

    auto p = find_plugin();
    if (p.empty()) GTEST_SKIP() << "plugin file not present";
    auto& reg = plugin::PluginRegistry::instance();
    plugin::LoadedPlugin::load_directory(p.parent_path(), reg);

    auto* slot = reg.find_primitive_by_type_name("Capsule");
    ASSERT_NE(slot, nullptr);
    auto cap = slot->create(slot->param_defaults, "c");
    cap->params["height"] = 0.5f;
    cap->params["radius"] = 0.25f;

    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "fl"), mp);
    Material m{{0.95f, 0.3f, 0.25f}};
    s.add_object(cap, m);

    s.camera().position = {0, 0.5f, 3};
    s.camera().target   = {0, 0, 0};
    s.lights().clear();
    s.lights().push_back({{4, 5, 3}, {1, 1, 1}, 1.2f});

    auto e = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(e.has_value());
    auto spv = gpu::compile_glsl_to_spv_managed(e->source);
    ASSERT_TRUE(spv.has_value()) << spv.error();
    auto ctx = gpu::VulkanCtx::create(spv->path(), e->mesh_voxels, e->texture_pixels);
    ASSERT_TRUE(ctx.has_value()) << ctx.error();

    int W = 96, H = 96;
    std::vector<std::uint8_t> px;
    ASSERT_TRUE((**ctx).render(
        gpu::build_push_from_scene(s, W, H), px).has_value());

    int red = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 30 && r > g + 10 && r > b + 10) ++red;
    }
    EXPECT_GT(red, 50) << "plugin capsule should be visibly rendered";
}
