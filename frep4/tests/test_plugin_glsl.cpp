// tests/test_plugin_glsl.cpp
//
// Regression test for the plugin GLSL emission fallback. Verifies that:
//   1. A FRepNode subclass that returns true from emit_glsl() AND uses
//      NodeKind::Plugin gets routed through the GLSL emitter's default
//      case and produces valid GLSL.
//   2. The emitted shader survives glslangValidator and renders on Vulkan.
//
// Rather than loading the capsule plugin from disk (which depends on
// the build layout being right), this test defines an inline plugin
// node alongside the test cases so the test is self-contained.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/operations.hpp"
#include "core/frep/transforms.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <llvm/IR/Intrinsics.h>

#include <gtest/gtest.h>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

namespace {

// Test plugin node: a Y-axis capsule, same SDF as the dynamic plugin
// but defined here for testing. Returns NodeKind::Plugin so the GLSL
// emitter dispatches through the emit_glsl() virtual.
class TestCapsuleNode final : public frep::FRepNode {
public:
    TestCapsuleNode(float h, float r, std::string nid = "tcap") {
        kind = frep::NodeKind::Plugin;
        id   = std::move(nid);
        params["height"] = h;
        params["radius"] = r;
    }

    const char* type_name() const noexcept override { return "TestCapsule"; }

    float eval(float x, float y, float z) const override {
        float h = params.at("height");
        float r = params.at("radius");
        float y_off = y - std::clamp(y, -h, h);
        return std::sqrt(x*x + y_off*y_off + z*z) - r;
    }

    llvm::Value* codegen(frep::CgCtx& c, llvm::Value* x,
                         llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        float h = params.at("height");
        float r = params.at("radius");
        auto ny = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::maxnum, y, c.fc(-h));
        auto cy = frep::llvm_compat::binary_intrinsic(b, llvm::Intrinsic::minnum, ny, c.fc(h));
        auto y_off = b.CreateFSub(y, cy);
        auto x2 = b.CreateFMul(x, x);
        auto y2 = b.CreateFMul(y_off, y_off);
        auto z2 = b.CreateFMul(z, z);
        auto sum = b.CreateFAdd(b.CreateFAdd(x2, y2), z2);
        auto len = frep::llvm_compat::unary_intrinsic(b, llvm::Intrinsic::sqrt, sum);
        return b.CreateFSub(len, c.fc(r), "tcap");
    }

    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("height")) * 31
             ^ std::hash<float>{}(params.at("radius"))
             ^ 0xCAFE'BEEFull;
    }

    bool emit_glsl(std::ostream& out,
                   const std::vector<std::string>& /*child_exprs*/,
                   const std::string& /*var_prefix*/) const override {
        float h = params.at("height");
        float r = params.at("radius");
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.7g", h);
        std::string hs = buf;
        if (hs.find('.') == std::string::npos) hs += ".0";
        std::snprintf(buf, sizeof(buf), "%.7g", r);
        std::string rs = buf;
        if (rs.find('.') == std::string::npos) rs += ".0";
        out << "(length(vec3(x, y - clamp(y, -" << hs << ", " << hs
            << "), z)) - " << rs << ")";
        return true;
    }
};

bool glslang_available() { return !frep::gpu::find_glslang().empty(); }

} // anon

TEST(PluginGlsl, EmitsExpression) {
    using namespace frep;
    SceneGraph s;
    Material m{{0.9f, 0.4f, 0.3f}};
    s.add_object(std::make_shared<TestCapsuleNode>(0.6f, 0.3f, "cap"), m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << (r ? "" : r.error());
    EXPECT_EQ(r->object_count, 1);
    // The emit_glsl output should appear in the shader source.
    EXPECT_NE(r->source.find("length(vec3(x, y - clamp(y"), std::string::npos);
}

TEST(PluginGlsl, ComposesWithCSG) {
    using namespace frep;
    // Plugin capsule unioned with a built-in sphere — ensures the
    // dispatch correctly recurses through child nodes that mix
    // built-in and plugin types.
    SceneGraph s;
    Material m{{0.9f, 0.4f, 0.3f}};
    auto cap = std::make_shared<TestCapsuleNode>(0.5f, 0.2f, "cap");
    auto sph = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(0.3f, "sp"),
        0.5f, 0, 0, "spt");
    s.add_object(std::make_shared<UnionNode>(cap, sph, "u"), m);

    auto r = gpu::GlslEmitter::emit(s);
    ASSERT_TRUE(r.has_value()) << (r ? "" : r.error());
    // Both branches should appear: capsule expr + sphere expr.
    EXPECT_NE(r->source.find("length(vec3(x, y - clamp(y"), std::string::npos);
    EXPECT_NE(r->source.find("min("), std::string::npos);  // from Union
}

TEST(PluginGlsl, RendersOnGpu) {
    if (!glslang_available())          GTEST_SKIP() << "no glslang";
    if (!frep::gpu::VulkanCtx::available()) GTEST_SKIP() << "no Vulkan";

    using namespace frep;
    SceneGraph s;
    Material mp{{0.55f, 0.55f, 0.55f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "fl"), mp);
    Material m{{0.95f, 0.3f, 0.25f}};
    s.add_object(std::make_shared<TestCapsuleNode>(0.5f, 0.25f, "cap"), m);

    s.camera().position = {0, 0.5f, 3};
    s.camera().target   = {0, 0, 0};
    auto& L = s.lights(); L.clear();
    L.push_back({{4, 5, 3}, {1, 1, 1}, 1.2f});

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

    // Should have some red-dominant pixels (the capsule's albedo).
    int red_count = 0;
    for (int i = 0; i < W * H; ++i) {
        int r = px[i*4+0], g = px[i*4+1], b = px[i*4+2];
        if (r > 30 && r > g + 10 && r > b + 10) ++red_count;
    }
    EXPECT_GT(red_count, 50) << "expected capsule rendered visibly";
}
