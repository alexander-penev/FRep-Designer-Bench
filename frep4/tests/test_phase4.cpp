// tests/test_phase4.cpp
//
// Tests for Phase 4 — Google Test.
// Covers the CustomExprCompiler parser and the Plugin API concepts.

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/frep/custom_expr.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/scene.hpp"
#include "core/io/scene_io.hpp"
#include "core/plugin/plugin_api.hpp"
#include "plugins/extra_primitives.hpp"
#include "tests/test_support.hpp"

#include <gtest/gtest.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>

#include <array>
#include <memory>
#include <vector>

using namespace frep;

namespace {

using test::SdfFn;
using test::jit_pool;

// Compiles a CustomExprCompiler expression directly.
SdfFn jit_custom_expr(const std::string& expr) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("test_expr", *ctx);

    CustomExprCompiler comp;
    auto* fn = comp.compile(*mod, *ctx, "render_tile", expr);
    if (!fn) return nullptr;
    fn->setLinkage(llvm::Function::ExternalLinkage);

    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto fn_or = jit_pool().back()->load(std::move(mod), std::move(ctx));
    if (!fn_or) return nullptr;
    return reinterpret_cast<SdfFn>(*fn_or);
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// CustomExpr parser
// ═════════════════════════════════════════════════════════════════════════════
TEST(CustomExpr, BasicArithmetic) {
    auto fn = jit_custom_expr("x + y + z");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(1.0f, 2.0f, 3.0f), 6.0f, 1e-5f);
    EXPECT_NEAR(fn(-1.0f, 0.5f, 1.5f), 1.0f, 1e-5f);
}

TEST(CustomExpr, OperatorPrecedence) {
    // x + y*z  ->  2 + 3*4 = 14, not (2+3)*4 = 20
    auto fn = jit_custom_expr("x + y*z");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(2.0f, 3.0f, 4.0f), 14.0f, 1e-5f);
}

TEST(CustomExpr, ParenthesesOverridePrecedence) {
    auto fn = jit_custom_expr("(x + y) * z");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(2.0f, 3.0f, 4.0f), 20.0f, 1e-5f);
}

TEST(CustomExpr, UnaryMinus) {
    auto fn = jit_custom_expr("-x - y");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(2.0f, 3.0f, 0.0f), -5.0f, 1e-5f);
}

TEST(CustomExpr, BuiltinSqrt) {
    auto fn = jit_custom_expr("sqrt(x*x + y*y + z*z)");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(3.0f, 4.0f, 0.0f), 5.0f, 1e-4f);
    EXPECT_NEAR(fn(0.0f, 0.0f, 0.0f), 0.0f, 1e-4f);
}

TEST(CustomExpr, BuiltinAbs) {
    auto fn = jit_custom_expr("abs(x) + abs(y) + abs(z)");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(-1.0f, 2.0f, -3.0f), 6.0f, 1e-5f);
}

TEST(CustomExpr, BuiltinMinMax) {
    auto fn = jit_custom_expr("max(min(x,y), z)");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(5.0f, 3.0f, 2.0f), 3.0f, 1e-5f);
    EXPECT_NEAR(fn(5.0f, 3.0f, 4.0f), 4.0f, 1e-5f);
}

TEST(CustomExpr, ConstantPi) {
    auto fn = jit_custom_expr("pi");
    ASSERT_NE(fn, nullptr);
    EXPECT_NEAR(fn(0, 0, 0), 3.14159265f, 1e-5f);
}

TEST(CustomExpr, UnknownFunctionErrors) {
    CustomExprCompiler comp;
    llvm::LLVMContext ctx;
    llvm::Module mod("test", ctx);
    auto* fn = comp.compile(mod, ctx, "test", "unknown_fn(x)");
    EXPECT_EQ(fn, nullptr);
    EXPECT_FALSE(comp.last_error().empty());
}

TEST(CustomExpr, UnclosedParenErrors) {
    CustomExprCompiler comp;
    llvm::LLVMContext ctx;
    llvm::Module mod("test", ctx);
    auto* fn = comp.compile(mod, ctx, "test", "(x + y");
    EXPECT_EQ(fn, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Plugin API
// ═════════════════════════════════════════════════════════════════════════════
TEST(PluginApi, TorusPluginConceptSatisfied) {
    static_assert(plugin::PrimitivePlugin<TorusPlugin>,
                  "TorusPlugin must satisfy PrimitivePlugin");
    TorusPlugin p;
    EXPECT_EQ(p.info().name, "torus");
    EXPECT_EQ(p.param_names().size(), 2u);
    EXPECT_EQ(p.param_defaults().size(), 2u);

    std::array<float, 2> vals = {1.5f, 0.4f};
    auto node = p.create(vals, "t1");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->id, "t1");
}

TEST(PluginApi, OctahedronPluginConceptSatisfied) {
    static_assert(plugin::PrimitivePlugin<OctahedronPlugin>);
    OctahedronPlugin p;
    EXPECT_EQ(p.info().name, "octahedron");
    EXPECT_EQ(p.param_names().size(), 1u);
}

TEST(PluginApi, RegistryEnumeratesPlugins) {
    plugin::PluginRegistry reg;
    reg.register_primitive(TorusPlugin{});
    reg.register_primitive(OctahedronPlugin{});

    EXPECT_EQ(reg.primitives().size(), 2u);

    bool found_torus = false, found_oct = false;
    for (const auto& slot : reg.primitives()) {
        if (slot.info.name == "torus") found_torus = true;
        if (slot.info.name == "octahedron") found_oct = true;
    }
    EXPECT_TRUE(found_torus);
    EXPECT_TRUE(found_oct);
}

TEST(CustomExprNode, ProducesWorkingSdf) {
    auto node = std::make_shared<CustomExprNode>(
        "sqrt(x*x + y*y + z*z) - 1.0", "us");

    auto ctx = std::make_unique<llvm::LLVMContext>();
    SceneCodegen cg(*ctx);
    auto* fn = cg.emit_scene_sdf(*node);
    ASSERT_NE(fn, nullptr);
    fn->setName("render_tile");

    auto mod = cg.take_module();
    jit_pool().emplace_back(std::make_unique<JitEngine>());
    auto fn_or = jit_pool().back()->load(std::move(mod), std::move(ctx));
    ASSERT_TRUE(fn_or.has_value());
    auto sdf = reinterpret_cast<SdfFn>(*fn_or);

    EXPECT_NEAR(sdf(0, 0, 0), -1.0f, 1e-4f);  // center
    EXPECT_NEAR(sdf(1, 0, 0),  0.0f, 1e-4f);  // surface r=1
    EXPECT_NEAR(sdf(2, 0, 0),  1.0f, 1e-4f);  // outside
}

// ─── Plugin deserialization round-trip ──────────────────────────────────────

// Helper: a private registry to avoid clobbering the singleton across tests.
static plugin::PluginRegistry make_test_registry() {
    plugin::PluginRegistry r;
    r.register_primitive(TorusPlugin{});
    r.register_primitive(OctahedronPlugin{});
    return r;
}

TEST(SceneIOPlugin, RegistryRecordsNodeTypeName) {
    auto reg = make_test_registry();
    // The registry should have remembered TorusNode::type_name() = "Torus"
    // and OctahedronNode::type_name() = "Octahedron".
    const auto* torus = reg.find_primitive_by_type_name("Torus");
    const auto* octa  = reg.find_primitive_by_type_name("Octahedron");
    ASSERT_NE(torus, nullptr);
    ASSERT_NE(octa,  nullptr);
    EXPECT_EQ(std::string(torus->info.name), "torus");
    EXPECT_EQ(std::string(octa->info.name),  "octahedron");
    // Unknown type → nullptr.
    EXPECT_EQ(reg.find_primitive_by_type_name("NoSuchType"), nullptr);
}

TEST(SceneIOPlugin, TorusRoundTrip) {
    auto reg = make_test_registry();

    // Build a scene with a plugin-based Torus primitive.
    SceneGraph scene;
    std::array<float, 2> params = {1.7f, 0.35f};
    auto torus = reg.find_primitive_by_type_name("Torus")->create(
        params, "torus1");
    scene.add_object(torus, Material{{0.8f, 0.2f, 0.4f}});

    // Serialize.
    auto json_text = io::serialize_scene(scene);
    // Should mention "Torus" as the type field.
    EXPECT_NE(json_text.find("\"Torus\""), std::string::npos);
    EXPECT_NE(json_text.find("\"R\""),     std::string::npos);
    EXPECT_NE(json_text.find("\"r\""),     std::string::npos);

    // Deserialize WITHOUT a registry → must throw with a clear message.
    EXPECT_THROW({
        io::deserialize_scene(json_text, /*registry=*/nullptr);
    }, std::runtime_error);

    // Deserialize WITH the registry → round-trips correctly.
    auto loaded = io::deserialize_scene(json_text, &reg);
    ASSERT_EQ(loaded.objects().size(), 1u);
    ASSERT_TRUE(loaded.objects().count("torus1"));
    const auto& obj = loaded.objects().at("torus1");
    EXPECT_EQ(std::string(obj.geometry->type_name()), "Torus");
    EXPECT_FLOAT_EQ(obj.geometry->params.at("R"), 1.7f);
    EXPECT_FLOAT_EQ(obj.geometry->params.at("r"), 0.35f);
    EXPECT_FLOAT_EQ(obj.material.albedo[0], 0.8f);
}

TEST(SceneIOPlugin, OctahedronRoundTrip) {
    auto reg = make_test_registry();

    SceneGraph scene;
    std::array<float, 1> params = {1.25f};
    auto octa = reg.find_primitive_by_type_name("Octahedron")->create(
        params, "oct1");
    scene.add_object(octa, Material{{0.1f, 0.7f, 0.3f}});

    auto loaded = io::deserialize_scene(io::serialize_scene(scene), &reg);
    ASSERT_TRUE(loaded.objects().count("oct1"));
    const auto& obj = loaded.objects().at("oct1");
    EXPECT_EQ(std::string(obj.geometry->type_name()), "Octahedron");
    EXPECT_FLOAT_EQ(obj.geometry->params.at("size"), 1.25f);
}

TEST(SceneIOPlugin, MixedSceneRoundTrip) {
    // Mix built-in and plugin-based primitives in one scene.
    auto reg = make_test_registry();

    SceneGraph scene;
    scene.add_object(std::make_shared<SphereNode>(1.0f, "ball"),
                     Material{{0.9f, 0.9f, 0.9f}});

    std::array<float, 2> tp = {1.2f, 0.3f};
    auto torus = reg.find_primitive_by_type_name("Torus")->create(tp, "ring");
    scene.add_object(torus, Material{{0.2f, 0.4f, 0.9f}});

    auto loaded = io::deserialize_scene(io::serialize_scene(scene), &reg);
    ASSERT_EQ(loaded.objects().size(), 2u);
    EXPECT_EQ(std::string(loaded.objects().at("ball").geometry->type_name()),
              "Sphere");
    EXPECT_EQ(std::string(loaded.objects().at("ring").geometry->type_name()),
              "Torus");
}

TEST(SceneIOPlugin, UnknownTypeStillThrowsWithRegistry) {
    auto reg = make_test_registry();
    // Hand-craft JSON with a node type that isn't built-in and isn't in
    // the registry.
    std::string json_text = R"({
        "version": "4.0",
        "objects": [{
            "id": "x",
            "geometry": { "type": "Mandelbulb", "params": {} }
        }]
    })";
    EXPECT_THROW(io::deserialize_scene(json_text, &reg), std::runtime_error);
}

// ─── Lights serialization round-trip ────────────────────────────────────────

TEST(SceneIOLights, RoundTripsMultipleLights) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"),
                 Material{{0.5f, 0.7f, 0.9f}});

    // Replace the default light list with 3 distinctive lights.
    auto& Ls = s.lights();
    Ls.clear();
    Ls.push_back(PointLight{{ 1.5f, 6.0f,  3.0f}, {1.0f, 0.5f, 0.25f}, 0.8f});
    Ls.push_back(PointLight{{-4.0f, 4.0f,  2.0f}, {0.4f, 0.5f, 1.0f},  0.6f});
    Ls.push_back(PointLight{{ 0.0f, 8.0f, -3.0f}, {0.9f, 1.0f, 0.7f},  0.5f});

    auto js = io::serialize_scene(s);
    // Sanity: lights array should appear in the JSON.
    EXPECT_NE(js.find("\"lights\""), std::string::npos);
    EXPECT_NE(js.find("\"intensity\""), std::string::npos);

    auto loaded = io::deserialize_scene(js);
    ASSERT_EQ(loaded.lights().size(), 3u);

    auto near = [](float a, float b){ return std::abs(a - b) < 1e-5f; };
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& a = Ls[i];
        const auto& b = loaded.lights()[i];
        EXPECT_TRUE(near(a.pos[0], b.pos[0]));
        EXPECT_TRUE(near(a.pos[1], b.pos[1]));
        EXPECT_TRUE(near(a.pos[2], b.pos[2]));
        EXPECT_TRUE(near(a.color[0], b.color[0]));
        EXPECT_TRUE(near(a.color[1], b.color[1]));
        EXPECT_TRUE(near(a.color[2], b.color[2]));
        EXPECT_TRUE(near(a.intensity, b.intensity));
    }
}

TEST(SceneIOLights, EmptyLightArraySurvives) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    s.lights().clear();  // an empty list is a valid choice the user can make
    auto js = io::serialize_scene(s);
    auto loaded = io::deserialize_scene(js);
    EXPECT_TRUE(loaded.lights().empty());
}

// ─── Camera projection serialization round-trip ─────────────────────────────

TEST(SceneIOCamera, ProjectionRoundTrip) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));

    // Default is perspective; flip to orthographic with a custom ortho size.
    s.camera().projection = Camera::Projection::Orthographic;
    s.camera().ortho_size = 5.5f;
    s.camera().fov_deg    = 42.0f;     // still serialized

    auto js = io::serialize_scene(s);
    EXPECT_NE(js.find("\"projection\""), std::string::npos);
    EXPECT_NE(js.find("\"orthographic\""), std::string::npos);
    EXPECT_NE(js.find("\"ortho_size\""), std::string::npos);

    auto loaded = io::deserialize_scene(js);
    EXPECT_EQ(loaded.camera().projection, Camera::Projection::Orthographic);
    EXPECT_FLOAT_EQ(loaded.camera().ortho_size, 5.5f);
    EXPECT_FLOAT_EQ(loaded.camera().fov_deg,    42.0f);
}

TEST(SceneIOCamera, DefaultsToPersepectiveWhenAbsent) {
    // Old scene files written before projection mode existed should still
    // load as perspective.
    std::string js = R"({
        "version": "4.0",
        "camera": {
            "position": [0, 0, 5],
            "target":   [0, 0, 0],
            "up":       [0, 1, 0],
            "fov_deg":  50.0
        },
        "objects": []
    })";
    auto loaded = io::deserialize_scene(js);
    EXPECT_EQ(loaded.camera().projection, Camera::Projection::Perspective);
}
