# FRep Designer 4.0 — Plugin Authoring Guide

This guide walks through writing a custom F-Rep primitive as a
dynamically loaded plugin. Plugins extend the modeling system with
new SDF nodes that integrate seamlessly with the GUI, the CPU JIT,
the GPU compute shader path, and the marching-cubes mesh exporter —
without rebuilding the host application.

A complete reference implementation lives at
[`plugins/dynamic/capsule_plugin.cpp`](../plugins/dynamic/capsule_plugin.cpp).

---

## What is a plugin?

A plugin is a shared library (`.so` on Linux, `.dll` on Windows)
that, at load time, registers one or more `FRepNode` subclasses with
the host's `PluginRegistry`. Each registered node behaves as a
first-class primitive: it appears in the node graph's "Add node"
menu, can be combined via CSG with built-ins, exports to STL/OBJ,
and renders on both CPU and GPU.

The plugin contract is small but covers four execution paths:

| Path | Method to implement | When called |
|---|---|---|
| CPU evaluation | `float eval(x, y, z)` | Picker, marching cubes, BVH |
| LLVM IR codegen | `llvm::Value* codegen(...)` | CPU JIT pipeline |
| GLSL emission | `bool emit_glsl(...)` | GPU compute shader |
| Hashing | `std::size_t structural_hash()` | Incremental recompile cache |

Skipping any of these works but disables the corresponding pipeline
for scenes containing the plugin node.

---

## Minimum viable plugin

A plugin source file has three parts: a node class, a plugin
descriptor, and a C-ABI entry point.

### 1. Node class

Inherit from `FRepNode` and set `kind = NodeKind::Plugin` so the
GLSL emitter routes through your `emit_glsl()` override:

```cpp
#include "core/frep/node.hpp"
#include "core/plugin/plugin_api.hpp"
#include "core/compiler/llvm_compat.hpp"
#include <llvm/IR/Intrinsics.h>

namespace frep {

class TorusNode final : public FRepNode {
public:
    TorusNode(float R, float r, std::string nid = "torus") {
        kind = NodeKind::Plugin;
        id   = std::move(nid);
        params["R"] = R;  // major radius
        params["r"] = r;  // minor radius
    }

    const char* type_name() const noexcept override { return "Torus"; }

    // ── CPU eval (required for picker, marching cubes, BVH) ──
    float eval(float x, float y, float z) const override {
        float R = params.at("R"), r = params.at("r");
        float xz = std::sqrt(x*x + z*z) - R;
        return std::sqrt(xz*xz + y*y) - r;
    }

    // ── LLVM IR codegen (required for CPU JIT) ──
    llvm::Value* codegen(CgCtx& c, llvm::Value* x,
                         llvm::Value* y, llvm::Value* z) const override {
        auto& b = c.b;
        float R = params.at("R"), r = params.at("r");
        auto x2 = b.CreateFMul(x, x);
        auto z2 = b.CreateFMul(z, z);
        auto sum_xz = b.CreateFAdd(x2, z2);
        auto len_xz = frep::llvm_compat::unary_intrinsic(
            b, llvm::Intrinsic::sqrt, sum_xz);
        auto xz_off = b.CreateFSub(len_xz, c.fc(R));
        auto xz_off_sq = b.CreateFMul(xz_off, xz_off);
        auto y_sq = b.CreateFMul(y, y);
        auto sum = b.CreateFAdd(xz_off_sq, y_sq);
        auto outer = frep::llvm_compat::unary_intrinsic(
            b, llvm::Intrinsic::sqrt, sum);
        return b.CreateFSub(outer, c.fc(r), "torus");
    }

    // ── GLSL emission (required for GPU compute) ──
    bool emit_glsl(std::ostream& out,
                   const std::vector<std::string>& /*child_exprs*/,
                   const std::string& /*var_prefix*/) const override {
        float R = params.at("R"), r = params.at("r");
        out << "(length(vec2(length(vec2(x, z)) - " << R
            << ", y)) - " << r << ")";
        return true;
    }

    // ── Hashing (required for incremental recompile cache) ──
    std::size_t structural_hash() const noexcept override {
        return std::hash<float>{}(params.at("R")) * 31
             ^ std::hash<float>{}(params.at("r"))
             ^ 0xDADAull;
    }
};

} // namespace frep
```

### 2. Plugin descriptor

A small struct that tells the host what type name your node has, its
default parameters, and how to construct instances:

```cpp
struct TorusPlugin {
    std::string_view type_name() const { return "Torus"; }
    std::span<const float> param_defaults() const {
        static const float defaults[] = {1.0f, 0.25f};  // R, r
        return defaults;
    }
    frep::FRepNode::Ptr create(std::span<const float> params,
                                std::string id) const {
        return std::make_shared<frep::TorusNode>(
            params.size() >= 1 ? params[0] : 1.0f,
            params.size() >= 2 ? params[1] : 0.25f,
            std::move(id));
    }
};
```

The host inspects `param_defaults()` to discover parameter count and
build the GUI's parameter list. The actual parameter names (`R`,
`r`) come from the node's `params` map.

### 3. C-ABI entry point

The host's `PluginLoader::load_directory()` walks the target folder,
`dlopen`s each `.so`, and calls a known symbol — typically named
`frep_register_plugin` — to do the registration:

```cpp
extern "C" {
    // Required by frep::plugin::LoadedPlugin::load_directory.
    // Signature is fixed; do not change it.
    void frep_register_plugin(frep::plugin::PluginRegistry& reg) {
        reg.register_primitive(TorusPlugin{});
    }
}
```

This is the **only** symbol the host looks up by name. The C linkage
prevents C++ name mangling from breaking the lookup.

---

## Building

Plugins are SHARED libraries with `frep_core` as a private
dependency. Add the following to your build system:

### CMake (recommended)

```cmake
add_library(torus_plugin SHARED
    plugins/dynamic/torus_plugin.cpp)
target_link_libraries(torus_plugin
    PRIVATE
        frep_core)
target_compile_options(torus_plugin PRIVATE -fno-rtti -fPIC)
set_target_properties(torus_plugin PROPERTIES
    PREFIX ""                                   # no "lib" prefix
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins")
```

The host project already adds `ENABLE_EXPORTS ON` and
`-Wl,--whole-archive frep_core` to its main executable, so the
plugin sees the host's copy of `FRepNode::codegen_grad`,
`FRepNode::eval` (defaults), etc. without symbol duplication.

### Manual

```bash
clang++-20 -std=c++26 -fno-rtti -fPIC -shared \
    -I/path/to/frep4 \
    -I/usr/lib/llvm-20/include \
    plugins/dynamic/torus_plugin.cpp \
    -o build/plugins/torus_plugin.so
```

---

## Discovery

The GUI searches four directories for plugins, in order:

1. `./plugins/` (relative to the current working directory)
2. `./build/plugins/` (typical CMake layout)
3. `<exe_dir>/plugins/` (alongside the executable)
4. `<exe_dir>/../lib/frep/plugins/` (Unix install layout)

The first directory containing at least one `.so` wins. Once loaded,
your new node appears in the node graph's "Add node" context menu
under its `type_name()`.

For command-line use:

```bash
./build/frep_designer --plugin-dir /path/to/plugins
```

---

## Testing your plugin

The project ships a test harness — `tests/test_plugin_runtime.cpp` —
that exercises the full dlopen path. Modeled after that file, a
typical plugin test does:

```cpp
TEST(MyPlugin, LoadsAndEvalsCorrectly) {
    auto& reg = frep::plugin::PluginRegistry::instance();
    auto loaded = frep::plugin::LoadedPlugin::load_directory(
        "build/plugins", reg);
    auto* slot = reg.find_primitive_by_type_name("Torus");
    ASSERT_NE(slot, nullptr);

    auto node = slot->create(slot->param_defaults(), "t");
    node->params["R"] = 1.0f;
    node->params["r"] = 0.25f;

    // On the surface at (R, 0, 0):
    EXPECT_NEAR(node->eval(1.0f, 0, 0), -0.25f, 1e-4);
    // Far outside:
    EXPECT_GT(node->eval(10, 0, 0), 5.0f);
}
```

For the GPU path, additionally check that `emit_glsl()` produces a
shader that parses with `glslangValidator` and runs without
producing all-black output.

---

## Common pitfalls

- **Forgetting `kind = NodeKind::Plugin`**: the GLSL emitter looks
  up the kind via switch and falls through to `emit_glsl()` only
  for `Plugin`. Without setting this, your node renders as nothing
  on GPU.

- **Missing `eval()` override**: the picker, marching cubes, and BVH
  voxelizer all call `eval()`. Without an override, they throw at
  the first sample. (The compiler will not warn about this — the
  base class has a defaulted `eval()` that throws at runtime.)

- **Hash collisions across runs**: include a unique magic number in
  your `structural_hash()` (we use `0xCAFE'D00D` in the Capsule
  example). Otherwise two different plugin nodes with the same
  parameter values hash equal, causing the incremental recompiler
  to reuse the wrong IR.

- **C++ name mangling on the entry point**: the `extern "C"` is
  mandatory. Without it the host can't find the symbol.

- **Linking `LLVM` directly**: don't. The host already links LLVM;
  your plugin gets it transitively via `frep_core`. Adding
  `-lLLVM-20` to your own link line causes duplicate symbols.

- **Loading plugins from untrusted sources**: a `.so` is arbitrary
  code. The host runs it in-process with full permissions. Only
  load plugins you trust or have audited.

---

## More complex plugin shapes

### Plugin with children (operators)

The plugin contract is the same — your `FRepNode` subclass's
`children` vector holds child nodes, and your `codegen()` /
`emit_glsl()` call into them recursively. The host's GLSL emitter
will have already filled `child_exprs` with the GLSL strings for
each child by the time `emit_glsl()` is called on you, so you can
just splice them in.

### Plugin with non-numeric parameters

The `param_defaults()` API uses `std::span<const float>` so it only
covers numeric values, but `FRepNode::params` is a
`std::unordered_map<std::string, float>`. For things like enums,
encode them as floats (e.g. `mode = 0` / `mode = 1`). Or, if your
plugin needs richer state (textures, lookup tables), allocate it in
the node's ctor and store it in member fields — the host doesn't
need to know.

### GPU buffers from a plugin

If your plugin needs auxiliary data on the GPU (a noise texture,
a precomputed table), implement `gpu_resources()` returning a
description; the host's GLSL emitter will bind the buffer slot for
you. See `core/frep/mesh_sdf.hpp` for an example (the built-in
MeshSDF node uses this mechanism for the voxel SDF grid).

---

## Where to read next

- [`plugins/dynamic/capsule_plugin.cpp`](../plugins/dynamic/capsule_plugin.cpp) —
  the reference implementation, fully commented
- [`core/plugin/plugin_api.hpp`](../core/plugin/plugin_api.hpp) —
  the registry interface
- [`tests/test_plugin_runtime.cpp`](../tests/test_plugin_runtime.cpp) —
  the test harness pattern
- [ARCHITECTURE.md](ARCHITECTURE.md) — how plugins integrate with
  the three execution back-ends
