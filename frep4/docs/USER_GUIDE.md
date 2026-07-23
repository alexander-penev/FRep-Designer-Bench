# FRep Designer 4.0 — User Guide

A complete user-facing guide to building, running, and extending FRep
Designer. For an architectural overview see the top-level `README.md`;
for performance numbers see `docs/BENCHMARKS.md`.

## Table of contents

1. [What FRep Designer is](#what-frep-designer-is)
2. [Building from source](#building-from-source)
3. [Running the GUI](#running-the-gui)
4. [Command-line tools](#command-line-tools)
5. [Scene file format](#scene-file-format)
6. [Writing a plugin](#writing-a-plugin)
7. [Performance tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

---

## What FRep Designer is

FRep Designer is a geometric modeling system that builds shapes from
*signed distance functions* (SDFs). Every primitive — sphere, box,
plane — is described by a function `f(x, y, z)` that returns the signed
distance to the surface; CSG operations and deformations combine and
transform these functions. The whole scene is compiled to native code
(via LLVM) or GLSL (for the GPU path) and ray-marched to produce
images.

What you get:

- **Procedural primitives + CSG** (Union, Intersection, Difference,
  smooth Union).
- **Transforms** (Translate, non-uniform Scale, RotateX/Y/Z) and non-linear
  **deformations** (Twist, Bend, Taper).
- **Mesh import** (OBJ/STL) — meshes are voxelised into SDF grids and
  participate in CSG as first-class objects.
- **Procedural materials** (Solid, Checker, Stripes, GradientY, Noise)
  plus **image textures** via triplanar projection.
- **Two rendering paths**: CPU JIT (LLVM, multithreaded, super-sampled,
  Cook-Torrance PBR) and GPU compute (Vulkan, GLSL emitted at runtime).
- **Live GUI** (Qt6) with orbit camera, scene inspector, node graph
  editor, undo/redo, and a checkable GPU mode toggle.
- **Plugin system** for adding new node types without rebuilding.

---

## Building from source

### Prerequisites

| Tool | Minimum version | Required for |
|---|---|---|
| C++ compiler | clang 20 / GCC 14 with C++26 | everything |
| CMake | 3.20 | everything |
| LLVM | 20 (also tested on 21+) | CPU JIT path |
| Qt6 | 6.4 (Core, Widgets) | GUI (optional) |
| GoogleTest | 1.14 | tests (optional) |
| Vulkan | 1.2 + glslangValidator | GPU compute path (optional) |
| llvm-spirv | matching LLVM major | SPIR-V emission (optional) |

On Ubuntu 24.04 the dependencies are installable in one line:

```bash
sudo apt install clang-20 cmake ninja-build \
    qt6-base-dev qt6-tools-dev libgtest-dev libgmock-dev \
    libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers \
    llvm-20-dev libllvm-20-ocaml-dev llvm-spirv-20
```

### Configure + build

```bash
git clone <repo> frep4
cd frep4
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_C_COMPILER=clang-20
cmake --build build -j$(nproc)
```

The configure step prints which optional features were enabled:

```
-- LLVM 20.1.2 from /usr/lib/llvm-20/lib/cmake/llvm
-- Vulkan 1.3.275 from /lib/x86_64-linux-gnu/libvulkan.so
-- Qt6 GUI: enabled (frep_designer)
```

### Run tests

```bash
cd build && ctest --output-on-failure
```

The full test suite should pass (300+ tests). If you see Vulkan-related tests reported as
SKIPPED, that's normal on systems without a Vulkan driver — the build
proceeds anyway.

### Build artefacts

| Binary | Purpose |
|---|---|
| `frep_designer` | Qt6 GUI |
| `frep_render` | Headless CPU JIT renderer (PPM output) |
| `frep_advanced` | Reference scene + SSAA from the command line |
| `frep_multipath` | Headless multi-path renderer / comparator (PPM); `--paths`, `--decompose`, `--merge`, post-process |
| `frep_gallery` | Generates the examples gallery (`tools/gallery.cpp`) |
| `frep_bench` | Runs the benchmark suite, prints markdown |
| `frep_tests` | GoogleTest suite |
| `plugins/capsule_plugin.so` | Example plugin |

---

## Running the GUI

```bash
./build/frep_designer                # default — opens an example scene
./build/frep_designer --empty        # start with an empty scene
./build/frep_designer --scene file.json    # load a scene file on startup
./build/frep_designer --realtime     # use QVulkanWindow real-time path
                                     # if hardware Vulkan is available
./build/frep_designer --version      # print version
./build/frep_designer --help         # show this list
```

The window has six side-panel tabs (Scene, Render, Expression,
Material, Lights, Node Graph) plus the viewport and toolbar:

- **Viewport** — the rendered scene. Click and drag to orbit; wheel to
  zoom; click an object to pick it. While dragging the camera, the
  viewport renders at half resolution for responsiveness; full res
  is restored on release.
- **Scene tab** — object list with parameter editor for the selected
  object, plus undo-aware add/remove/visibility/color controls.
- **Render tab** — viewport settings: CPU vs GPU toggle, SSAA level,
  render time readouts.
- **Expression tab** — text editor for CustomExpr nodes with 6 pre-
  canned samples, live syntax validation, parse errors with column
  numbers.
- **Material tab** — pattern, dual albedo color pickers, scale /
  roughness / metallic spin boxes, texture file browser.
- **Lights tab** — add / remove point lights, position spin boxes,
  intensity slider, color picker.
- **Node Graph tab** — visual editing of the FRepNode tree for the
  currently selected object. "Editing:" dropdown switches the active
  object; "Follow" checkbox controls whether selection changes
  elsewhere update the graph; "Fit" frames all nodes in view. The
  right-click palette groups node types by category — Primitives,
  Operations, Transforms (Translate, non-uniform Scale, RotateX/Y/Z),
  Deformations (TwistY, BendXY, TaperY), and Plugins. Instances appear
  in the graph as a pink node labelled with the target object's id (→ id);
  they aren't created from the palette (an instance needs a target — make
  one from the Scene toolbar), and their shared target subtree is not
  re-expanded in the graph to avoid duplicating it.

### About `--realtime`

The default viewport renders the scene to an offscreen Vulkan storage
buffer, reads it back to host memory, and blits it into a QWidget via
QPainter. This works on any system with Vulkan (including Mesa's
software `llvmpipe`), but the host-side readback imposes a 5-15 ms
floor on frame time.

The `--realtime` flag opts into a QVulkanWindow-based path that
renders directly into the swapchain image without host readback, at
60 FPS on real GPU hardware. The renderer compiles the live scene to
GLSL → SPV at startup, dispatches the compute shader into a private
storage image, and `vkCmdBlitImage`s the result into the swapchain
image every frame.

Interactive controls in the real-time viewport:

- **Left-mouse drag** — orbit the camera around the world origin
  (same yaw / pitch sensitivity as the offscreen viewport).
- **Mouse wheel** — dolly the camera in and out (clamped to
  2–40 world units).
- The camera state is local to the real-time window and is
  initialised from `scene_->camera()` on first interaction, so a
  freshly-loaded `.frep` keeps its authored viewpoint until the user
  actually drags.

Scene editing — adding objects, moving them, changing materials, or
adjusting lights from the inspector — triggers a pipeline rebuild on
the next frame. Geometry edits are typically 10-50 ms on a discrete
GPU (the user perceives a brief pause); camera-only changes are
free, since the camera lives in push constants and is re-read every
frame without recompilation.

Caveats:

- The flag only takes effect when a hardware Vulkan device is
  detected (discrete GPU, integrated GPU, or virtual GPU); on
  systems exposing only a software driver, it prints a diagnostic
  and falls back to the offscreen path.
- When `--realtime` is active, the offscreen viewport is hidden but
  still exists internally; some signal sinks (status bar render-time
  messages, the screenshot-save action) still read from it, so the
  status bar may show stale times. This is purely cosmetic and will
  be cleaned up alongside an `IViewport` abstraction in a future
  release.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

### Adding objects

The toolbar buttons place a primitive at the origin with default
parameters. The **Plugin** dropdown shows every registered plugin
type — click `+ Add plugin` to insert one. Once added, an object
appears in the Scene inspector and the node graph.

### Editing parameters

In the Scene inspector's Scene tab, click an object to select it.
The properties below the list show:

- **Visible** — toggle the object on/off without removing it.
- **Albedo** — primary surface colour. Click the colour swatch to open
  the picker.
- Plugin-specific sliders / spinboxes for radius, dimensions, twist
  amount, etc.

Editing any parameter triggers a debounced re-render (~80 ms delay).
Parameter changes use the incremental compilation path so most edits
recompile in a few milliseconds.

### Importing a mesh

`File → Import mesh (OBJ/STL) as SDF...` opens a file picker. After
the file is loaded, two dialogs appear:

1. **Voxelisation resolution** (default 48; range 8–256). Higher =
   sharper but slower to import and more RAM.
2. **Sparse-octree tolerance** (default 0 = dense). Setting this above
   zero compresses the voxel grid (see "Sparse compression" below).

The status bar reports the imported triangle count, voxel resolution,
RAM cost, and (if sparse) compression ratio.

### Switching to GPU rendering

`Render → Use GPU (Vulkan compute)` toggles between the CPU JIT path
and the GPU compute path. If Vulkan is missing or `glslangValidator`
isn't on PATH, the checkbox reverts to off and the status bar shows
why.

GPU mode is significantly faster on complex scenes (see Benchmarks
section). The rendered image is visually almost identical — the only
deliberate difference is that GPU shading uses simpler Phong while
CPU uses Cook-Torrance.

### Exporting

- `File → Export rendered image` — PNG / JPEG of the current viewport.
- `File → Export mesh` — OBJ or STL extracted via marching cubes.
  A dialog asks for grid resolution and isosurface value.

### Undo / redo

`Ctrl+Z` / `Ctrl+Y` (or the Edit menu). The undo stack records:

- Add / remove object
- Parameter change (one entry per slider release, not per pixel)
- Material change
- Camera preset change

Resolution changes and view-only camera orbits are NOT in the undo
stack on purpose — they're considered transient view state.

---

## Command-line tools

### Render a fixed scene to PPM

```bash
./build/frep_render output.ppm 1280 720
```

The "advanced" variant uses a richer hand-coded scene with super-sampling:

```bash
./build/frep_advanced output.ppm 1280 720 --ssaa 2
```

### Render on GPU

```bash
# Built-in test scene:
./build/frep_multipath examples/01_csg_basic.json --paths gpu_glsl --merge stitch --width 1280 --height 720 --out output.ppm

# A scene file:
./build/frep_multipath my_scene.json --paths gpu_glsl --merge stitch --width 1280 --height 720 --out output.ppm
```

### Generate the gallery

```bash
./build/frep_gallery /tmp/gallery
# Writes 01_csg.ppm through 07_hero.ppm into /tmp/gallery/.
```

### Run benchmarks

```bash
./build/frep_bench > benchmarks.md
```

Outputs a markdown report with timings for the JIT path, GPU vs CPU,
incremental compilation, sparse compression, and BVH voxelization.

---

## Scene file format

Scenes are stored as JSON. The structure:

```json
{
  "objects": {
    "ball": {
      "geometry": {
        "type": "Sphere",
        "r": 1.0
      },
      "material": {
        "albedo": [0.9, 0.4, 0.3]
      },
      "visible": true
    },
    "floor": {
      "geometry": {
        "type": "Plane",
        "nx": 0, "ny": 1, "nz": 0,
        "d": 1.0
      },
      "material": {
        "albedo": [0.55, 0.55, 0.55]
      }
    }
  },
  "lights": [
    {"pos": [5, 7, 4], "color": [1, 1, 0.95], "intensity": 1.0}
  ],
  "camera": {
    "position": [0, 1.8, 5.5],
    "target": [0, 0, 0]
  }
}
```

### Node types

Each `geometry` object has a `type` field plus type-specific parameters:

| `type` | Parameters | Notes |
|---|---|---|
| `Sphere` | `r` | radius |
| `Box` | `hx`, `hy`, `hz` | half-extents |
| `Plane` | `nx`, `ny`, `nz`, `d` | normal + signed distance offset |
| `Union` | `child` × 2 | min(a, b) |
| `Intersection` | `child` × 2 | max(a, b) |
| `Difference` | `child` × 2 | max(a, -b) |
| `SmoothUnion` | `child` × 2 + `k` | quintic smin |
| `Negate` | `child` | flips inside/outside |
| `Translate` | `child` + `tx, ty, tz` | offset in space |
| `Scale` | `child` + `s` (uniform) or `sx, sy, sz` (non-uniform) | scale; non-uniform makes ellipsoids etc. |
| `RotateX` | `child` + `a` (radians) | rotate around X |
| `RotateY` | `child` + `a` (radians) | rotate around Y |
| `RotateZ` | `child` + `a` (radians) | rotate around Z |
| `TwistY` | `child` + `k` | helix-twist with rate k |
| `BendXY` | `child` + `k` | bend around X by curvature k |
| `TaperY` | `child` + `t, h` | linear taper, ratio t over height h |
| `Instance` | `target_id` | live reference to another object's geometry (shares it, not a copy) |

Composite nodes recursively contain a `child` (or `children`) JSON
object that follows the same schema.

### Materials

```json
"material": {
  "albedo": [r, g, b],
  "albedo2": [r, g, b],
  "pattern": "Checker",
  "pattern_scale": 4.0
}
```

`pattern` can be `Solid`, `Checker`, `Stripes`, `GradientY`, `Noise`,
or `Texture`. For `Texture`, the loader doesn't read image files
directly — use the C++ API or the GUI to assign the image. (Future:
add a `texture_path` field referencing a file.)

---

## Writing a plugin

A plugin is a `.so` (Linux) / `.dylib` (macOS) that registers one or
more new node types. The example plugin in `plugins/extra_primitives.cpp`
adds a `Capsule` node. The structure:

```cpp
#include "core/plugin/plugin_api.hpp"
#include "core/frep/node.hpp"

using namespace frep;

class CapsuleNode : public FRepNode {
public:
    CapsuleNode(float r, float h, std::string nid = "cap") {
        id = std::move(nid);
        params["r"] = r;
        params["h"] = h;
    }
    const char* type_name() const noexcept override { return "Capsule"; }

    float eval(float x, float y, float z) const override {
        float r = params.at("r");
        float h = params.at("h");
        float qy = std::max(0.0f, std::abs(y) - h * 0.5f);
        return std::sqrt(x*x + qy*qy + z*z) - r;
    }

    llvm::Value* codegen(CgCtx& c) const override {
        // Emit the same formula in LLVM IR. See codegen.cpp for examples.
        return /* ... */;
    }
};

extern "C" void frep_plugin_register(plugin::PluginRegistry& reg) {
    reg.add("Capsule",
        [](){ return std::make_shared<CapsuleNode>(0.3f, 1.0f); },
        /* parameter spec */ {
            {"r", 0.1f, 2.0f, 0.3f, "Capsule radius"},
            {"h", 0.0f, 5.0f, 1.0f, "Capsule height"}
        });
}
```

Plugins are loaded at startup from a configurable directory (default:
`plugins/`). The registry feeds the GUI's plugin dropdown automatically.

For nodes that should work in GPU rendering, currently the GLSL emitter
hard-codes its supported nodes (see `core/gpu/glsl_emitter.cpp` —
`emit_node` dispatcher). Adding GPU support requires editing that
file; a plugin-friendly registry for GLSL emitters is on the roadmap.

---

## Performance tips

- **Use GPU mode for interactive work.** Even on a software Vulkan
  driver (Mesa llvmpipe) the GPU path is 2–5× faster than the CPU
  path on most scenes. On a real GPU expect another 10×.
- **Lower the SSAA factor while iterating.** SSAA 2 (the default for
  some tools) quadruples render time. Drop to 1 while editing, raise
  to 2 or 3 for the final render.
- **For mesh-heavy scenes, use sparse compression.** At 128³ resolution
  with tolerance 0.10, the voxel grid shrinks 30× with only minor
  surface detail loss. Set tolerance to 0 to disable.
- **Stay in Incremental compile mode while sweeping a slider.** The
  Auto mode picks this automatically after a few rapid edits. The
  cost: ~ms per recompile instead of ~hundreds of ms.
- **Marching cubes resolution is independent of MeshSDF resolution.**
  When exporting a mesh, the MC grid can be much higher than the SDF
  voxel grid that you imported the original mesh into.
- **Camera orbit doesn't recompile.** Only parameter / topology edits
  trigger compile work. Orbiting around the scene is purely a render
  cost — fast on both CPU and GPU.

---

## Troubleshooting

### "Vulkan not available" when toggling GPU mode

Check that:

- `libvulkan.so.1` is installed (`apt install libvulkan1`).
- `vulkaninfo --summary` reports at least one device (Mesa's
  `llvmpipe` is fine for testing).
- Your user has access to `/dev/dri/render*` if using a real GPU.

If only `llvmpipe` is listed, GPU mode will work but won't be much
faster than CPU because llvmpipe runs on CPU cores anyway.

### "glslangValidator not found"

Install `glslang-tools` (Ubuntu) or `glslang` (Arch). The binary
should be on `PATH`. The build configure step warns when it's
missing and disables `FREP_BUILD_GPU` automatically.

### CMake fails to find LLVM

Point it at the right CMake config:

```bash
cmake -B build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm ...
```

If LLVM is in a non-standard location (e.g. a Homebrew install), use:

```bash
cmake -B build -DLLVM_DIR=$(brew --prefix llvm@20)/lib/cmake/llvm ...
```

### Tests fail with "Bus error" or similar

Some Vulkan validation layers can crash on the sandbox. Run with
validation disabled:

```bash
VK_LOADER_LAYERS_DISABLE='*' ./build/frep_tests
```

### The GUI is too slow on a Retina / 4K display

The viewport renders at native pixel resolution. Drop the window
size before editing, or toggle GPU mode (`Render → Use GPU`).

### "Plugin failed to load"

Check `LD_LIBRARY_PATH` includes the directory containing the .so,
and that the plugin was built against the same LLVM version as the
host. A mismatch shows as obscure symbol errors at load time.

---

_Last updated: 2026-05-19. For questions and contributions see the
project's GitHub issues._
