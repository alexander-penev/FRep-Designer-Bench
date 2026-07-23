# FRep Designer 4.0 — Architecture

This document maps out the major components of the system and shows
how data flows between them. For user-facing documentation, see
`docs/USER_GUIDE.md`.

## High-level view

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Qt6 GUI Layer                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   Viewport   │  │ NodeGraphView│  │     SceneInspector       │   │
│  │ (orbit, pick │  │ (drag/drop   │  │ (sliders, color pickers, │   │
│  │  CPU/GPU)    │  │  node editor)│  │  light editor, exports)  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │
└─────────┼─────────────────┼───────────────────────┼─────────────────┘
          │                 │                       │
          └─────────────────┼───────────────────────┘
                            │ all edits go through →
                  ┌─────────▼─────────┐
                  │    UndoStack      │   Command pattern; param edits,
                  │ (core/undo/...)   │   topology changes, materials.
                  └─────────┬─────────┘
                            │ applies to →
              ┌─────────────▼─────────────────────────┐
              │             SceneGraph                │
              │   (core/frep/scene.hpp)               │
              │                                       │
              │  ┌──────────────┐  ┌──────────────┐   │
              │  │   Objects    │  │   Camera +   │   │
              │  │ (FRepNode +  │  │   Lights     │   │
              │  │  Material)   │  │              │   │
              │  └──────┬───────┘  └──────────────┘   │
              └─────────┼─────────────────────────────┘
                        │
                        │ FRepNode tree
                        │
        ┌───────────────┴────────────────┐
        │                                │
        │ CPU path                       │ GPU path
        ▼                                ▼
  ┌──────────┐                     ┌──────────────┐
  │SceneCode-│                     │ GlslEmitter  │
  │  gen     │                     │              │
  │(IR build)│                     │ (text build) │
  └────┬─────┘                     └──────┬───────┘
       │ llvm::Module                     │ GLSL source
       ▼                                  ▼
  ┌──────────┐                     ┌──────────────────┐
  │ JitEngine│                     │ glslangValidator │  (subprocess)
  │(orcjitv2)│                     │                  │
  └────┬─────┘                     └────────┬─────────┘
       │ function pointer                   │ SPIR-V binary
       ▼                                    ▼
  ┌──────────┐                     ┌──────────────┐
  │TileSched-│                     │  VulkanCtx   │
  │  uler    │                     │ (compute     │
  │(thread   │                     │  pipeline)   │
  │ pool +   │                     │              │
  │  SSAA)   │                     │              │
  └────┬─────┘                     └──────┬───────┘
       │ float[W*H*4] RGBA               │ uint8[W*H*4] RGBA
       │                                  │
       └────────────┬─────────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │    QImage    │  Always blits through the same QImage,
            │ (paintEvent) │  so the rest of the viewport is path-agnostic.
            └──────────────┘
```

## Component responsibilities

### `core/frep/` — the F-Rep data model

The "what" of the program. No rendering, no compilation logic — just
the data structures that describe geometry, materials, and scenes.

- `node.hpp` — `FRepNode` base class, `NodeKind` enum, `CgCtx`
  (codegen context bundle), `AABB`.
- `primitives.hpp` — `SphereNode`, `BoxNode`, `PlaneNode`.
- `operations.hpp` — `UnionNode`, `IntersectionNode`, `DifferenceNode`,
  `SmoothUnionNode`, `NegateNode`.
- `transforms.hpp` — `TranslateNode`, `ScaleNode`, `RotateYNode`.
- `deformations.hpp` — `TwistYNode`, `BendXYNode`, `TaperYNode`. Each
  carries a Lipschitz correction so the resulting field stays a valid
  SDF.
- `scene.hpp` — `SceneGraph` (object + material + lights + camera),
  `Material`, `PointLight`, `Camera`.
- `mesh_sdf.hpp` — `MeshSDFNode`, which voxelises an imported mesh
  into a 3D SDF grid (optionally sparse).

Every node has three core APIs:

1. `eval(x, y, z)` — pure C++ evaluation, used by the picker, BVH,
   marching cubes, and unit tests.
2. `codegen(CgCtx&)` — emits LLVM IR for the JIT path.
3. `structural_hash()` — hashes topology (not parameter values) so
   the incremental compiler can detect when parameters changed but
   structure didn't.

### `core/mesh/` — mesh utilities

- `marching_cubes.{hpp,cpp}` — both directions: extract a surface mesh
  from any FRepNode tree, and (separately) load OBJ / STL files.
- `triangle_bvh.hpp` — bounding-volume hierarchy used to accelerate
  the voxelisation step of `MeshSDFNode` by ~27× over the brute-force
  closest-triangle search.
- `sparse_sdf_octree.hpp` — pooled-node octree that compresses dense
  SDF grids by merging uniform regions; built top-down from a dense
  grid, "flattened" back to dense for the IR codegen path (which still
  takes a flat array as the JIT input).

### `core/compiler/` — the CPU JIT pipeline

- `codegen.{hpp,cpp}` — walks an FRepNode tree and emits a
  `__render_tile` function in LLVM IR. The function takes a tile
  rectangle and writes RGBA8 pixels into a caller-supplied buffer.
  Includes the full shading pipeline (normal estimation, Cook-Torrance
  PBR, shadows, optional AO, optional procedural pattern mixing).
- `jit_engine.{hpp,cpp}` — wraps LLVM ORCv2: takes a Module, runs the
  optimization pipeline (O3), and JITs it into an executable function.
  Single shared linker; modules can be unloaded.
- `incremental.hpp` — caches the JIT'd module and reuses it when only
  parameter values changed. Three modes: `Constant` (always recompile),
  `Incremental` (only patches param constants), `Auto` (switches to
  Incremental after sustained editing).
- `picker.hpp` — BVH-accelerated SDF intersection for mouse picking.
- `spirv_external.hpp` — wraps the Khronos `llvm-spirv` tool as a
  subprocess to translate the same JIT module into SPIR-V (used by
  the old, non-GLSL GPU path; superseded by the GLSL emitter for
  actual rendering but kept for SPIR-V emission tests).
- `retarget_spirv.hpp` — sets the IR module's target triple +
  data layout to SPIR-V before handing it to the external translator.

### `core/tracer/` — the renderer

- `tile_scheduler.{hpp,cpp}` — splits the framebuffer into 64×64
  tiles, runs them through `std::async`, gathers the pixels. Handles
  SSAA by rendering at `N × W × H` and box-filtering down.
- The actual shading and ray-marching lives inside the JIT'd code
  (emitted by `core/compiler/codegen.cpp`), not in a C++ function —
  that's the whole point of JIT.

### `core/gpu/` — the Vulkan compute pipeline

- `vulkan_ctx.{hpp,cpp}` — minimal compute Vulkan context. Owns the
  instance, device, descriptor set, pipeline, command buffer, fence,
  and (optionally) storage buffers for mesh voxels and texture
  pixels. `render()` uploads push constants, dispatches the workgroup
  grid, copies the storage image into a host-visible buffer, and
  returns RGBA8 bytes.
- `glsl_emitter.{hpp,cpp}` — the GPU mirror of `core/compiler/codegen`.
  Walks an FRepNode tree and emits GLSL source for a compute shader
  with the same calling convention as the hand-written
  `sphere_trace.comp`. Each object's SDF expression gets its own
  `{ ... }` scope so SSA variable names cannot leak.
- `glsl_compile.hpp` — runtime GLSL → SPIR-V via a `glslangValidator`
  subprocess. Same pattern as `spirv_external.hpp` on the CPU side.
- `../../gpu/sphere_trace.comp` — the hand-written reference shader,
  built once at CMake configure time (the reference shader used by
  the GPU_GLSL path and the realtime Vulkan viewport).

### `core/io/` — file formats

- `scene_io.{hpp,cpp}` — JSON save / load of `SceneGraph`. Needs a
  `PluginRegistry*` at load time so it can instantiate plugin nodes.
- `bmp_loader.hpp` — header-only BMP loader (24/32-bit, uncompressed)
  used by image textures. PNG is on the roadmap (drop in stb_image.h).

### `core/plugin/` — extensibility

- `plugin_api.hpp` — the `PluginRegistry` that dynamically loaded
  `.so`/`.dylib`s register their node factories into.
- `plugin_loader.hpp` — `dlopen` wrapper that scans a directory.

### `core/undo/` — undo/redo

- `undo_stack.hpp` — `Command` interface plus the stack itself.
  Specific commands (AddObject, RemoveObject, SetParameter, etc.)
  live alongside.

### `gui/` — the Qt frontend

- `main_window.{hpp,cpp}` — top-level window, menus, undo wiring,
  plugin loading, file dialogs.
- `viewport.{hpp,cpp}` — the rendering area. Talks to
  `IncrementalCompiler` and the GPU pipeline. Async render via
  `QTimer` debounce.
- `scene_inspector.{hpp,cpp}` — properties pane (object list,
  parameter sliders, light editor, render settings).
- `node_graph.{hpp,cpp}` — visual node graph editor with
  drag-and-drop wires. A node here corresponds to an FRepNode in
  the scene; topology changes write through the undo stack.

### `tools/` — command-line utilities

- `render_ppm.cpp` → `frep_render` — minimal CPU JIT renderer.
- `render_advanced.cpp` → `frep_advanced` — richer reference scene
  with SSAA.
- `multipath_driver.cpp` → `frep_multipath` — multi-path render /
  compare / split driver (`--paths`, `--decompose`, `--merge`).
- `dist_render.cpp` → `frep_dist_render` — distributed master/worker.
- `gallery.cpp` → `frep_gallery` — generates the examples gallery
  (seven scenes, one PPM each).
- `benchmarks.cpp` → `frep_bench` — runs the performance benchmark
  suite, prints markdown.

### `tests/` — GoogleTest suite

The test suite (300+ tests) covers roughly:

- Node primitive evaluation
- CSG composition correctness
- LLVM codegen (round-trip against `eval()`)
- Marching cubes mesh extraction
- Mesh import (OBJ/STL parsing)
- BVH correctness
- Sparse SDF octree round-trip
- Undo / redo
- Incremental compilation
- SPIR-V emission
- Vulkan availability + render
- GLSL emitter (per node type, plus end-to-end render)
- GPU patterns + MeshSDF
- BMP loader + GPU texture rendering

## Data-flow summary

A typical interactive frame, in 9 steps:

1. User drags a slider in the Scene inspector.
2. The inspector pushes a `SetParameter` command onto the undo stack
   and applies it to the SceneGraph.
3. The scene's `change_observer` fires; the viewport's `dirty_` flag
   is set and a debounce timer is started.
4. After ~80 ms with no further edits, the timer fires
   `do_render()`.
5. `IncrementalCompiler::compile_if_changed()` is called. If only
   parameter values changed since the last frame, it patches the
   IR constants and reuses the cached JIT'd function — typically
   under 5 ms. Otherwise it rebuilds + JITs the module — ~30-100 ms.
6. `TileScheduler::render()` splits the framebuffer into tiles and
   dispatches them across `std::async` workers.
7. Each tile invokes the JIT'd `__render_tile` function which does
   ray-marching, shading, AO, shadows, and writes RGBA floats.
8. The viewport gathers all tiles into a `QImage` and calls
   `update()` → `paintEvent` blits to the window.
9. The status bar shows compile-ms + render-ms + whether the cache
   was hit.

When GPU mode is on, steps 5–8 are replaced by:

5'. `GlslEmitter::emit(scene)` produces a GLSL source string
    (cached against the scene's structural hash).
6'. `compile_glsl_to_spv()` invokes `glslangValidator` as a
    subprocess (cached).
7'. `VulkanCtx::create()` reuses the cached pipeline; `render()`
    pushes the current camera as push constants and dispatches.
8'. The returned RGBA8 buffer is memcpy'd into the same QImage.

## CustomExprNode — runtime-text math expressions

`CustomExprNode` lets users define an SDF from a text expression at
runtime, e.g. `"sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)"` for
the gyroid surface. Three back-ends consume the same expression:

```
                 ┌─ frep::expr::parse() ─┐
                 │                        │
                 │   recursive descent    │
                 │   tokenizer + parser   │
                 │   in expr_ast.cpp      │
                 │                        │
                 └────────────┬───────────┘
                              │
                              ▼
                  ┌────────── AST ──────────┐
                  │ (frep::expr::Node tree, │
                  │  shared_ptr-managed,    │
                  │  immutable)             │
                  └────────────┬────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
     │ LLVM IR     │   │ Direct CPU  │   │ GLSL source  │
     │ codegen     │   │ interpreter │   │ emission     │
     │             │   │             │   │              │
     │ CustomExpr  │   │ eval_ast()  │   │ emit_glsl_   │
     │ Compiler::  │   │ in          │   │ ast() in     │
     │ compile()   │   │ custom_expr │   │ custom_expr  │
     │             │   │ .cpp        │   │ .cpp         │
     └─────┬───────┘   └─────┬───────┘   └──────┬───────┘
           │                 │                  │
           ▼                 ▼                  ▼
     used by:           used by:           used by:
     scene_sdf()        FRepNode::         GlslEmitter
     in JIT path        eval() —           default branch
     and CPU/x86        picker, BVH,       (NodeKind::Plugin
     codegen            marching cubes     fallback)
```

This shared-AST architecture (added in a refactor) replaces what
were originally two independent parsers (one in
`CustomExprCompiler` for LLVM, one as a `Parser` struct inside
`CustomExprNode` for eval). The old design had to be kept in sync
manually — adding a new function meant editing two lexers, two
parsers, two arity tables. The shared AST means new operators or
functions are added in three places that are all visible from the
same source file (`custom_expr.cpp`), with the grammar itself
defined exactly once (`expr_ast.cpp`).

The AST is cached on the `CustomExprNode` (as a `shared_ptr<const
expr::Node>`) so repeated calls to any back-end re-use the parse
result. The cache is filled lazily on first back-end invocation.

## Choosing between CPU and GPU

| Situation | Recommended path |
|---|---|
| Editing parameters interactively | Either; CPU is fine for simple scenes |
| Complex scenes (5+ objects, deformations) | GPU |
| MeshSDF-heavy scenes | GPU (memory bandwidth wins) |
| High-resolution final render (>1080p) | GPU |
| Want PBR / Cook-Torrance shading | CPU (GPU uses simpler Phong) |
| No Vulkan available | CPU (only option) |
| Need exact reproducibility for tests | CPU (deterministic; GPU varies by driver) |

---

_Last updated: 2026-05-19._
