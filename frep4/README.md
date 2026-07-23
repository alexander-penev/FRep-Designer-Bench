# F-Rep Designer — Proof of Concept

An implementation of the "everything is code" approach from the papers
*F-Rep Designer 2.0* and *Geometric Modeling Systems as Visual Programming
Languages*:

- **The model = a program** in a visual language (the FRepNode tree = AST)
- **Visualization = compilation + execution** (LLVM JIT)
- **Geometry + normals + materials + shaders + ray tracer = one LLVM module** → O3 → native code
- **Plugin system** + user-supplied analytic expressions, compiled JIT-style
- **Retarget the same model across four heterogeneous executors** — CPU JIT,
  CUDA (NVPTX), Vulkan compute (GLSL/SPIR-V), and Vulkan ray tracing — and run
  them concurrently so their throughputs **add**

## Stack

| Layer | Technology |
|-------|-----------|
| **Language** | **C++26** (`-std=c++26`, pack indexing, modules-ready) |
| Code generation | **LLVM 20+** (clang-20 / LLVM 20 is the tested toolchain) |
| JIT / Optimization | ORC LLJIT + PassBuilder O3 |
| Retargeting | CPU JIT · NVPTX→CUDA · GLSL→SPIR-V→Vulkan compute · RT shaders→Vulkan ray tracing |
| Ray tracing | Sphere tracing + soft shadows + AO, emitted in LLVM IR; hardware RT via Vulkan RT pipeline |
| Normals | Forward-mode AD in LLVM IR (3 sdf calls, exact) |
| Acceleration | BVH for scene_material; per-CSG-group BLAS broad-phase on RTX; IncrementalCompiler scene-hash cache |
| Energy | CPU RAPL (perf_event / powercap) + GPU NVML, reported as Mpix/kWh |
| Distributed | TCP master/worker, pull + push schedulers, confirmed on a real LAN |
| GUI | Qt6 (viewport, node graph editor, scene inspector, ray-cast pick, File I/O) |
| Plugins | dlopen dynamic loading + compile-time PrimitivePlugin concepts |
| Serialization | JSON (.frep files), no external dependencies |
| Concurrency | `std::jthread` tile scheduler |
| Build | CMake 3.28+ |
| **Tests** | **GoogleTest 1.14+** (gtest + gmock) — 300+ tests + a node graph logic test |

## Documentation

- 🎓 [**Tutorial**](docs/TUTORIAL.md) — build your first scene step
  by step, from empty viewport to STL export.
- 📖 [**User guide**](docs/USER_GUIDE.md) — building, running the GUI,
  scene file format, plugin authoring, troubleshooting.
- 🏗 [**Architecture overview**](docs/ARCHITECTURE.md) — component map,
  data flow, design decisions.
- 🔌 [**Plugin authoring**](docs/PLUGIN_AUTHORING.md) — write a custom
  F-Rep primitive as a `.so`/`.dll` plugin, with full code example.
- ⚡ [**Performance tuning**](docs/PERFORMANCE_TUNING.md) — when to use
  CPU vs GPU, MeshSDF resolution sweet spots, memory budgets.
- 🖼 [**Examples gallery**](docs/GALLERY.md) — eleven showcase scenes
  demonstrating CSG, deformations, mesh import, patterns, textures,
  and CPU-vs-GPU comparison.
- 📊 [**Benchmark report**](docs/BENCHMARKS.md) — performance numbers
  across all major features, regenerable via `./build/frep_bench`.
- 📂 [**Example scenes**](examples/) — hand-editable `.json` scene files
  for `frep_gpu_render --scene`.
- 📚 **API reference** — generated from inline Doxygen-style comments:
  ```bash
  cmake --build build --target docs
  ```
  outputs `docs/api/html/index.html`. Requires `doxygen` + `graphviz`
  installed.
- 📈 **Performance regression detector** — committed baseline numbers
  at `tools/perf_baseline.json`, with a comparison script that fails
  CI on >2× slowdowns:
  ```bash
  ./build/frep_bench --json > /tmp/bench.json
  python3 tools/perf_check.py /tmp/bench.json
  ```

## License

[MIT](LICENSE). The project links against LLVM (Apache-2.0 with LLVM
Exceptions), Qt6 (LGPL v3 or commercial), Vulkan (Apache-2.0 / MIT),
and GoogleTest (BSD-3-Clause); see `LICENSE` for details.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the full
build, all tests, every CLI tool's smoke test, and renders every
example scene on every push.

## Implemented components

### Core engine + ray tracer

```
core/frep/        FRepNode AST
  primitives.hpp     Sphere (true SDF), Box, Plane
  operations.hpp     Union (min), Intersection (max), Difference, SmoothUnion, Negate
  transforms.hpp     Translate, Scale, RotateY
  scene.hpp          SceneGraph, Material, Camera, PointLight
  node.hpp           Base FRepNode + CgCtx

core/ad/          Automatic differentiation
  forward_ad.hpp     Dual<T>, gradient() (for tests and normals)

core/compiler/    Code generation and JIT
  codegen.hpp/cpp    SceneCodegen → unified LLVM Module
  jit_engine.hpp     LLJIT wrapper + O3 PassBuilder
  llvm_compat.hpp    LLVM version compatibility shims (CreateMinNum, etc.)

core/tracer/
  tile_scheduler.hpp std::jthread parallel renderer

tools/
  render_ppm.cpp     CLI: scene → IR → JIT → PPM
```

### Plugin API + CustomExpr

```
core/plugin/
  plugin_api.hpp     PrimitivePlugin, RayTracerPlugin, RetargetPlugin concepts
                     + PluginRegistry with runtime registration

core/frep/
  custom_expr.hpp/cpp  CustomExprNode — a text expression → LLVM IR
                       (recursive descent parser; supports +-*/, parentheses,
                       sin/cos/tan/sqrt/abs/exp/log/min/max/pow, pi, e)

core/compiler/
  retarget_spirv.hpp   SPIR-V retarget: LLVM IR → spirv64-unknown-unknown triple

plugins/
  extra_primitives.hpp  TorusPlugin + OctahedronPlugin (demo plugins)

tools/
  render_advanced.cpp   CLI: scene with custom exprs + plugins + SPIR-V emit

tests/
  test_phase4.cpp       CustomExpr parser, Plugin API, end-to-end
```

### GUI + dynamic plugins + acceleration

```
gui/
  viewport.hpp/cpp        Qt6 viewport — orbit camera, async render thread
  main_window.hpp/cpp     main window, toolbar, File → Open/Save menu
  scene_inspector.hpp/cpp object list, property editor (color, visibility)
  node_types.hpp          catalog of node type metadata
  node_graph.hpp/cpp      node graph editor — QGraphicsScene with the FRepNode tree
  main.cpp                entry point; loads dynamic plugins from ./plugins/

core/plugin/
  plugin_loader.hpp       dlopen-based dynamic loading of .so plugins

core/compiler/
  incremental.hpp         IncrementalCompiler — scene-hash caching
  bvh.hpp                 BVH tree for scene_material acceleration (O(log n))
  picker.hpp              ScenePicker — ray-cast object selection (JIT scene_pick)

core/frep/
  ad_ir.hpp               forward-mode AD over LLVM IR Value*
  node.cpp                codegen_grad fallback (finite-diff)
  primitives_ad.cpp       exact AD for Sphere/Box/Plane
  operations_ad.cpp       exact AD for CSG operations and transforms
  aabb.cpp                axis-aligned bounding box for all node types

core/io/
  json.hpp                minimal JSON parser/writer (no dependencies)
  scene_io.hpp/cpp        SceneGraph <-> JSON serialization (File → Open/Save)

plugins/dynamic/
  capsule_plugin.cpp      example .so plugin (Capsule primitive)
```

**New in 4.0:**
- **Qt6 GUI** — viewport with orbit camera, scene inspector, property panel,
  primitive-adding toolbar, File → Open/Save menu
- **Soft shadows + ambient occlusion** — emitted directly in LLVM IR
- **Multi-light shading** — each scene light contributes its own colour-tinted
  diffuse + specular + soft shadow ray; light loop is unrolled at compile time
  into the JIT-ed `render_tile`. Default scene falls back to one key light if
  empty.
- **Lights editor in GUI** — dedicated "Lights" tab lets the user add/remove
  point lights and edit each light's position (X/Y/Z), colour (via a colour
  picker), and intensity. Changes trigger a recompile and re-render. Lights
  are persisted in the .frep file along with the geometry.
- **Image export from GUI** — File → "Export rendered image..." saves the
  current viewport contents as PNG, JPEG, BMP, or PPM (format chosen by the
  file extension).
- **Mesh export (marching cubes)** — File → "Export mesh (OBJ/STL)..."
  samples the SDF on a 64^3 grid and runs the classic Lorensen-Cline
  marching cubes algorithm to extract an iso-surface triangle mesh. The
  algorithm runs in plain C++ (no JIT) via a new `FRepNode::eval` virtual
  method, so any node tree composed of built-in primitives, CSG ops,
  transforms, or plugin primitives can be meshed. Output formats: Wavefront
  OBJ (indexed) and ASCII STL (with face normals). 64^3 ~ 20k triangles in
  ~30 ms for a typical scene; 128^3 ~ 80k triangles in ~200 ms.
- **Node graph parameter editing** — double-click on any parameter row of
  a graph node pops a numeric input dialog (honouring the per-param
  min/max range with 4 decimals of precision). A faint dotted underline
  under each value is the visual cue that the field is editable. Changes
  emit `graph_changed`, which rebuilds the FRepNode tree and re-renders.
- **Plugin primitives in node graph** — context menu now exposes a
  "Plugins" submenu when a `PluginRegistry` is registered with the graph
  scene; each registered primitive (Torus, Octahedron, Capsule, ...)
  becomes a draggable node, with parameter ranges auto-derived from the
  plugin's `param_defaults`. `build_tree()` instantiates the right
  FRepNode by consulting the registry for any type not in the built-in
  catalog.
- **BVH-accelerated picker (ray-AABB early-exit)** — the JIT-emitted
  `scene_pick` now starts with a slab-method ray-AABB test against the
  scene's combined bounding box. Rays that miss the box return -1
  immediately, skipping the sphere-tracing loop entirely. Skipped when
  any object has a non-finite AABB (e.g. an infinite Plane).
- **Orthographic projection + camera presets** — the camera now has a
  `projection` field (`Perspective` or `Orthographic`) and an `ortho_size`
  parameter. The JIT-emitted ray generator branches on the sign of the
  passed-in view scale: positive → perspective (rays diverge from the
  camera position), negative → orthographic (parallel rays, origin
  sweeps the view plane). Projection mode and ortho size round-trip
  through `.frep` files. GUI exposes a "Camera" group on the Render tab
  with a Projection combobox, FoV / Size spinboxes, and four preset
  buttons (Front / Top / Right / Iso).
- **Undo / redo** — Edit menu now has Undo (Ctrl+Z) and Redo (Ctrl+Y)
  entries. A new `core/undo/undo_stack.hpp` provides a Command-pattern
  stack: each scene-mutating action (add/remove object, set material,
  toggle visibility, add/remove/edit light) wraps in a small `UndoCommand`
  that captures enough state to reverse itself. The stack truncates
  pending redos when a new command lands, fires a change observer for
  the GUI to refresh action labels ("Undo (Add ball)"), and clears on
  scene reload. 11 unit tests cover the stack mechanics and each
  command's round-trip behaviour.
- **Per-parameter incremental compilation** — node parameters (Sphere::r,
  Box dims, Plane normal/d, Translate offsets, Scale s, RotateY a) can be
  emitted either as IR constants (Constant mode, full O3 folding) or as
  runtime loads from a `float[]` buffer (Incremental mode). Three modes:
  Constant / Incremental / Auto. Auto starts Constant and latches to
  Incremental after 3 recompiles within 5 s. The JIT'd `render_tile`
  signature now takes a trailing `float* params` arg (unused / DCE'd in
  Constant mode). Slider edits that hit the fast path skip JIT entirely
  and only rewrite the buffer — ~75 ms saved per edit. 9 new tests
  cover binding-table population, Constant↔Incremental equivalence
  (bit-exact pixel match), buffer-edit-no-recompile flow, and the Auto
  policy state machine. GUI exposes the choice in a "Compile:" combobox
  on the Render tab.
- **Binary SPIR-V emission via Khronos llvm-spirv** — `SPIRVRetarget` now
  produces real binary `.spv` modules (not just LLVM-IR text). The
  pipeline writes the module to LLVM bitcode in-memory, spawns the
  Khronos `llvm-spirv` translator as a subprocess (auto-discovered on
  PATH: tries llvm-spirv-22..18 and the unsuffixed name), reads back
  the binary, and optionally invokes `spirv-val` for spec compliance.
  Tools and the side panel surface the validator verdict alongside the
  emit timing. The translator and validator are optional runtime
  dependencies — when missing, the retarget gracefully falls back to
  IR-only output. 6 new tests cover translator discovery, magic-number
  validation, calling-convention rewriting, and triple metadata.
- **Mesh import as SDF + booleans on imported meshes** — File → "Import
  mesh (OBJ/STL) as SDF..." loads a Wavefront OBJ or STL file (both
  ASCII and binary STL supported), voxelizes the triangle mesh into a
  signed-distance grid at a user-chosen resolution (default 48^3), and
  inserts the result as a `MeshSDFNode` — a regular FRepNode. Because
  it implements the standard FRepNode interface, every existing CSG
  operation (Union, Intersection, Difference, SmoothUnion, Negate) and
  Transform (Translate, Scale, RotateY) just works on it, so the user
  can perform booleans between imported meshes and procedural primitives
  with no extra plumbing. JIT codegen embeds the voxel grid as an
  LLVM `private constant [N x float]` and emits a trilinear-interpolation
  load at every query point; sphere tracing then runs the imported
  geometry at the same speed as procedural shapes. 8 new tests cover
  OBJ/STL round-trip, SDF approximation accuracy, AABB containment,
  empty-mesh robustness, and CSG operations on a MeshSDFNode child.
- **BVH-accelerated mesh voxelization** — the inside-out point-triangle
  distance and inside/outside parity tests during MeshSDFNode
  construction used to be O(N³ × T) brute force. A new
  `core/mesh/triangle_bvh.hpp` builds a binary BVH (median split on
  longest centroid axis, leaf size 8) once per import, and both queries
  traverse it with min-distance-to-AABB / slab-method pruning. Result:
  voxelization of a 5000-triangle sphere at 32³ went from ~2.9 s to
  ~108 ms — a **27× speedup**. The full ctest suite now runs in 2.3 s
  instead of 5.5 s. 5 new tests cover empty-BVH handling, single-
  triangle correctness, brute-force comparison on random query points,
  ray-hit parity validation, and leaf-size shape control.
- **Non-linear deformation nodes (twist / bend / taper)** — three new
  FRepNode subclasses in `core/frep/deformations.hpp` that warp the
  query space before evaluating the child SDF. `TwistYNode` rotates
  around the Y axis with angle proportional to y (k radians/unit);
  `BendXYNode` curves the X axis into an arc of radius 1/k;
  `TaperYNode` linearly scales the XZ plane with Y, optionally
  collapsing to a point at the top. Each emits IR with the appropriate
  warp + Lipschitz-bound correction so sphere tracing remains stable,
  and uses central-difference gradients for normal computation
  (`core/frep/deformations_ad.cpp`). 6 unit tests verify zero-parameter
  identity, topology preservation, taper-shrinks-top, and end-to-end
  JIT rendering.
- **Procedural materials (Checker / Stripes / GradientY / Noise)** —
  the `Material` struct gained `pattern` (enum) and `pattern_scale`
  fields plus a secondary `albedo2` colour. `scene_material` emits an
  in-IR pattern evaluator at each hit point, blending between the two
  albedos based on world-space position: 3D checkerboard, Y-axis
  stripes, smooth vertical gradient, or hash-based value noise. All
  evaluation happens in the JIT'd kernel — no textures, no external
  dependencies, just pure math. Cook-Torrance / Blinn-Phong shading
  applies on top normally. 3 unit tests verify checker color
  alternation, solid-material correctness, and noise produces
  statistical variation in surface brightness.
- **Sparse octree compression for MeshSDF** — a new
  `core/mesh/sparse_sdf_octree.hpp` compresses dense voxel SDF grids
  into a regular octree where regions of uniform value collapse into
  a single leaf. The Node struct uses a tagged-index layout (one int32
  per child reference, leaves and internals in separate pooled arrays),
  giving an 8-byte effective node footprint. MeshSDFNode gained an
  optional `sparse_tolerance` parameter — when > 0, the dense grid is
  rebuilt through the octree, leaves coalesce voxels within the
  tolerance band, and storage stats are recorded. Compression results
  for a typical voxelised sphere: at 128³ resolution the dense 8 MB
  grid compresses to **274 KB at tolerance 0.10 — a 30× reduction**.
  At tight tolerances the per-leaf overhead can exceed the dense float
  size, so sparse mode is opt-in and is documented as a trade-off.
  The JIT and GPU paths both re-flatten the octree back to a dense buffer,
  so the win is RAM-at-rest; an in-kernel octree walk (avoiding the flatten)
  is possible future work. 9 new tests cover power-of-two
  validation, reconstruction accuracy within tolerance, sample/flatten
  agreement, and end-to-end MeshSDFNode integration.
- **GPU compute path via Vulkan** — a proof-of-concept GPU renderer
  built on top of Vulkan 1.2. `gpu/sphere_trace.comp` is a hand-written
  GLSL compute shader that does sphere tracing on a fixed scene
  (sphere + ground plane, point-light Phong with soft shadows, sky
  gradient). It is compiled to SPIR-V at build time via
  `glslangValidator` (validates clean against the SPIR-V spec, unlike
  the LLVM-emitted SPIR-V from earlier milestones). `core/gpu/vulkan_ctx`
  hosts the Vulkan side: instance, physical-device pick (first with a
  compute queue), logical device, storage image (RGBA8), descriptor
  set, push constants matching the shader's `Push` block, command
  buffer with the standard layout-transition / dispatch / copy-to-buffer
  flow, fence-synchronised readback into a `std::vector<uint8_t>`.
  RAII teardown via a PIMPL `Impl` struct. The standalone tool
  `frep_gpu_render` saves a PPM. On Mesa's `llvmpipe` software Vulkan
  driver (CI environment) a 800×600 render takes 68 ms — already
  ~16× faster than the CPU JIT path on the equivalent scene, despite
  having no real GPU underneath. A real Vulkan GPU would push that
  further. The build system gates the whole path behind
  `FREP_BUILD_GPU` and probes for both `find_package(Vulkan)` and
  `glslangValidator`; if either is missing the rest of the project
  builds normally without it. 5 new tests cover availability probing,
  context creation, non-blank-image render, expected red-sphere albedo,
  and resize stability across multiple renders.
- **Scene → GLSL translator** — `core/gpu/glsl_emitter.{hpp,cpp}` walks
  an `FRepNode` tree and emits a complete GLSL compute shader with the
  same calling convention as the hand-written `sphere_trace.comp`,
  so the existing `VulkanCtx` consumes it unchanged. Supported nodes:
  Sphere, Box, Plane primitives; Union, Intersection, Difference,
  SmoothUnion, Negate CSG ops; Translate, Scale, RotateY transforms;
  TwistY, BendXY, TaperY deformations. Each object's SDF expression is
  emitted in its own `{ ... }` scope so SSA variable names cannot leak
  between objects or between the SDF and albedo functions. A
  `compile_glsl_to_spv` helper (in `core/gpu/glsl_compile.hpp`) spawns
  `glslangValidator` as a subprocess, mirroring the approach we use
  for the Khronos llvm-spirv translator. `frep_gpu_render` learned a
  `--scene <file.json>` argument that loads a saved scene, emits GLSL,
  compiles it at runtime, and renders it. End-to-end this lets arbitrary
  FRepNode trees execute on the GPU — not just the one hard-coded
  scene from the previous milestone. 8 new tests cover emit
  well-formedness for primitives, CSG, and deformations, plus
  glslangValidator round-trip and full Vulkan render for basic / CSG /
  deformed scenes.
- **GPU procedural materials** — the GLSL emitter's `scene_albedo`
  function now emits per-object `mix(albedo, albedo2, t)` blends where
  `t` is computed from the world-space hit point: 3D checkerboard
  (parity of `floor(s*x) + floor(s*y) + floor(s*z)`), Y-axis stripes,
  smooth `clamp(...)` gradient, and Murmur-style integer-hash noise
  (same `0x9E3779B9 / 0x85EBCA6B / 0xC2B2AE35` constants as the CPU
  IR path). The pattern type, scales, and the two colours come from
  `Material::pattern` etc., so the GLSL output matches the CPU JIT
  output pixel-for-pixel. 5 new tests check emit well-formedness
  (presence of `mix`, `floor`, `clamp`, the noise hash constants) and
  an end-to-end render that verifies the checker produces both
  primary and secondary colours.
- **MeshSDF GPU support** — MeshSDFNode now translates to GLSL by
  appending its voxel grid into a shared `mesh_voxels` array (returned
  from `GlslEmitter::emit()` alongside the source) and emitting a
  per-mesh `sample_mesh_<i>` function that does trilinear interpolation
  against a `std430` storage buffer bound at descriptor set 0,
  binding 1. `VulkanCtx::create()` gained an optional
  `std::vector<float>& mesh_voxels` parameter — when non-empty, it
  allocates a host-visible storage buffer, uploads the data once,
  and writes the descriptor binding. The shader's `sample_mesh_<i>`
  also reproduces the CPU side's outside-bbox correction
  (`sampled + sqrt(ex² + ey² + ez²)`) so the SDF stays Lipschitz
  everywhere. 3 new tests cover storage-buffer emission, no-buffer
  fallback when the scene has no meshes, and an end-to-end GPU render
  of a voxelised sphere.
- **Live GPU viewport** — the Qt viewport gained a `set_gpu_mode(bool)`
  toggle. When on, `do_render()` routes through a new `do_render_gpu()`
  helper that emits GLSL, compiles, creates a `VulkanCtx`, dispatches,
  and blits the result into the same `QImage` the CPU path uses, so
  the rest of the widget (camera orbit, picking, image export) stays
  identical. The Vulkan context is cached and only rebuilt when the
  scene's structural hash changes (parameter edits keep the cached
  pipeline). Failure paths — missing Vulkan, missing glslang, shader
  compile error — fall back to CPU gracefully and surface a status
  message. The main window's new `&Render` menu exposes a checkable
  "Use GPU (Vulkan compute)" action. On the sandbox's `llvmpipe`
  software driver, GPU mode delivers an ~80 ms render of a typical
  scene at viewport size.
- **Image texture maps via triplanar projection** — Materials gained
  a `Texture` pattern with an embedded `texture_rgba` byte buffer plus
  width/height. A new BMP loader (`core/io/bmp_loader.hpp`, header-only,
  no external deps) handles 24- and 32-bit uncompressed BMP files in
  both orientations. The GLSL emitter concatenates all material
  textures into a single uint32-packed storage buffer (binding = 2) and
  emits two helper functions per texture: `_sample_tex_<i>_uv(uv)`
  reads a pixel by `fract(uv) → (u,v) → index` and unpacks the
  RGBA8 word, and `triplanar_sample_<i>(p, scale)` computes the
  surface normal via central differences over `scene_sdf` then blends
  three planar samples by `pow(|n|, 4)` axis weights (the pow-4
  weighting makes axis-aligned regions stay clean and confines blending
  to a narrower diagonal band). Because SDFs lack native UV coordinates,
  triplanar is the natural mapping for arbitrary geometry. PNG loading
  is also supported via libpng when present (`png_loader.hpp` plus the
  `load_image()` dispatcher that picks the right loader by extension).
- **PBR with multi-light Cook-Torrance on GPU** — the GLSL emitter
  produces a full microfacet BRDF in the auto-generated shader: GGX
  normal distribution (`_D_GGX`), Smith/Schlick geometry term
  (`_G_Smith`), and Schlick Fresnel (`_F_Schlick`). The shader loops
  over up to four point lights packed into push constants
  (`pc.lights[4]`), accumulating contributions with energy
  conservation. Per-object roughness/metallic come from a parallel
  `scene_pbr()` function that mirrors `scene_albedo()`, so each
  material gets its own surface response (rough red vs. polished
  metal vs. mid-gloss blue all render correctly in the same scene).
  The redesigned `ShaderPush` struct also drove a `shader_push_builder.hpp`
  helper that centralizes camera frame + light setup, replacing ~150 LoC
  of duplicated code across viewport, gallery, benchmarks, frep_gpu_render,
  and tests.
- **Plugin GPU support + CustomExpr on GPU** — `FRepNode` gained an
  `emit_glsl()` virtual method (default returns false). The GLSL
  emitter's dispatch switch has a `default:` branch that recursively
  emits child expressions then asks the node itself for its GLSL via
  `emit_glsl()`. Combined with a new `NodeKind::Plugin` enum value,
  this lets plugin authors ship GPU support alongside CPU codegen:
  the `capsule_plugin.so` example overrides `emit_glsl` to produce
  the capsule SDF as a GLSL expression, and gets rendered through
  the regular Vulkan compute pipeline. The same mechanism gives
  `CustomExprNode` (runtime-text math expressions) free GPU
  execution — the gallery includes a gyroid surface
  `sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)` rendered at GPU
  speed from a string literal. Required CMake adjustments:
  `ENABLE_EXPORTS ON` on `frep_designer` and `--whole-archive`
  linkage so the host exports its `FRepNode::codegen_grad` base
  symbols to dynamically loaded plugins.
- **Cook-Torrance PBR shader** — physically-based microfacet BRDF (GGX
  normal distribution, Smith-GGX geometry, Schlick Fresnel). Honours
  per-material `roughness` and `metallic`. Energy-conserving (verified
  in tests). Blinn-Phong remains available as a fast alternative; the
  GUI exposes a shading-model selector.
- **SSAA anti-aliasing** — 2x2 or 3x3 supersampling at the tile scheduler
  level (renders the scene at `W*ssaa x H*ssaa`, box-filters to size).
  GUI selector under "SSAA" in the render panel.
- **Forward-mode AD in LLVM IR** — mathematically exact normals (3 SDF
  evaluations instead of 6 finite-diff), verified against analytic solutions
- **BVH acceleration** for scene_material — O(log n) instead of O(n) for many objects
- **IncrementalCompiler** — scene-hash caching, reuses JIT-ed code
- **Dynamic plugin loading** — dlopen-based loading of .so plugins
- **Scene serialization** — save/load scenes as JSON (.frep files)
- **Plugin-aware scene loading** — `load_scene(path, &registry)` restores plugin
  primitives (Torus, Octahedron, Capsule, ...) by looking up the registered type
  name in the PluginRegistry; mixed scenes (built-in + plugin nodes nested
  inside transforms) round-trip correctly
- **`structure_hash` on FRepNode/SceneGraph** — structure-only hash that
  ignores parameter values; lets the IncrementalCompiler distinguish
  "tree shape changed" from "only sliders moved", reported in the GUI
  status bar ("params only") and ready to be exploited by a future
  per-parameter incremental path
- **Node graph editor** — visual editing of the FRepNode tree (QGraphicsScene),
  drag-to-connect ports, cycle detection, auto-layout
- **Ray-cast object selection** — clicking in the viewport selects an object
  (the JIT-compiled `scene_pick` function casts a ray and returns the hit object's index)

## Build

### Dependencies (Ubuntu 24.04)

The tested toolchain is **clang-20 / LLVM 20**. Newer LLVM (21/22) also works —
the IRBuilder API drift is absorbed by `core/compiler/llvm_compat.hpp`.

```bash
# LLVM 20 ships in the Ubuntu 24.04 repos; or add apt.llvm.org for a specific
# version. Install the dev packages:
sudo apt install -y cmake clang-20 llvm-20-dev libclang-20-dev \
                    libgtest-dev libgmock-dev libvulkan-dev \
                    ocl-icd-opencl-dev glslang-tools

# GUI (optional):
sudo apt install -y qt6-base-dev

# SPIR-V tools (optional):
sudo apt install -y spirv-tools spirv-headers
```

### Compilation

```bash
cd frep
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_C_COMPILER=clang-20 \
    -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build build -j$(nproc)
```

### Newer LLVM (21/22)

The API used by frep (IRBuilder, PassBuilder, LLJIT, opaque pointers) is
stable from LLVM 17 onward. The LLVM 22 IRBuilder API changes (FMFSource
parameter in CreateMinNum/MaxNum and the unary/binary intrinsic helpers)
are absorbed by `core/compiler/llvm_compat.hpp`, so the code builds on
LLVM 21/22 as well — point `-DLLVM_DIR` and the compilers at that version.
For strict mode (FATAL_ERROR when LLVM < 22):

```bash
cmake -B build -S . -DFREP_STRICT_LLVM_VERSION=ON ...
```

### Running

```bash
# Basic renderer: scene → IR → JIT → PPM
./build/frep_render demo.ppm 1920 1080

# Scene with custom expressions and plugin primitives
./build/frep_advanced advanced.ppm 800 600

# + SPIR-V emit (writes a .spirv.ll file)
./build/frep_advanced advanced.ppm 800 600 --spirv

# Multipath: render across several paths, compare or composite, with energy
./build/frep_multipath examples/01_csg_basic.json --paths cpu_ir,gpu_glsl --energy

# Convert PPM → PNG (requires ImageMagick)
convert demo.ppm demo.png
```

### Tests

```bash
cd build && ctest --output-on-failure
# Full suite passes (GoogleTest suite + the node_graph_logic headless test)

# Or just the GoogleTest suite directly:
./build/frep_tests

# The node graph logic test separately (requires an offscreen Qt):
QT_QPA_PLATFORM=offscreen ./build/test_node_graph
```

## Architecture flow

```
SceneGraph (FRepNode tree)
   | VPL program
   | - native nodes (Sphere, Box, ...)
   | - plugin nodes (Torus, Octahedron — PrimitivePlugin)
   | - CustomExprNode("sqrt(x*x+y*y+z*z) - 1.0")
   v
SceneCodegen::emit_render_tile()
   |
   |- for every FRepNode::codegen() emit LLVM IR recursively
   |  |- for CustomExprNode  -> CustomExprCompiler parses the expression
   |  |  and emits a separate function in the module, which is then called
   |  |- for PluginPrimitive nodes -> the same codegen() path
   |  '- for native nodes -> direct IR emit
   |
   |- emit_scene_sdf      -> float scene_sdf(x,y,z)
   |- emit_scene_normal   -> forward-mode AD (3 sdf calls)
   |- emit_scene_material -> per-object albedo lookup
   |- emit_shader         -> Phong (ambient + diffuse)
   '- emit_tracer         -> sphere tracing loop
   |   one llvm::Module
   v
   .---------------------+----------------------------.
   |                                                  |
   v JIT path                                         v Retarget path
JitEngine::load()                              SPIRVRetarget::retarget()
   PassBuilder O3 -> inlining                 Modify triple -> spirv64-unknown-unknown
   LLJIT -> x86 native code                   Emit IR with SPIR_KERNEL calling conv
   | RenderTileFn                                  | .spirv.ll file
   v                                                |
TileScheduler::render()                              v
   std::jthread tiles                          (external) llvm-spirv translator
   |                                                |
   v                                                v
PPM image                                       .spv binary -> Vulkan
```

## F-Rep* semantics

In this implementation (F-Rep*), the sign convention is **`f(X) <= 0` means inside**.

| Operation | Formula |
|-----------|---------|
| Union (A v B) | `min(fA, fB)` |
| Intersection (A ^ B) | `max(fA, fB)` |
| Difference (A \ B) | `max(fA, -fB)` |
| Negate (!A) | `-fA` |
| SmoothUnion (k) | IQ smin: `mix(fB, fA, h) - k*h*(1-h)/2` |

The Sphere is implemented as a **true SDF**: `f = sqrt(x^2+y^2+z^2) - r` —
the value equals the Euclidean distance, which makes sphere tracing
correct and fast.

## Performance (Intel reference machine)

| Scene | Resolution | Codegen | JIT O3 | Render | Total |
|-------|------------|---------|--------|--------|-------|
| Demo (5 objects) | 800x600   | ~1 ms | ~55 ms | ~700 ms | ~760 ms |
| Demo (5 objects) | 1280x720  | ~1 ms | ~55 ms | ~1.4 s  | ~1.5 s  |
| Demo (5 objects) | 1920x1080 | ~1 ms | ~60 ms | ~2.7 s  | ~2.8 s  |

JIT compilation happens once; every subsequent frame is render-only.
IncrementalCompiler caches the JIT-ed code by scene hash — if the scene
has not changed, the next `compile()` is free.

## Capabilities

**GUI (Qt6)**
- Viewport with real-time preview (async render thread, orbit camera); a single
  path uses its most efficient backend (GLSL → real-time Vulkan swapchain,
  CPU/CUDA → offscreen), several paths composite in the ExecutorViewport
- Unified checkable path picker (tick 1+ paths) mirrored in the Render menu,
  with a Multi-view layout submenu (split / weighted / tiles)
- Scene inspector — object list, property editor (color, visibility)
- Toolbar for adding primitives (including plugin-based ones)
- File → Open/Save — JSON serialization of scenes
- IncrementalCompiler — scene-hash caching of JIT-ed code
- Ray-cast object selection — a click casts the JIT-compiled `scene_pick` ray
- Node graph editor (QGraphicsScene) — visual FRepNode tree, drag-to-connect
  ports, cycle detection, auto-layout

**Plugins + expressions**
- Plugin API with C++23/26 concepts + PluginRegistry
- CustomExprNode (recursive descent parser → LLVM IR)
- Dynamic plugin loading — dlopen-based loading of .so plugins
- Forward-mode AD in LLVM IR — exact normals (verified)

**Four retargeting paths** (same model, heterogeneous executors)
- **CPU_IR** — LLVM IR → ORC JIT → native x86. The ground truth.
- **GPU_IR** — same IR → NVPTX → PTX → CUDA driver. (Transcendentals are
  inlined minimax polynomials: NVPTX JIT can't resolve libdevice externs.)
- **GPU_GLSL** — independent GLSL emitter → SPIR-V → Vulkan compute.
- **GPU_RTX** — RT shaders lifted from the GLSL emitter → Vulkan ray tracing,
  with per-CSG-group BLAS so the RT cores do real broad-phase culling. Full
  feature parity (17/17 scenes) hardware-confirmed on an RTX 2080.

**Acceleration + measurement**
- BVH for scene_material; the scene-BVH node buffer is exposed std430-ready for
  GPU upload, with the CPU crossover measured (`frep_bench` §6)
- Soft shadows + ambient occlusion (emitted in LLVM IR; shadows toggleable)
- RT pipeline cache amortizing setup across interactive frames
- Energy as Mpix/kWh from real counters (CPU RAPL + GPU NVML), in both
  `frep_rtx_bench --energy` and `frep_multipath --energy`
- Distributed render over TCP (master/worker, pull + push schedulers), confirmed
  on a real two-machine LAN

**Throughput framing.** Per-path Mpix/s and their heterogeneous sum — not a
"speedup" ratio, which is misleading across non-commensurable CPU/GPU cores.
Energy (Mpix/kWh) and cost (pix/$) are separate axes. See `docs/MULTIPATH.md`.

## Possible future work

- **Per-function incremental** — the cache is whole-module; per-function
  granularity via `JITDylib::remove()` would be finer.
- **Scene-BVH GPU traversal** — the upload buffer is ready and the crossover is
  measured; wiring it in also needs per-object SDF functions in the GLSL emitter
  (today one monolithic `scene_sdf`). Only pays off for many-object scenes.
- **RTX real-time swapchain** — RTX renders through the offscreen executor with
  an amortized pipeline cache; a dedicated RT→swapchain renderer would remove the
  readback, but needs the RT pipeline to run on Qt's Vulkan device (RtxCtx owns
  its own today).
- **FPGA retarget** — would require Vivado HLS; out of scope for this PoC.
