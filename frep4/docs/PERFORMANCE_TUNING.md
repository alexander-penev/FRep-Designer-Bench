# FRep Designer 4.0 — Performance Tuning Guide

A guide to picking the right execution path and modeling strategy
for your scene complexity and quality target. The benchmark numbers
in this guide are taken from the build environment (Ubuntu 24,
LLVM 20, Vulkan via Mesa llvmpipe); your hardware will be faster.

For absolute numbers run `./build/frep_bench` to get measurements
specific to your machine.

---

## TL;DR cheat sheet

| Question | Quick answer |
|---|---|
| Default rendering path | GPU (Vulkan compute) — 4-6× faster than CPU |
| When CPU JIT wins | Tiny scenes (one primitive), debugging, no Vulkan |
| Best for high-resolution finals | GPU at 1080p+ |
| Best for interactive parameter tweaking | Either — GPU stays fast at low res |
| Best for editing CustomExpr | GPU — recompile is ~50 ms vs CPU ~200 ms JIT |
| MeshSDF resolution sweet spot | 64³ for editing, 128³ for finals |
| Marching cubes for export | 64³ enough for most prints, 128³ for hero shots |

---

## CPU JIT vs GPU compute

The headline number: at 800×600 resolution, the GPU pipeline is
**4-6× faster than CPU JIT** across all benchmark scenes:

| Scene | CPU @ 800×600 | GPU @ 800×600 | Speedup |
|---|---|---|---|
| Simple (1 sphere) | 260 ms | 64 ms | 4.1× |
| Moderate (CSG diff) | 307 ms | 67 ms | 4.6× |
| Complex (5 objs, deforms) | 1460 ms | 252 ms | 5.8× |
| Heavy (MeshSDF + CSG) | 590 ms | 188 ms | 3.1× |
| CustomExpr (gyroid) | 576 ms | 114 ms | 5.1× |

The speedup grows with scene complexity because the GPU's parallelism
amortizes setup. For trivial scenes the per-frame overheads
(pipeline bind, push constants, readback) become a larger fraction.

### Pick CPU JIT when:

- **No Vulkan available**. The CPU path always works, the GPU one
  requires a Vulkan 1.3+ driver. Software Vulkan (Mesa llvmpipe)
  works but isn't dramatically faster than the JIT.
- **Need exact reproducibility**. CPU is deterministic;
  llvmpipe and real GPU drivers may produce off-by-one pixel
  differences for the same scene.
- **PBR matters more than speed**. The CPU pipeline implements full
  Cook-Torrance with proper GGX/Smith/Schlick. The GPU shader has
  the same model but at lower precision; on llvmpipe especially
  the GPU output can look slightly flatter.
- **You're debugging the rendering math**. CPU traces line up with
  the LLVM IR — easier to step through in a debugger.

### Pick GPU compute when:

- **Resolution > 400×300**. The fixed costs of GPU init (~40 ms) are
  amortized once you have ~100K pixels to render.
- **Final output**. The 4-6× speedup is meaningful for 1080p+ frames.
- **Iterating on materials or lights**. These don't require shader
  recompile (they go through push constants) — GPU stays at near-
  zero latency.
- **Scene contains a MeshSDF**. The voxel grid lives in a GPU
  storage buffer; CPU repeatedly traverses it from main memory.

---

## When recompile cost matters

Both pipelines compile your scene to native code before rendering.
The cost of that compile is non-trivial:

| Pipeline | What gets compiled | One-time cost |
|---|---|---|
| CPU JIT | LLVM IR → x86 via Orc JIT | ~75 ms (Simple), ~200 ms (Complex) |
| GPU compute | GLSL → SPIR-V via glslangValidator | ~35-50 ms (any complexity) |

The incremental recompile cache (see `core/compiler/incremental.hpp`)
hashes the scene graph and only recompiles when geometry actually
changes. Material and light edits flow through push constants without
shader recompilation.

**Tip**: if you're tweaking a single parameter in a tight loop (e.g.
animating a sphere's radius), prefer to do that via the parameter
itself rather than via re-construction of nodes — the recompile
cache will reuse the compiled shader.

---

## CustomExpr — runtime-text expressions

A CustomExpr node defines its SDF via a text expression like
`sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)` (the gyroid). The
text is parsed into a shared AST and consumed by three back-ends:

- **CPU eval**: AST walked recursively, ~50-100 ns per sample
- **CPU JIT**: AST → LLVM IR, eventually as inlined SIMD-able code
- **GPU GLSL**: AST → GLSL string, inlined into the scene shader

A constant-folding pass simplifies the AST before emission:
`pi*x` becomes `3.14159*x`, `pow(2,3)*x` becomes `8*x`, etc.
Hot paths (per-pixel ray-march body) are then as fast as
hand-written code.

**Best practice**: use CustomExpr for surfaces that are *naturally*
expressed mathematically (gyroid, Schwarz P, math art) rather than
those built from primitives. Building "sphere minus box" via
CustomExpr is possible but produces verbose code; use the CSG
operations instead.

---

## MeshSDF — imported triangle meshes

When you import an STL or OBJ, the host builds a triangle BVH and
then voxelizes the mesh into a 3D SDF grid:

| Resolution | Voxel count | Memory | Voxelization time | Quality |
|---|---|---|---|---|
| 32³ | 33K | 132 KB | ~50 ms | Crude — silhouette only |
| 64³ | 262K | 1.0 MB | ~250 ms | Good for editing/preview |
| 96³ | 884K | 3.5 MB | ~600 ms | Production interactive |
| 128³ | 2.1M | 8.4 MB | ~1.4 s | Hero-shot quality |
| 192³ | 7.1M | 28 MB | ~4.8 s | Beyond detail of most imports |

The sparse-octree compression (see
[ARCHITECTURE.md](ARCHITECTURE.md#sparse-octree-compression)) cuts
memory by ~30× when the surface is sparse — for a 128³ sphere it
fits in 280 KB rather than 8.4 MB.

**Picking a resolution**:
- Use **64³** while editing — fast voxelization, good silhouette.
- Use **128³** for the final render — surface detail matches imported
  triangles even at high zoom.
- Beyond **128³** the SDF is already finer than ray-marching can
  resolve at typical viewport resolutions. Use higher resolutions
  only for marching-cubes mesh export where you want a clean
  re-mesh.

---

## Marching cubes mesh export

Both export paths (File → Export mesh, or `mesh::extract_iso_mesh`
in code) sample your scene's SDF on a regular 3D grid. Output
size and runtime scale as O(N³):

| Resolution | Sample count | Time (typical scene) | Output size (OBJ) |
|---|---|---|---|
| 32³ | 33K | 30 ms | 200 KB |
| 64³ | 262K | 250 ms | 1.5 MB |
| 96³ | 884K | 850 ms | 5 MB |
| 128³ | 2.1M | 2.0 s | 12 MB |
| 192³ | 7.1M | 6.8 s | 40 MB |
| 256³ | 16.8M | 16 s | 95 MB |

For 3D printing, anything past 128³ is usually wasted — most slicers
will down-sample to their nozzle resolution anyway. For external
rendering or further mesh processing, 192³ gives a clean re-mesh.

The exporter does central-difference normals on output, so triangle
shading looks correct in any DCC tool you import the mesh into.

---

## Picking the right rendering tile size

The GPU compute shader dispatches in 8×8 thread workgroups. The
runtime tile dimensions are derived from the render resolution
automatically — there's no tuning knob exposed. On llvmpipe the
optimal is 8×8 (matches the local thread group); real GPUs prefer
16×16 but the CPU bottleneck dominates anyway.

The CPU JIT path uses dynamic work-stealing chunks — also tuned
automatically based on `std::thread::hardware_concurrency()`. On
a 12-core machine the typical chunk is 128 rows.

---

## Memory budget at a glance

For a typical "complex" scene (5 objects, one of them MeshSDF at
64³, full PBR with one 256×256 texture):

| Resource | Size |
|---|---|
| Voxel SDF grid | 1.0 MB |
| Texture | 256 KB |
| LLVM IR module | ~150 KB |
| Compiled native code | ~30 KB |
| Vulkan pipeline + descriptors | ~50 KB |
| **Total per scene** | **~1.5 MB** |

Modeling a full architectural scene with 50 MeshSDF objects at 128³
each would need ~420 MB just for voxel grids — past the sweet spot
where pure F-Rep makes sense. For that workload reach for a hybrid
approach with traditional polygon rendering.

---

## Performance regression detection

The project ships a regression-detection script:

```bash
./build/frep_bench --json > /tmp/bench.json
python3 tools/perf_check.py /tmp/bench.json
```

This compares your measurements against the committed baseline in
`tools/perf_baseline.json` and exits 1 if any scene became more
than 2× slower. CI runs this on every PR (currently in
`continue-on-error` mode until baseline noise is characterized).

After intentional performance changes, regenerate the baseline:

```bash
./build/frep_bench --json > tools/perf_baseline.json
git add tools/perf_baseline.json
git commit -m "perf: update baseline after <reason>"
```

---

## Profiling your own modifications

For LLVM IR inspection:

```bash
./build/frep_advanced --dump-ir /tmp/scene.ll
```

For GLSL inspection (saved to `/tmp/frep_glsl_<pid>_*.spv` by default
via `frep_dump --paths gpu_glsl`, which writes the `.glsl` and `.spv`).

For frame-level CPU profiling, `perf` works on the JIT'd code — Orc
emits symbol info that `perf report` can map. For GPU profiling on
real hardware use `renderdoc` or the vendor's profiler (NVIDIA Nsight,
AMD RGP).

---

## Tuning checklist

If your scene feels slow:

1. **Switch to GPU**. The biggest single win.
2. **Lower MeshSDF resolution while editing**. Drop from 128 to 64 —
   you'll get back ~80% of the speed.
3. **Use procedural primitives instead of imported meshes** where the
   shape allows.
4. **Avoid `SmoothUnion` with very large `k`**. Each large smooth
   adds significant work to the SDF evaluation.
5. **Defer marching cubes export to the end of your session**. It's
   expensive and doesn't affect viewport responsiveness.
6. **Pre-compute via marching cubes + MeshSDF**. If you have a scene
   that's slow to evaluate (deep CSG tree, many CustomExprs), export
   it once, then re-import as a single MeshSDF node for further
   editing. You lose parameter editability but gain massive speed.
