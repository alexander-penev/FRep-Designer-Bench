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

For the PTX the GPU-IR path hands to the CUDA driver, set
`FREP_DUMP_PTX=/tmp/scene.ptx` and run any GPU-IR render; `ptxas -arch=sm_75
-v /tmp/scene.ptx` then reports registers, spills and local-frame size.
`FREP_NVPTX_ARCH=sm_75` overrides the PTX target (default `sm_50`), and
`FREP_NVPTX_NO_O3=1` skips the middle-end pipeline on the GPU module for an
A/B against the backend-only lowering.

Where a frame's time goes, per path, outside the kernel: `FREP_EXEC_PROFILE=1`
prints one line per frame from the GPU-IR and GPU-GLSL executors (whether
codegen/emit was skipped, compile, run, and the launch / submit-to-signal /
readback split inside run); `FREP_RTX_PROFILE=1` does the same for the RT
path. Under either flag the Vulkan paths also record GPU timestamps around
the dispatch (or trace) and the image-to-buffer copy that follows it, and
print the two — the submit-to-signal time they otherwise report holds both
plus the queue's latency, and the two parts are very unequal: at 512² the
copy of the 1 MB frame across PCIe is 0.068 ms on this card whatever the
scene, the compute 0.025 ms for the sphere and 0.2 ms for the colonnade,
and the host's memcpy out of the mapped buffer another 0.07 ms, so a GPU
frame of a simple field is three quarters transfer. The benchmark
`frep4_pathcost` separates `run` (the executor's own timer) from `wall` (the
whole `render()` call) — the difference is host work the executor does
around the GPU — and `FREP_PATHS=gpu_ir,gpu_rtx` restricts a run to the
named paths.

Two things to know before trusting a sub-millisecond GPU number. The Vulkan
paths spin on the frame fence before they block (`core/gpu/vk_fence_wait.hpp`):
with a plain blocking wait the core idles between frames, the frequency
governor lowers its clock, and the same frame measures two to three times
longer and drifts over a run; `FREP_VK_WAIT=block` restores the blocking wait
for a comparison. And `frep4_pathcost` warms each path for a time window
(`FREP_WARM_MS`, default 500 ms) before it times anything, because the GPU's
boost clock takes a few hundred milliseconds to ramp from idle and three
frames of a sub-millisecond path would measure the ramp.

Two more things the measurements turned out to hinge on. A two-socket host
puts the card on one socket's PCIe root, and a process the scheduler places
on the other pays an inter-socket hop on every readback — 15–48% of a 512²
frame here. Each GPU context now pins the thread that creates it to the
card's NUMA node (`core/gpu/gpu_numa.hpp`; `FREP_GPU_NUMA=0` disables), and
the CPU render pool's workers reset themselves to the launch affinity, so
the pin never spreads to the CPU path; `paper_numbers.sh` also binds its GPU
runs with `numactl`. And the energy floor: the card sits in P5 (≈27 W) for
about ten seconds after a load before dropping to P8 (≈14 W), so a baseline
taken too soon under-subtracts by 13 W — 9 mJ per frame at 1500 frames per
second. `frep4_pathcost` waits 12 s before taking it.

And two the CPU path turned out to hinge on, both found with
`FREP_CPU_PROFILE=1`, which prints where the host side of each CPU frame
goes (hash, result allocation, render). A result buffer is a `std::vector`,
and a vector zero-fills: the 4 MB frame at 512² was zeroed on the calling
thread over cache lines the workers had just written, 0.5–0.9 ms per frame,
and the render then paid again to take the lines back. Result buffers are
recycled now (`core/tracer/pixel_buffers.hpp`: `TileResult`'s destructor
keeps a few large ones, the executors take them back sized but not
initialised, and every path writes its frame or tile in full), which took
the sphere's CPU frame from 1.85 to 0.58 ms and the GPU paths' 1 MB RGBA8
results along with it. And the process-wide render pool used to broadcast
one condition variable to wake 13 of its 55 workers; the other 42 woke to
find they were not needed, on whatever core the kernel chose — in a
heterogeneous render, the core the GPU thread was spinning on. Each worker
now has its own condition variable and spins 150 µs before blocking, so
sub-tiles and queue tiles that arrive back to back cost no futex round
trip: the CPU path on 128-px tiles went from 29–32 to 45–48 Mpix/s at 14
threads.

Composition, measured with that in place (`frep4_tilequeue_bench`, 800²,
128-px tiles, CPU_IR and GPU_GLSL on one queue, five interleaved rounds):
where the CPU path is a quarter to a third of the GPU's speed per tile
(sphere, CSG difference, blend, twisted bar) the combined run delivers
89–101% of the two devices' sum, 12–31% more than the GPU alone; where it
is a sixth or less (gyroid, gear, colonnade) the combination lands *below*
the GPU alone — by 2–17% under the "fastest takes the tail" rule
(`QueuePolicy::FastestTail`, what `Auto` means) and 12–27% under the plain
queue. The tail rule is 5–16% ahead on those unbalanced scenes and 1–8%
behind on the balanced ones, where tile cost varies thirtyfold across a
frame and the last tiles in row-major order are the cheap background ones
a competitive CPU could have taken; the two are within the bench's
run-to-run spread there. What neither rule recovers is paid throughout the
frame, not at its tail: the GPU's host-side tile work (fifty microseconds
of submit, fence wait and readback per 128-px tile) runs slower while the
CPU band works, whichever cores either side is given. The bench reports
both policies (`hetero_queue`, `hetero_plain`).

Knobs for A/B comparisons, all read from the environment:

| Knob | Effect |
|---|---|
| `FREP_FAST_MATH=0` | no relaxed FP flags (contract/nsz/arcp/reassoc) on the IR paths |
| `FREP_VK_WAIT=block` | plain `vkWaitForFences` instead of spin-then-block |
| `FREP_GPU_NUMA=0` | do not pin GPU threads to the card's socket |
| `FREP_BBOX_CLIP=0` | no ray-box clip on any path (IR, GLSL, RTX) |
| `FREP_CUSTOMEXPR_FD=1` | finite-difference normals for CustomExpr instead of the analytic dual |
| `FREP_NORMAL_METHOD=analytic\|adir\|fd` | (pathcost, scene_parity) how the IR paths differentiate a CustomExpr |
| `FREP_CULL_SLABS=n` | (pathcost, scene_parity) per-tile cull with n slabs |
| `FREP_DEBUG_BOUNDS=1` | print the bounds inferred for each CustomExpr, with the cell count and time the search took |
| `FREP_CPU_PROFILE=1` | one line per CPU frame: content hash, result allocation, render (ms) |
| `FREP_RTX_CELLS=n` | RT path: build the acceleration structure from the field's shell cells, n per axis (default 1, the single box; see the RT section) |
| `FREP_RTX_MODE=query` | RT path: march inline in a compute shader with `rayQueryEXT` against the same structure, instead of in an intersection shader |
| `FREP_PATHS`, `FREP_WARM_MS` | pathcost: which paths, how long to warm up |
| `--format float\|rgba8` | pathcost: the TileResult payload (default: each path's fastest); `float` on a GPU path renders into an rgba32f image, so the floats come back unquantised (four times the bytes) |
| `--policy plain\|tail\|guided`, `--min-ms` | tilequeue bench: what Auto resolves to (default guided, see above), energy window; every policy is measured |

The same choices are `TracerConfig` fields for code: `output_format`,
`normal_method`, `bbox_clip`, `cull_slabs`, and `QueuePolicy` on the
multi-path runner. Measured on the benchmark set at 512²: the tile cull
(8 or 16 slabs) is within noise on every path and scene, so it stays off by
default; the ray-box clip is worth 5–12% on the GPU paths and, on the CPU
path, between a third and two thirds of the frame for the bounded scenes
(the unbounded colonnade is unaffected). The clip used to measure the
other way on the CPU — 17–56% *slower* with it than without, on every
bounded scene, first noticed as "+25% on the twisted bar" — and the cause
was not the march, which the clip shortens on every ray, but the clock.
The clip makes every sky tile nearly free, so with 64-px tiles a 512²
frame is mostly idle workers; the frequency governor parks their cores at
the idle clock, and the one dense tile that decides the frame (about a
thousand shaded pixels, shadow and ambient occlusion) lands on a cold core
and runs there at 1.2 GHz, 3× slower than hot, while without the clip the
sky work kept every core clocked up. The tile heuristic in
`TileScheduler` assumed uniform tile cost; it now halves the tile until
there are sixteen per thread rather than four (the 16-px floor stays), so
the dense tile is four tiles on four cores, and the clip is faster than
no clip everywhere. The same change removed the drop the CPU path showed
at 1280² (3.6 ns per pixel against 2.0 at 800²): per-pixel cost now falls
with resolution, 3.0 → 2.0 → 1.6 ns at 400², 800² and 1280².

The bounds search itself is a few hundred interval evaluations for every
scene here, the twisted bar included. It used to be 88 000 and 80 ms for
that one, because the interval of x·cos(1.2z)+y·sin(1.2z) over a cell
whose z turns the angle through a quarter turn straddles zero wherever the
cell is; no per-coordinate interval, however exact, discards a cell far
from the bar. Two rules in every interval evaluator (the CPU-IR codegen,
the numeric twin in `node_interval.hpp`, the GLSL emitter and the bounds
search's own) fix it: a rotation pair a·cosθ ± b·sinθ is bounded by
±√(a²+b²) over the box, and max(|P|+c₁, |Q|+c₂) over two such pairs of one
angle with the factors exchanged is bounded below by √((A²+B²)/2) + min(c),
since P²+Q² = a²+b². The bar's box went from ±1 to ±0.5 in x and y, the
search to 521 cells and 0.8 ms, and no other scene's box moved.

### The RT path against the compute path

The RT path renders the same field as the compute path and, on a single
field, has measured slower per frame since the first audit (0.25 against
0.19 ms on the sphere at 512²). Timestamps on both paths put the difference
where it is: the frame image's trip to the host is the same on both (the
0.068 ms PCIe copy and the 0.07 ms memcpy above), the host work around the
submit is within 0.01 ms, and the GPU's own time is 0.057 ms on the RT path
against 0.025 ms on compute for the sphere, 0.070 against 0.047 on the
gyroid, 0.237 against 0.205 on the colonnade — a fixed 0.02–0.03 ms per
512² frame, about 0.1 ns per ray amortised over the card, whatever the
field. That is the cost of one hardware ray cast per pixel (the RTX 2080's
traversal rate is of the order of ten gigarays per second), and a single
box gives the traversal unit nothing to do for it: every ray that enters
the box runs the whole march in the intersection shader exactly as the
compute shader does. A compute march of these fields costs about as much
as one cast, so the RT path cannot win on them by construction, and the
two obvious remedies were built and measured so the statement is not an
argument. Splitting the field into shell cells (`FREP_RTX_CELLS=n`,
`core/gpu/rtx_cells.hpp`: the leaves of an octree over the scene box whose
interval bound straddles zero, so the hardware skips the empty cells and
the march runs one cell at a time) made every scene slower at 8, 16, 32
and 64 cells per axis — sphere 0.057 → 0.066–0.084 ms, gyroid 0.070 →
0.095–0.105, gear 0.080 → 0.118–0.139 — because each cell a ray enters is
another cast and another shader invocation, and on these fields that costs
more than the marching it saves. Running the march inline from a compute
shader with `rayQueryEXT` against the same structure (`FREP_RTX_MODE=query`)
removes the shader-stage transitions and changes nothing within noise
(sphere 0.056, gyroid 0.063, colonnade 0.210 ms; the gear 0.101 against
0.080), which locates the cost in the cast itself rather than in the
pipeline's stages. Both stay in the tree as knobs, off by default, for
fields where the march is the expensive part. Where the structure does
have work to do the RT path pulls ahead: on the structured colonnade (27
instanced columns, `scenes/frep4/c2_colonnade_struct.json`) it renders in
0.357 ms against the compute path's 0.389, and the N-sphere grid of the
instance table scales it to several times the compute path. The images are
the same on every variant: the shell-cell and ray-query renders match the
CPU to the parity floor, with the usual handful of silhouette pixels.

For frame-level CPU profiling, `perf` samples the JIT'd code but cannot
name it: the ORC JIT here registers no perf map, so samples inside
`render_tile` show as raw addresses (calls into libm and the kernel are
still named). Compare counters and A/B runs instead, or inspect the IR
offline: `frep_dump` writes the pre- and post-O3 `.ll`, and `llc
-mcpu=<host>` on the latter shows what the JIT compiles. For GPU
profiling on real hardware use `renderdoc` or the vendor's profiler
(NVIDIA Nsight, AMD RGP).

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
