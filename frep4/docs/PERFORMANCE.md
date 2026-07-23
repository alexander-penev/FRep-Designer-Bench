# FRep Designer — Performance & Memory Analysis

_Measured on the CPU JIT path (LLVM-20, `frep_advanced`, release build),
400×300 unless noted. The CPU path is representative of raymarch cost;
the GPU path shares the same algorithm and the same adaptive-step logic._

## 1. Where render time goes

Benchmark: a sphere∪box CSG plus a ground plane, shadows/AO off to
isolate the march, best of 3 runs.

| Config (max_steps / safety_factor) | Render |
|------------------------------------|--------|
| 128 / 1.00                         | 64.7 ms |
| 128 / 0.85                         | 78.2 ms |
| 192 / 0.85                         | 83.5 ms |
| 192 / 1.00                         | 67.7 ms |
| 256 / 0.85                         | 85.4 ms |

Two findings:

- **`safety_factor` dominates, not `max_steps`.** Dropping the step to
  0.85 costs ~21% (64.7 → 78.2 ms at 128 steps) because *every* ray
  takes ~18% more, smaller steps. Raising the step cap 128 → 192 costs
  only ~7%, since few rays actually reach the cap — most converge well
  before it.
- The cap raise (4.2.11, for interior-silhouette convergence) was
  therefore cheap; the conservative step (4.2.5, for CSG correctness)
  was the real cost.

### Shadows / AO (192 / 0.85)

| Config | Render |
|--------|--------|
| no shadows, no AO | 83 ms |
| shadows only      | 125 ms |
| shadows + AO      | 129 ms |

Shadows are the single biggest cost (+50%): each shaded hit casts a
secondary shadow march per light. AO adds little on top (+3%).

## 2. Optimisation applied — adaptive `safety_factor`

`safety_factor < 1` is only needed where the distance field isn't a
true Euclidean SDF: CSG (union/intersection/difference/smooth-union),
plugins, mesh-SDF, custom expressions, or simply more than one object
(combined with `min()`, itself a union). A scene that is a **single
primitive** — optionally affine/twist transformed — is a true SDF, so a
full step never oversteps.

Both back-ends now detect this per-compile (`node_requires_safety_step`
+ object count) and raise `safety_factor` to 1.0 when it's safe, falling
back to 0.85 otherwise. The grazing-ray rescue still backstops either
choice, so correctness is unchanged.

| Scene | Before | After |
|-------|--------|-------|
| single sphere (true SDF) | ~19 ms | **15 ms** |
| CSG union (needs 0.85)   | 26 ms  | 26 ms (unchanged) |
| two objects (needs 0.85) | 27 ms  | 27 ms (unchanged) |

~20% faster for single-primitive scenes (common while modelling one
shape), zero change — and zero new artefacts — for CSG scenes.

## 3. Memory

Peak RSS (`/proc/<pid>/VmRSS`), default scene:

| Resolution | Peak RSS |
|------------|----------|
| 400×300    | 74 MB |
| 800×600    | 79 MB |
| 1600×1200  | 101 MB |

- The ~74 MB floor is the LLVM/JIT runtime (ORC, the compiled module,
  target machine). Fixed cost of the CPU path; the GPU path avoids it
  but pays Vulkan driver overhead instead.
- Per-pixel growth is linear and expected: the RGBA **float** framebuffer
  is 16 bytes/pixel, so 1600×1200 is ~30 MB — matching the +22 MB
  observed over the 800×600 case.
- No leak or super-linear growth across resolutions.

### SSAA memory note

The supersampled path allocates a scratch buffer at `ssaa²×` the output
pixel count, still in RGBA-float. At 1600×1200 that's ~30 MB × ssaa²:
~120 MB at 2×, ~270 MB at 4×. This is the main memory hotspot, paid only
while a supersampled render is in flight (the scratch is freed after the
box-downsample). A future option would be to downsample in tiles, or to
use RGBA8 for the scratch where the extra float precision isn't needed.

## 4. Remaining opportunities

These are genuine further optimizations, none of them on the critical path for
correctness or the heterogeneous-throughput result:

- **Shadow march cost** (+50%) is the largest single lever. Shadows are already
  a toggle (`TracerConfig::enable_shadows`, off → the shadow function constant-
  folds away), so the cost is opt-out today; reducing it *while on* would mean
  caching/limiting shadow rays or a cheaper soft-shadow approximation at low
  sample counts. Needs care not to reintroduce the penumbra artefacts fixed in
  4.2.6.
- **SSAA scratch in RGBA8** — straightforward memory win on the offscreen path.
- **Adaptive epsilon** by distance to camera — fewer steps for far geometry.
  Interacts with the grazing rescue; needs validation.
- **Scene-BVH on the GPU** — the flattened node buffer is exposed
  (`Bvh::gpu_nodes()`, std430-ready) and the CPU crossover is measured
  (`frep_bench` §6: BVH wins from ~2 objects up), but GPU traversal isn't wired
  in; it also needs per-object SDF functions in the GLSL emitter (today one
  monolithic `scene_sdf`). Only worth it for many-object scenes.

The adaptive-step win was the high-value, low-risk change and is shipped.

## 5. Scalability benchmark suite

A dedicated sweep (`frep_bench --scaling`) charts render time as scene
complexity grows along four independent axes. It runs **standalone on
real hardware** and reports CPU JIT and (when Vulkan is present)
offscreen GPU timings for each scene at each size:

```
frep_bench --scaling             # CPU + GPU (if available), markdown
frep_bench --scaling --paths cpu_ir  # CPU rows only
frep_bench --help                # all options
```

The four axes and their generators (in `tools/benchmarks.cpp`, each
driven by one size parameter so any N is reachable):

- **Object count** — N spheres on a grid (`make_many_spheres`)
- **Node depth** — one primitive in N nested transforms (`make_deep_transforms`)
- **CSG depth** — N chained boolean ops (`make_deep_csg`)
- **Mixed primitives** — N objects cycling every primitive/deform (`make_mixed_primitives`)

### Measured curve (CPU JIT, 800×600, build environment)

These build-environment numbers are slow in absolute terms (a sandbox
software target); the **shape** of each curve is the takeaway, and real
hardware will be far faster in absolute ms.

| Axis | size 10/1 | size 100/10 | size 1000/100 |
|------|-----------|-------------|---------------|
| Object count (render) | 568 ms | 3.3 s | 24 s |
| Node depth (render)   | 259 ms (d=1) | 1.0 s (d=10) | 13.6 s (d=100) |
| CSG depth (render)    | 332 ms (d=1) | 0.58 s (d=10) | 3.4 s (d=100) |
| Mixed primitives (render) | 3.4 s | 23.6 s | (minutes) |

JIT-compile time grows alongside: ~11 s to compile the 1000-sphere
scene, ~0.5 s for depth-100.

### Interpretation

- **Object count scales ~linearly** (568 ms → 3.3 s → 24 s for 10× steps).
  Expected: `scene_sdf` evaluates a flat `min()` over all N objects, so
  every ray-step is O(N). This is the clearest case for a **bounding-
  volume hierarchy** — cull objects whose AABB the ray can't reach and
  the per-step cost drops toward O(log N). The AABB infrastructure
  already exists on `FRepNode`; wiring it into a BVH-accelerated
  `scene_sdf` is the highest-impact future optimisation for large scenes.
- **Node depth scales super-linearly** (worse than CSG depth): each
  nested transform adds work to every SDF evaluation along the whole
  ray, and deformations (twist) add trig per level. Deeply-nested single
  objects are costlier than the same node count spread across a boolean
  tree.
- **CSG depth scales ~linearly** and more gently than node depth — the
  boolean tree is evaluated once per ray-step, and the adaptive step
  already keeps it at 0.85 (correctly, since it's not a true SDF).
- **Mixed primitives** track object count with a constant-factor premium
  for the heavier per-object shapes (twist/smooth-union).

The dominant lesson: **large scenes are bottlenecked on the O(N)
flat-union `scene_sdf`, not on shading**. A BVH is the single change that
would move the object-count curve from linear to logarithmic.

### Hardware results (NVIDIA GTX 1050 Ti, 800×600)

Running `--scaling` on real hardware revised the picture in an important
way: **render is no longer the bottleneck at scale — compile is.**

- **GPU render beats CPU render 6–24×** across every scene, and both
  scale ~linearly with object count (no BVH on either path yet). At 1000
  spheres: CPU render 2.9 s vs GPU render 0.48 s.
- **CPU JIT explodes super-linearly.** 49 ms → 422 ms → 22.9 s for
  10/100/1000 spheres; 139 s to compile the 1000-object mixed scene.
  Codegen stays trivial (<70 ms even at 1000) — it's LLVM's O3 lowering
  of one huge unrolled function that costs, not IR construction.
- **GPU init blows up on large shaders.** init was 0.3–0.9 s for small
  scenes but **21.6 s at 1000 spheres and ~23 minutes for mixed-1000**.
  This is almost certainly the driver compiling an enormous unrolled
  SPIR-V shader to native code at `vkCreateComputePipelines` time. The
  benchmark now splits init into phases (device / shader-module /
  **pipeline** / buffers / misc) to confirm this directly — the pipeline
  column is the driver compile.
- **Memory confirms two very different ceilings.** CPU peak RSS grows
  with the JIT module: 150 MB → 853 MB (mixed-1000). GPU device memory
  stays ~1 MB regardless of object count — objects are unrolled into the
  shader program, not stored in buffers — so the GPU's limit is *compile
  time*, not memory.

**Revised priority.** The earlier "BVH is highest-impact" conclusion held
when render dominated; the hardware data shows the wall at ~1000 objects
is **compilation** (CPU LLVM O3, and the GPU driver's pipeline compile),
not per-frame render. The higher-leverage architectural change is to stop
unrolling N objects into one giant function and instead **iterate objects
in a loop over a per-object parameter buffer** (as MeshSDF already does
for voxels), making the shader/IR a fixed size independent of N. That
attacks the CPU JIT explosion and the GPU driver-compile blow-up at once.
A BVH remains useful for render at high object counts but is now
secondary.

### Why CPU JIT can't be made to scale (three dead ends)

After the hardware data pinned the wall at ~1000 objects on the CPU JIT
(≈23 s to compile, codegen trivial), three approaches to taming it were
measured and **all three failed**:

1. **Lower the optimisation level** (`--opt-sweep`). O0/O1/O2/O3 differ by
   <3% in compile time at 1000 objects (O1 22.6 s vs O3 23.2 s), and
   render is essentially identical across levels (the raymarch is
   branch/memory-bound, so O3's vectorisation buys nothing). O0 is even
   *slower* to compile (naïve IR overwhelms instruction selection).
2. **Split one giant function into N per-object functions**
   (`--func-split`). Compile time was unchanged (23.2 s inlined vs 23.6 s
   split at 1000 objects — within noise) and render was **3× slower**
   (call overhead per march step). So the super-linear cost is **not** the
   size of a single function.
3. (Implied) Since neither the optimisation passes nor the function
   structure is the cause, the cost is the irreducible work of lowering
   ~10 000+ instructions (the 1000 inlined object SDFs) to machine code —
   instruction selection and register allocation over that aggregate,
   regardless of how it's partitioned into functions.

**Conclusion: the CPU JIT path does not scale to thousands of objects,
and no codegen-structure change fixes it.** The right answer is to route
large scenes to the **GPU**, which the same hardware run shows handles
them well: mixed-1000 compiles in ~0.6 s (glslang) + ~0.3 s (init, driver
pipeline compile ≤16 ms) and renders in ~6 s, with device memory ~1 MB
(objects unroll into the shader program, not buffers). The GPU's only
"cost" at scale is per-frame render, which scales linearly and stays
tractable.

Practical guidance, now documented for users: **CPU JIT is ideal for
interactive editing of small/medium scenes (≤~100 objects, sub-second
compile); switch to GPU mode for large scenes (hundreds to thousands of
objects)**, where compile stays under a second and render is an order of
magnitude faster than the CPU. The unified render-mode selector (4.3.0)
makes this a one-click switch. A future convenience would be to suggest
the switch automatically past an object-count threshold.

## 6. Spatial acceleration (BVH) for the nearest-distance query

Render scales linearly with object count on both back-ends because the
scene SDF is a flat `min()` over every object at each march step. A
bounding-volume hierarchy over the objects' AABBs turns that into a
roughly logarithmic query.

**Why it's valid for SDFs.** Sphere-tracing needs the true nearest
distance, so we can't cull objects the way ray-triangle tracing does.
But the distance from the query point to an object's AABB is a *lower
bound* on the distance to that object's surface, so any object whose box
is farther than the best distance found so far cannot be the nearest and
can be skipped — without changing the result. A BVH applies that test
hierarchically, visiting the nearer child first so the bound tightens
fast and whole subtrees prune in one test.

**Measured (prototype, 50k random queries over spread objects):**

| Objects | brute min() | flat AABB prune | BVH | BVH vs brute |
|---------|-------------|-----------------|-----|--------------|
| 10   | 25 ms  | 11 ms | 9 ms  | 2.7× |
| 100  | 258 ms | 43 ms | 18 ms | 14× |
| 1000 | 2569 ms| 270 ms| 32 ms | **80×** |

BVH render cost is nearly flat (18→32 ms for 100→1000 objects) — the
logarithmic behaviour we wanted — while the flat scan stays linear.

**Two correctness subtleties found while building it:**

1. The prune is only sound for *exterior* queries. AABB distance is
   always ≥ 0, but inside an object the SDF is negative, and an
   overlapping object can be more negative (nearer). So pruning is
   disabled once the best distance goes ≤ 0; the hierarchy is still
   walked, just not pruned, in interiors. Sphere-tracing rarely samples
   deep interiors (it stops at surfaces), so exterior/near-surface points
   — the vast majority — keep the speedup.
2. Building this surfaced a latent bug: `BoxNode`'s CPU eval and JIT
   codegen used `max(dx,dy,dz)` — the Chebyshev (L∞) distance — which
   under-estimates the distance to a far corner and so violated the
   AABB lower-bound invariant. (The GLSL path was already correct.) Fixed
   to the true Euclidean box SDF `length(max(d,0)) + min(max(d),0)` on all
   three paths; this also makes sphere-tracing of boxes take correctly
   sized steps and brings CPU/GPU box rendering into agreement.

**Status.** The host-side `frep::accel::Bvh` (core/accel/bvh.hpp) builds
the hierarchy and answers eval-based queries, verified identical to
brute force across primitive types, spread/overlapping layouts, unbounded
planes, and CSG units (8 tests). Integrating it into the JIT and GPU
render paths — where the generated `scene_sdf` is static code rather than
a tree walk — is the next step and a larger design question (a static
shader doing a dynamic stack-based traversal), tracked separately.

### Build-time spatial guards in the JIT (approach 1)

The first integration approach materialises the flat prune directly in
the generated code, no runtime hierarchy: each object becomes a
non-inlined function gated by an inline AABB-distance test against the
running best distance (`emit_scene_sdf_guarded`). It works on the JIT
path today (and ports to GLSL later) because it needs no dynamic stack.

Measured on the JIT path (full render_tile, spread spheres, 400×300, best
of 2; the AD-gradient SDF used for normals is still inlined, so this is a
lower bound on the achievable speedup):

| Objects | inline render | guarded render | speedup |
|---------|---------------|----------------|---------|
| 65   | 2296 ms | 1383 ms | 1.7× |
| 100  | 250 ms  | 147 ms  | 1.7× |
| 300  | 751 ms  | 405 ms  | 1.9× |

**But on real hardware (NVIDIA NUC, x86), guarding bare spheres does NOT
help** — measured 0.7× / 1.0× / 0.9× at 11 / 101 / 1001 spheres. The
sandbox is a software-rendering CPU with no real branch prediction or
SIMD; on real x86 the inline `min()` chain vectorises (AVX) and is
branchless, while the guard adds a data-dependent branch and a `sqrt` per
object. For a bare sphere — whose SDF is a single subtract + length — the
guard costs more than the evaluation it skips, and breaks the
vectorisation. The flat prune that won 80× in the eval-based prototype
loses once the baseline is inlined and vectorised.

**The guard's payoff depends entirely on per-object SDF cost.** Re-run
with expensive objects (a twisted box smooth-unioned with a sphere —
deformation + CSG per object) and the result inverts: **2.2× at just 9
objects**, where bare spheres gave 0.7×. Skipping a costly SDF saves far
more than the guard's sqrt+branch; skipping a near-free one doesn't.

**Conclusion.** Build-time spatial guards are not a universal win — they
*hurt* simple-primitive scenes on real hardware and *help* CSG/deform-
heavy ones. So they belong behind an adaptive heuristic (enable only when
the average per-object SDF is expensive), never unconditionally. For
simple primitives at scale, the vectorised inline `min()` is already hard
to beat on the CPU, and the GPU remains the answer for large counts. The
guard machinery stays available (`SceneSdfMode::Guarded`) for the
heavy-object niche it genuinely helps.

### Runtime calibration (making the niche automatic)

Because the crossover (which per-object cost makes guarding win) is
CPU-dependent, it's measured on the host rather than hardcoded.
`frep::accel::calibrate()` renders a few tiny inlined-vs-guarded scenes at
increasing per-object complexity and finds the lowest node count where
guarding wins; the result caches to disk (keyed by CPU model, so a moved
machine recalibrates) and loads instantly thereafter (~200 ms to measure,
once). `should_guard(cal, object_count, avg_node_count)` then enables the
guarded path only when a scene has enough objects (≥8) and their average
node count clears the calibrated threshold. `IncrementalCompiler` wires
this in behind `set_spatial_guards_enabled(true)` (off by default; the
inlined path stays the safe baseline). Inspect or refresh the cached
threshold with `frep_bench --calibrate` / `--recalibrate`.







### Running the suite on real hardware

The benchmark links against the same core + GPU pipeline as the app, so
it measures the real renderers. Build it (it's a normal target):

```
cmake --build build --target frep_bench
```

Then run any of:

```
./build/frep_bench --scaling              # CPU + offscreen GPU, full sweep
./build/frep_bench --scaling --paths cpu_ir   # CPU JIT only
./build/frep_bench --smoke                # tiny sizes, ~1 min sanity check
./build/frep_bench                        # the standard fixed-scene suite
./build/frep_bench --json                 # machine-readable (perf_check.py)
```

On a machine with hardware Vulkan, `--scaling` adds **GPU emit+compile**
and **GPU render** columns next to the CPU ones for every scene/size, so
the same sweep compares both offscreen back-ends directly. `--smoke`
runs every generator at tiny sizes and is the quickest way to confirm
the whole suite still builds and renders after a change.

### Timing breakdown — compile vs execute

Every row separates the one-time compile from the per-frame render, on
both back-ends, so you can see where the cost lands:

- **CPU**: `codegen` (build the LLVM IR from the scene) and `JIT` (lower
  that IR to native via the O3 pipeline) are the two compile phases;
  their sum is the one-time compile. `render` is what repeats per frame.
  In practice codegen is tiny (1–2 ms) and JIT dominates the compile —
  LLVM's optimisation/lowering is the cost, not IR construction.
- **GPU**: `emit` (scene→GLSL), `compile` (GLSL→SPIR-V via glslang), and
  `init` (Vulkan context + pipeline + buffer upload) make up the one-time
  build; `render` is the per-frame dispatch + readback.

This matters because the compile cost is paid once per scene edit
(incrementally — only the changed subtree recompiles in the live app),
while render repeats every frame: a scene can have an expensive compile
but cheap frames, or vice-versa, and the split tells them apart.

### Memory — system capacity for large scenes

The sweep also reports memory, to bound how large a scene each back-end
can hold:

- **CPU ΔRSS** — resident-memory growth across compile+render for that
  scene (its working set). **CPU peak** — process high-water mark.
- **GPU host ΔRSS** — host-side growth (driver objects, staging, the
  readback buffer). **GPU device** — *computed* VRAM footprint from the
  buffers we allocate (output image `W·H·4`, ×ssaa² when supersampling,
  plus mesh-voxel and texture storage buffers). Device allocations don't
  appear in process RSS, so this computed figure — a lower bound — is the
  one that bounds GPU capacity as scene/output size grows.

Measured CPU memory vs object count (peak RSS, single-primitive spheres):

| Objects | Peak RSS |
|---------|----------|
| 10  | 71 MB |
| 100 | 80 MB |
| 500 | 143 MB |

The ~70 MB floor is LLVM/JIT; beyond it each object adds IR plus compiled
native code, so memory grows with object count (≈140 MB by 500 objects).
This is the practical ceiling to watch for very large CPU scenes — the
JIT module itself, not the framebuffer, is what grows. On the GPU the
shader is a fixed-size program regardless of object count (objects are
unrolled into one scene_sdf), so device memory tracks output resolution
and any voxel/texture buffers rather than object count — a different, and
usually higher, capacity ceiling.



