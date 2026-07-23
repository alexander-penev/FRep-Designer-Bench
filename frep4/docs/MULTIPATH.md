# Multi-path execution (Model D)

The `core/exec/` framework runs a render across one or more retargeting
paths / executors and recombines the results. It exists to use whatever
compute hardware is available for a single render — CPU and GPU, local and
(later) remote — regardless of each executor's architecture or control
language.

## The three strategy axes

A run is factored into three orthogonal, independently-pluggable decisions,
each an abstract base class (`core/exec/multipath.hpp`):

| Axis | Base class | Built-in subclasses |
|---|---|---|
| **decompose** — split the output into `Tile`s | `IDecomposeStrategy` | `WholeFrame`, `HorizontalBands(n)`, `GridDecompose(w,h)` |
| **dispatch** — assign tiles to executors | `IDispatchStrategy` | `OnePathPerTile`, `AllPathsPerTile` |
| **merge** — recombine the tile results | `IMergeStrategy` | `StitchMerge`, `CompareMerge(tol)` |

Because the axes are separate, behaviours compose — N×M×K combinations from
a handful of subclasses — and a user can add a custom strategy by
subclassing one base without touching the others.

## Models as configurations

The "models" we discussed are just configurations:

- **cross-path equivalence check** = `WholeFrame` + `AllPathsPerTile` +
  `CompareMerge` — every path renders the whole frame; the merge reports
  max/mean per-channel difference and a CONSISTENT/DIVERGENT verdict.
- **CPU/GPU frame split** = `HorizontalBands(2)` + `OnePathPerTile` +
  `StitchMerge` — each path renders one band; the bands are stitched.
- **distributed render** (future) = `GridDecompose` + remote executors +
  `StitchMerge`.

Visual equivalence across paths is a *prerequisite* for a seamless split —
otherwise the band boundary is visible. So `CompareMerge` is both a model
in its own right and the diagnostic that makes `StitchMerge` sound. Build
equivalence first (measure with compare), then split.

## Measured equivalence (the central result)

Three pairwise comparisons on the same scene (benchmark_simple_spheres_27,
200×150) tell the whole story of why a shared IR matters:

| Pair                | mean \|Δ\| | what it isolates                        |
|---------------------|-----------|------------------------------------------|
| CPU_IR vs GPU_IR    | 0.00073   | shared IR, across CPU↔GPU hardware       |
| GPU_IR vs GPU_GLSL  | 0.0092*   | independent path, same GPU (pre-fixes)   |
| CPU_IR vs GPU_GLSL  | 0.00076   | independent path, after full alignment   |
| CPU_IR vs GPU_RTX   | 0.00022   | independent path, ray tracing vs raymarch |

The **GPU_RTX** row holds on both software RT (llvmpipe) and **hardware RT
(RTX 2080)**: the ray-tracing path reaches the floor across all 17 parity
scenes — including textures (mean 0.0002) and mesh (mean 0.00045) — at 400×300
on real RT cores, identical to the software-RT figure. Parity is
backend-independent because the SDF/normal/shade/sky are lifted from the GLSL
emitter, so the only difference from CPU_IR is the JIT-vs-SPIR-V numeric floor
plus the hardware-AABB broad-phase (which
doesn't change the surface).

A note on speed and broad-phase structure: with a single BLAS over the whole
scene, the RTX path is *slower* than the CPU JIT, because the RT cores do one
trivial AABB test per ray and *all* the real work happens in the custom
intersection shader (a software sphere-trace of the full O(N) scene_sdf) — the
hardware BVH never gets to cull anything, and the per-frame acceleration-
structure build, pipeline creation, and SBT assembly add fixed overhead. The
system instead builds one BLAS per CSG group, so the RT cores cull groups a ray
misses; that's where the hardware broad-phase pays off on many-object scenes.

### Broad-phase result (RTX 2080, sphere-grid sweep, 256×256)

Per-CSG-group BLAS (one per sphere) vs the O(N) flat-union CPU JIT raymarch.
Hot-loop only on both sides (CPU raymarch excl. JIT; RT `vkCmdTraceRays` excl.
pipeline/SBT/AS setup):

| N   | cpu trace (ms) | rtx trace (ms) |
|-----|----------------|----------------|
| 1   | ~46            | 0.3            |
| 4   | ~66            | 0.3            |
| 16  | ~137           | 0.9            |
| 64  | ~368           | 2.7            |
| 256 | ~2400          | 6.5            |

The slopes are the result: the CPU raymarch grows roughly linearly with N (every
march step evaluates the min over all N spheres), while the RT trace grows far
slower (~22× over a 256× object increase) because the BVH culls groups a ray
misses, so each ray sphere-traces only the few groups its pixel overlaps.

**Reporting these as a "speedup" ratio is misleading** — a CPU core and a GPU
core aren't commensurable units, and dividing by core counts doesn't fix it.
The defensible framing is *throughput*: pixels/s per path. The architectural
result is that independent devices **add** — running CPU and RT concurrently
delivers `cpu + rtx` pixels/s, which scaling a single device can't reach (you
can't turn 12 CPU cores into 512). Adding the other paths (gpu_glsl, gpu_ir,
remote nodes) raises the aggregate further. That heterogeneous aggregate, not
any single ratio, is the system's contribution. `frep_rtx_bench` reports
per-path and summed Mpix/s for exactly this reason.

Other metrics answer other questions and each is valid on its own terms:
pix/kWh (energy — informs whether to grow a datacenter with GPUs or CPUs, and
whether running CPU+GPU together is even energy-justified), pix/$ (capital),
pix/s (capacity). The energy axis is **measured, not invented**: both
`frep_rtx_bench --energy` (single RTX path) and `frep_multipath --energy` (the
heterogeneous aggregate, where Σ Mpix/kWh is the honest cross-device efficiency
number) read real hardware counters — Intel/AMD RAPL for the CPU package (via
the perf_event PMU, or powercap sysfs) and NVML for the GPU — and report
Mpix/kWh. On the RTX 2080 the GPU path measured tens to ~130k Mpix/kWh via NVML;
a counter that isn't available prints a dash rather than a fabricated figure.
pix/$ (capital cost) remains a named axis — it's an accounting input, not
something the renderer can measure.

The **shared-IR** pair (CPU_IR↔GPU_IR) is near-perfect at 0.0007 essentially
for free: both backends consume the identical LLVM IR, so only hardware
float rounding differs. The **independent** GLSL path started far worse
(mean 0.022, max 0.69) and required aligning every algorithm by hand to
reach parity:

- raymarch loop (t₀ = 0.001, bound-check at top of iteration)
- smooth-union IQ smin (the ×0.5 factor)
- analytic normals via dual-number AD (not central differences)
- specular visibility in the cancelled `D·F/(4·gv·gl)` form
- sky gradient from NDC `uv_y`, not the normalized ray direction
- explicit `sqrt`/`sqrt(dot)` instead of `length()`/`normalize()`
- **per-light colour** — the GLSL path had ignored `light.color` entirely,
  lighting every scene as if the light were pure white; for a warm light
  this was the single largest systematic offset (~10% on the blue channel
  of every lit pixel).

After all of these, CPU_IR↔GPU_GLSL reaches mean 0.00076 — the same order as
the shared-IR floor, with 99.9% of pixels within tolerance (the residual is
sub-pixel silhouette coverage). The result holds across scene types:

| scene            | mean \|Δ\| | ≤tol    | max   | residual                |
|------------------|-----------|---------|-------|-------------------------|
| simple_spheres   | 0.00076   | 99.89%  | 0.028 | smooth-surface edges    |
| heavy_csg        | 0.00091   | 99.37%  | 0.295 | sharp CSG edges (0.04%) |
| blob (smin)      | 0.00078   | 99.58%  | 0.027 | smooth-union edges      |

All three land at mean ≈ 0.0008, the shared-IR floor. The larger heavy_csg
max is concentrated in ~0.04% of pixels along the sharp box-minus-sphere
concave edges, where sub-pixel hit/miss between two independent paths is
inherent — not a shading difference.

The lesson for the paper: an independent retarget *can* be driven to parity,
but only by manually re-deriving every algorithmic detail the IR path gets
for free. The shared IR delivers equivalence by construction; the
independent path makes it laborious and fragile. (*The 0.0092
GPU_IR↔GPU_GLSL figure is pre-fix; it confirmed the gap was algorithmic, not
hardware — same GPU, 13× the shared-IR floor.)


## Systematic equivalence: many scenes, not one

The figures above come from three busy showcase scenes. Equivalence should be
demonstrated across the whole *feature surface*, not from a few scenes that
happen to exercise several features at once — a per-feature regression can
hide inside a busy scene's aggregate. So there is a library of small focused
scenes, one feature each, in `core/exec/parity_scenes.hpp`:

> sphere, box, union, intersection, difference, smooth_union, twist, taper,
> rotate, scale, customexpr, checker, stripes, gradient

Each is one or two objects on a shared floor with one camera and one warm
light, so a divergence localizes to the feature under test. Two harnesses use
the library:

- **`tests/test_path_parity.cpp`** — in the sandbox (no GPU) it renders every
  scene on two independent CPU_IR executors and asserts bit-identical frames
  (proving determinism across the whole surface), and checks that *both*
  shading models render every scene. 3 tests walking all scenes.
- **`frep_parity_check`** — renders every scene on two chosen paths and prints
  a per-scene mean/max |Δ| table plus an aggregate, exiting non-zero if any
  scene exceeds tolerance:

  ```
  frep_parity_check --paths cpu_ir,gpu_glsl --width 200 --height 150
  ```

  On hardware this is how CPU_IR↔GPU_GLSL (and ↔GPU_IR) equivalence is shown
  scene-by-scene; in a CPU-only sandbox `--paths cpu_ir,cpu_ir` exercises the
  harness itself (all zeros, confirming determinism). The metric matches the
  study above (mean / max absolute per-channel difference).

Image textures are sampled on all three paths: the GPU_GLSL emitter samples a
texture SSBO, and CPU_IR (hence GPU_IR, which shares the IR) samples an
embedded module-constant texture buffer via triplanar nearest lookup that
matches the shader byte-for-byte. The `texture` scene is in the parity library,
so texture equivalence is measured exactly like every other feature — and sits
at the shared-IR floor (15/15 scenes within ~0.00075 mean |Δ| at 400×300).

**Sampling note.** High-frequency material patterns (stripes, fine checker)
must be compared at a resolution proportional to the pattern frequency. At
200×150 a scale-8 stripe is only ~8 px tall, so a sub-pixel difference in
where a stripe edge lands flips a boundary pixel — `stripes`/`gradient` then
read as ~0.018 divergent. Rendered at 400×300 the same two paths are
bit-equivalent (119998/120000 pixels < 0.01, none above 0.05): the divergence
was boundary aliasing, not an algorithmic path difference. This is a property
of point-sampling a near-discontinuous albedo, identical to geometric
silhouette aliasing, and is resolved by adequate sampling (higher resolution
or SSAA proportional to the pattern frequency).


## Executors

An `IExecutor` renders a given `Tile` of a scene on one path:

- `CpuIrExecutor` — codegen → LLVM IR → JIT → tiled scheduler. Always
  available. The JIT'd `render_tile` is natively tile-addressed, and
  `RenderParams::region` restricts which tiles run, so a sub-region costs
  proportionally less (no whole-frame waste).
- `GpuGlslExecutor` — codegen → GLSL → SPIR-V → Vulkan compute. Available
  only with a real Vulkan device. The shader takes tile bounds in its push
  constants and the dispatch is sized to the tile, so it too renders only
  its region. The Vulkan context is built once and cached across tiles.

The GPU-IR path (LLVM IR → llvm-spirv → SPIR-V) shares the same push
constants and shader entry, so it gains tile support the same way; its
executor follows.

`RtxExecutor` — the fourth path, GPU ray tracing (`gpu_rtx`). It renders the
*same implicit SDF* as the others rather than a polygonized mesh: a hardware
acceleration structure does broad-phase (which object's AABB does a ray
enter), and a custom intersection shader sphere-traces the exact `scene_sdf`
inside that box. Parity is structural, not coincidental — the RT intersection
and closest-hit shaders are generated by lifting the scene_sdf / scene_albedo /
shade region straight out of the GLSL compute emitter and reusing the compute
path's push-constant builder, so the SDF evaluation and camera/light inputs are
byte-identical to `gpu_glsl`. Built on Vulkan ray tracing
(`VK_KHR_ray_tracing_pipeline`); runs on real RT cores (Turing+, e.g. the RTX
2080) and on a CPU emulation that advertises the extensions (llvmpipe), which
lets the pipeline be developed without RT cores. The naive Phase-1 strategy
uses one BLAS over the whole scene; CSG-aware per-group BLAS (real broad-phase,
benchmarked against the O(N) flat union) is the Phase-3 evolution. Available
only when a ray-tracing-capable Vulkan device is present.

## Running

`MultiPathExecutor(decompose, dispatch, merge).run(scene, W, H, executors,
mode)` does decompose → dispatch → execute → merge. `mode` is `Concurrent`
(one `std::async` per job) or `Serial`. Unavailable executors are filtered
before dispatch, so a GPU path simply drops out on a machine without a
device. The result carries every tile's pixels + metrics, the merged
outcome, and wall-vs-sum timings.

## CLI: `frep_multipath`

```
frep_multipath <scene.json> [options]
  --paths LIST       cpu_ir,gpu_glsl                (default cpu_ir)
  --decompose K      whole | halves | bands:N | weighted[:w0,w1,..] | grid:WxH
  --dispatch K       one_per_tile | all_paths       (default by merge)
  --merge K          compare | stitch
  --mode K           concurrent | serial
  --width N --height N
  --tolerance F      compare tolerance per channel
  --out FILE.ppm
```

Measure CPU-vs-GPU equivalence:
```
frep_multipath scene.json --paths cpu_ir,gpu_glsl --merge compare
```
exits non-zero (2) if the paths diverge beyond tolerance — scriptable as an
equivalence check. Try a frame split:
```
frep_multipath scene.json --paths cpu_ir,gpu_glsl \
               --decompose halves --merge stitch --out split.ppm
```

A **dynamic work-stealing** split needs no calibration — decompose into a
grid and let executors pull tiles from a shared queue as they finish:
```
frep_multipath scene.json --paths cpu_ir,gpu_glsl \
               --decompose grid:64x64 --mode dynamic --merge stitch --out split.ppm
```
A faster executor grabs more tiles automatically; the first tile per worker
warms its compile cache. This is more robust than `weighted` (no probe to
mis-estimate, and per-tile cost variation balances itself), at the cost of
per-tile dispatch overhead — pick a tile size large enough to amortize it.


