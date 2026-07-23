# GpuRtx path — hardware validation guide

The GpuRtx path (fourth retargeting path, Vulkan ray tracing) is written and
compiles, and its shaders validate to SPIR-V, but it cannot run in the
sandbox (no Vulkan device there). This guide lists the checks to run on a
machine with a ray-tracing-capable Vulkan device — the workstation's llvmpipe
(software RT, slow but correct) or an RTX 2080 (hardware RT).

## 1. Capability + device + acceleration structure

```
./build/frep_rtx_probe
```

Expected on the workstation (GTX 1050 Ti + llvmpipe):

- detection lists both devices; llvmpipe is classified **Software** (CPU type),
  the 1050 Ti has no RT extensions.
- RT device creation: **created OK** on llvmpipe (software/CPU).
- acceleration structure build: **BLAS+TLAS built OK**, prints a
  non-zero TLAS device address.

On an RTX 2080: the card is classified **Hardware**, and 1a/1b run on it.

If 1a fails with a feature/extension message, the driver doesn't expose the
full RT feature chain. If 1b fails, note the Vulkan step in the message.

## 2. First end-to-end RT render

Start small — llvmpipe RT is slow:

```
./build/frep_parity_check --only sphere --paths cpu_ir,gpu_rtx \
    --width 200 --height 150 --dump-images rtimg/
```

- A **RENDER FAIL** line prints the failing path's error (which Vulkan/compile
  step). Share it verbatim.
- On success it writes `rtimg/sphere_cpu_ir.ppm` and `rtimg/sphere_gpu_rtx.ppm`.
  Compare them by eye first: the RTX image should show the same sphere in the
  same place. Then read the mean/max |Δ|.

The goal is parity at the shared-SDF floor (~0.0007 mean), the same
as the other paths, because the RT intersection shader runs the lifted
`scene_sdf` with the same epsilon/safety-factor and the same camera/light push
constants.

## 3. Widen to the analytic scenes

Once sphere matches, the other analytic scenes should too (textures/mesh are
rejected on the RT path until a later phase):

```
./build/frep_parity_check --paths cpu_ir,gpu_rtx --width 200 --height 150 \
    --only union
# then: intersection, difference, smooth_union, twist, taper, rotate, scale,
#       customexpr, box, checker, stripes, gradient
```

## 4. Validation on RTX 2080 hardware

When the RTX 2080 is attached, the same parity run should pass on real RT
cores. The probe should now classify it as **Hardware**:

```
./build/frep_rtx_probe
# device[N] "NVIDIA GeForce RTX 2080" type=discrete ray_tracing_pipeline=yes ...
# result: hardware ray tracing on "NVIDIA GeForce RTX 2080" (RT cores, ...)
```

Then run the analytic parity set with the backend line and timings:

```
./build/frep_parity_check --paths cpu_ir,gpu_rtx --width 400 --height 300 --timing
```

- The `[gpu_rtx backend]` line should say **hardware ray tracing on "NVIDIA
  GeForce RTX 2080"** — confirming the numbers are from real RT cores, not the
  llvmpipe emulation. This is what lets the paper claim hardware RT.
- All 14 analytic scenes should be at the floor (~0.0002 mean), same as on
  software RT — parity is hardware-independent because the SDF/shade is lifted
  from the shared emitter.
- The `--timing` columns give the first RTX-vs-CPU render times. On hardware RT
  the gpu_rtx times should be far below llvmpipe's; this is the seed of the
  the broad-phase performance story (RTX vs O(N) flat union).

If a scene diverges only on hardware (not on llvmpipe), that points to a
driver/precision difference rather than an algorithmic one — dump-images and
compare. None is expected, since the shaders are identical.


## Likely first-trace issues (and where to look)

The whole Vulkan RT chain is untested until the first run, so expect to iterate:

- **Black image, no error** — rays miss the AABB (camera basis / box bounds),
  or the intersection shader's t-range vs `gl_RayTmin/TmaxEXT` is off. Dump
  images and check whether *any* pixel is lit.
- **Validation error on pipeline create** — SBT alignment
  (`shaderGroupBaseAlignment` / `HandleAlignment`) or the hit-group type
  (`PROCEDURAL_HIT_GROUP` for AABB geometry) — see `rtx_pipeline.cpp`.
- **Wrong colours / shading** — push-constant layout drift; the
  `RtxPipeline.PushConstantLayoutMatchesComputePath` test guards the struct,
  but the GLSL `Push` block in the emitter must also match.
- **Crash in AS build** — device-address buffer flags / scratch size; see
  `rtx_accel.cpp`.

Report the exact message (or the dumped PPMs) and the failing stage; each maps
to a specific spot in the RT modules.

## 5. Multi-BLAS broad-phase scaling benchmark

The performance study: does per-CSG-group BLAS let the RT cores beat the O(N)
flat-union compute path as object count grows?

```
./build/frep_rtx_bench --width 256 --height 256 --counts 1,4,16,64,256
```

Reads as: for each N, a grid of N disjoint spheres (→ N CSG groups → N BLASes),
timing the CPU flat-union render vs the RT multi-BLAS trace. The backend line
should say hardware RT on the RTX 2080.

What to look for:
- The `cpu_ir ms` column should grow roughly linearly with N — every march step
  evaluates the min over all N spheres.
- The `rtx trace ms` column should grow much more slowly once N is large: the
  RT cores cull groups a ray misses, so each ray only sphere-traces the few
  groups its pixel actually overlaps.
- The `speedup` column crossing 1.0× marks where RT broad-phase starts winning.
  At small N the RT path loses (per-frame pipeline/SBT/AS overhead dominates);
  the interesting result is the trend and the crossover.

If a row prints "rtx failed", share the message — it maps to the multi-BLAS AS
(rtx_accel build_groups) or the SBT/pipeline (rtx_trace_groups).
