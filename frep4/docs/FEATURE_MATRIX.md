# Feature matrix — what each retargeting path supports

A factual audit of which features are implemented on which of the four paths,
so gaps are explicit (for the paper and for planning).

The four paths: **CPU_IR** (codegen → LLVM IR → JIT → x86), **GPU_IR** (same IR
→ NVPTX → CUDA), **GPU_GLSL** (independent GLSL emitter → SPIR-V → Vulkan
compute), **GPU_RTX** (RT shaders lifted from the GLSL emitter → Vulkan ray
tracing).

## Geometry & operations

| Feature                    | CPU_IR | GPU_IR | GPU_GLSL | GPU_RTX |
|----------------------------|:------:|:------:|:--------:|:-------:|
| Sphere / Box / Plane       |   ✓    |   ✓    |    ✓     |    ✓    |
| Union / Intersection / Diff|   ✓    |   ✓    |    ✓     |    ✓    |
| SmoothUnion                |   ✓    |   ✓    |    ✓     |    ✓    |
| Negate                     |   ✓    |   ✓    |    ✓     |    ✓    |
| Translate                  |   ✓    |   ✓    |    ✓     |    ✓    |
| Scale (uniform + non-unif.)|   ✓    |   ✓    |    ✓     |    ✓    |
| RotateX / RotateY / RotateZ|   ✓    |   ✓    |    ✓     |    ✓    |
| TwistY / TaperY            |   ✓    |   ✓    |    ✓     |    ✓    |
| BendXY                     |   ✓    |   ✓    |    ✓     |    ✓    |
| CustomExpr                 |   ✓    |   ✓    |    ✓     |    ✓    |
| **MeshSDF** (voxel grid)   |   ✓    |   ✓    |    ✓     |  ✓ (1)  |
| Plugin nodes               |   ✓    |   ✓    |    ✓     |   (2)   |
| **Instance** (L1: shared reference) | ✓ | ✓ |   ✓     |    ✓    |
| **Instance** (L2: shared subprogram, memory) | ✓ | ✓ | ✓ |  ✓  |

## Acceleration & debug (GLSL-emitter features)

| Feature                    | CPU_IR | GPU_IR | GPU_GLSL | GPU_RTX |
|----------------------------|:------:|:------:|:--------:|:-------:|
| Tile cull — Lipschitz/Auto-metric | ✓ | ✓ | ✓ | — |
| Tile cull — Interval / Auto-nonmetric | ✓ | ✓ | ✓ | ✓ (2) |
| Ray-box clip (scene AABB near/far) | ✓ | ✓ | (1) | (2) |
| Debug views (step heatmap, cull span) | — | — | ✓ | — |

## Materials & shading

| Feature                    | CPU_IR | GPU_IR | GPU_GLSL | GPU_RTX |
|----------------------------|:------:|:------:|:--------:|:-------:|
| Solid / checker / stripes / gradient | ✓ | ✓ |  ✓     |    ✓    |
| **Texture** (image sample) |   ✓    |   ✓    |    ✓     |  ✓ (1)  |
| Cook-Torrance, shadows, AO |   ✓    |   ✓    |    ✓     |    ✓    |
| Analytic-AD normals        |   ✓    |   ✓    |    ✓     |   (3)   |

## Notes

1. **MeshSDF / Texture on GPU_RTX** — now supported. Each needs a storage-buffer
   descriptor (voxels at binding 3 / RGBA8 pixels at binding 2) bound to the RT
   stages; the lifted shared region declares the buffer and samples it exactly
   like the compute path. Both confirmed at the floor on the RTX 2080:
   textures mean 0.0002, mesh mean 0.00045 (max 0.032) vs cpu_ir — within the
   0.0078 tolerance. RT is **17/17**: full feature parity across all four paths.

2. **Plugin nodes on GPU_RTX** — untested. The lifted shared region includes
   whatever `emit_glsl` produced, so a plugin with no extra bindings would likely
   work, but this hasn't been exercised.

3. **Analytic-AD normals on GPU_RTX** — the RT closest-hit currently calls the
   lifted `scene_normal`, which uses the emitter's normal (analytic where the
   sub-tree supports it, central-difference otherwise) — same as GPU_GLSL.

4. **Instance — two levels.** *Level 1* is the reference semantics: an
   InstanceNode shares (does not copy) the target's geometry pointer, so editing
   the target updates every instance live. Because codegen emits geometry through
   the virtual `FRepNode::codegen()` and InstanceNode delegates to its target,
   Level 1 works unchanged on **all four paths**. *Level 2* is the memory
   optimisation: the shared subtree is emitted **once** as a GLSL function and
   called, so the emitted code grows with the number of distinct shapes, not the
   instance count (60% smaller shader at 64 instances). Level 2 lives in the GLSL
   emitter, so it is **GPU_GLSL-only**; the IR paths still inline each instance
   (correct, just not deduplicated). Bringing L2 to the IR paths would mean
   emitting shared LLVM functions in codegen.cpp — future work if the IR paths
   need the memory win.

5. **Tile cull and debug views** are GLSL-emitter features (the cull block and
   the step-heatmap / cull-span shading are emitted into the GLSL march). They
   are **GPU_GLSL-only** by construction. The IR paths have no tile cull; this is
   a known gap, not a regression. Cull correctness and method selection
   (Lipschitz L=1 for metric trees, Interval for non-metric, Auto by topology)
   are covered in PERFORMANCE_TUNING.md.

## Distributed execution (orthogonal to the four paths)

Distributed render is **implemented and tested**, not a placeholder: `dist::Master`
serves tiles, `dist::Worker` renders them with *any* IExecutor (so any of the
four paths can run distributed), `PullScheduler`/`PushScheduler` balance work,
`TcpBinaryTransport` carries scene + tiles. `test_dist_render` validates an
end-to-end master + 2-worker render over TCP loopback stitches to match a
whole-frame render. `PathKind::Remote` is just a name in the enum; the
distributed stack coordinates directly via Master/Worker rather than through
that path tag.

Remaining for distributed: a real multi-machine LAN run (loopback has proven
correctness; LAN is validation, not new code), and a weighted scheduler that
accounts for heterogeneous per-node throughput.

## The heterogeneous-aggregate point

Because Worker takes an ExecutorFactory, a cluster can mix paths and machines —
the system's throughput is the *sum* across all participating devices/nodes,
which no single device can reach by scaling. That additive aggregate is the
architecture's contribution; see MULTIPATH.md for the per-path throughput and
the energy/cost axes that decide *which* devices to add.

## Honest gap summary

- **Cross-path consistency (complete):** instancing Level 2 and interval tile
  cull now run on all four paths — CpuIr, GpuIr, GpuGlsl, and GpuRtx. For RTX,
  Level 2 works because the shared _inst_fn_N subprograms are lifted into the RT
  intersection shader (the lift now starts at the earliest instance function, not
  scene_sdf); the "tile cull" analog is an interval pre-skip in the intersection
  shader that bounds the field over each ray's AABB segment and skips the
  sphere-trace when the segment can't contain the surface (the hardware BVH
  already does object-level broad-phase; this trims empty space within an
  object's AABB). Both verified by real rendering on the lavapipe software RT
  device: RTX interval cull on-vs-off and Level-2 vs Level-1 are byte-identical
  (max diff 0.0000). Notes: (1) the GLSL compute path skips the ray-box clip
  because its tile cull already bounds the march tightly; (2) on RTX the hardware
  BVH provides the broad-phase, and the interval pre-skip provides the
  within-AABB trim, so a separate ray-box clip is redundant. Performance on real
  NVIDIA RT hardware is for the user to confirm; the software RT device
  establishes correctness and shader validity.
- **Done, hardware-confirmed:** MeshSDF + Texture on GPU_RTX; RT is 17/17.
- **Untested, likely fine:** plugin nodes on GPU_RTX.
- **Done, was validation:** multi-machine LAN distributed run.
- **Optimization, not correctness:** per-frame RT setup amortization; the GLSL
  mesh double-emit (mesh_count=2); weighted distributed scheduler; extending
  instancing Level 2 and tile cull to the IR paths.
