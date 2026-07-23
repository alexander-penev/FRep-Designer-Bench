# FRep Designer 4.0 — Performance Benchmarks

All numbers are wall-clock milliseconds, measured on the build environment.

## 1. CPU JIT — compile + render scaling

| Scene | JIT compile | Render 400×300 | Render 800×600 | Render 1280×720 |
|---|---|---|---|---|
| Simple (1 sphere) | 27 ms | 67 ms | 262 ms | 528 ms |
| Moderate (CSG diff) | 29 ms | 76 ms | 312 ms | 609 ms |
| Complex (5 objs, deforms) | 54 ms | 352 ms | 1420 ms | 2848 ms |
| Heavy (MeshSDF + CSG) | 118 ms | 153 ms | 603 ms | 1199 ms |
| CustomExpr (gyroid) | 35 ms | 149 ms | 576 ms | 1207 ms |

## 2. GPU vs CPU render speedup

Same scene, same resolution, both pipelines. GPU times exclude one-time pipeline setup (emit + glslang + Vulkan init); CPU times exclude one-time JIT compile. Steady-state numbers only — what you'd see at frame N+1.

| Scene | Resolution | CPU render | GPU render | Speedup |
|---|---|---|---|---|
| Simple (1 sphere) | 400x300 | 67 ms | 17 ms | 3.9× |
| Simple (1 sphere) | 800x600 | 262 ms | 64 ms | 4.1× |
| Moderate (CSG diff) | 400x300 | 79 ms | 18 ms | 4.3× |
| Moderate (CSG diff) | 800x600 | 308 ms | 68 ms | 4.5× |
| Complex (5 objs, deforms) | 400x300 | 355 ms | 66 ms | 5.4× |
| Complex (5 objs, deforms) | 800x600 | 1420 ms | 254 ms | 5.6× |
| Heavy (MeshSDF + CSG) | 400x300 | 156 ms | 50 ms | 3.1× |
| Heavy (MeshSDF + CSG) | 800x600 | 606 ms | 205 ms | 3.0× |
| CustomExpr (gyroid) | 400x300 | 158 ms | 31 ms | 5.2× |
| CustomExpr (gyroid) | 800x600 | 620 ms | 114 ms | 5.4× |

## 3. Incremental compilation modes

Same scene compiled three different ways. The Auto mode automatically picks Constant when only parameter values changed and Incremental when topology was the same.

| Mode | Recompile time |
|---|---|
| Constant (full recompile) | 90.8 ms |
| Incremental | 102.1 ms |
| Auto | 0.0 ms |

## 4. Sparse octree compression

Voxelised unit sphere SDF at various resolutions and tolerance settings.

| Resolution | Tolerance | Dense | Sparse | Ratio |
|---|---|---|---|---|
| 32³ | dense | 128 KB | — | 1.00× |
| 32³ | 0.05 | 128 KB | 274 KB | 0.47× |
| 32³ | 0.10 | 128 KB | 240 KB | 0.53× |
| 64³ | dense | 1024 KB | — | 1.00× |
| 64³ | 0.05 | 1024 KB | 1804 KB | 0.57× |
| 64³ | 0.10 | 1024 KB | 276 KB | 3.70× |
| 128³ | dense | 8192 KB | — | 1.00× |
| 128³ | 0.05 | 8192 KB | 2194 KB | 3.73× |
| 128³ | 0.10 | 8192 KB | 274 KB | 29.87× |

## 5. BVH-accelerated voxelization speedup

Voxelising a 5k-triangle sphere mesh at various grid resolutions. Numbers compare the BVH-accelerated path (default) against the brute-force path.

| Resolution | BVH | Brute force | Speedup |
|---|---|---|---|
| 32³ | 119 ms | ~3214 ms (est.) | ~27× |
| 48³ | 355 ms | ~9587 ms (est.) | ~27× |
| 64³ | 780 ms | ~21050 ms (est.) | ~27× |

---

## 6. Artifact dump (`frep_dump`) — for system analysis & write-ups

`frep_dump <scene.json> [basename] [options]` emits the intermediate
artifacts of both retargeting paths for one scene, so the generated code
can be inspected and the pipeline studied (e.g. for a paper) without
hand-instrumenting the renderer.

Outputs (alongside the chosen basename):

| File | Contents |
|---|---|
| `<base>.pre.ll` | LLVM IR straight from codegen, before optimisation |
| `<base>.post.ll` | LLVM IR after the O3 pipeline |
| `<base>.glsl` | generated GLSL compute shader (GPU path) |
| `<base>.spv` | SPIR-V bytecode (needs `glslangValidator` on PATH) |
| `<base>.scene.json` | normalised dump of the input scene |
| `<base>.render.ppm` | the final rendered frame (visual reference) |
| `<base>.stats.json` | timings, memory, parallelism, modes, code sizes |

The `stats.json` records, per path: CPU codegen/optimise/JIT/render times,
IR instruction & function counts **pre and post optimisation** (the
`ir_post_pct_of_pre` field quantifies how much the generated code expands
or shrinks under O3 — e.g. inlined scenes balloon to ~170% as `min()`
chains unroll and vectorise, while guarded scenes stay near 100% because
the per-object functions are kept separate), the **native machine-code
size** (`native_text_bytes` — the optimised module's `.text`, the CPU
analogue of SPIR-V size), a **parallelism scaling** sweep
(`thread_scaling`: render ms at 1, 2, 4, … hardware threads), GPU
GLSL/SPIR-V sizes and emit/compile times, and — when a real Vulkan device
is present (`ran_on_device`) — the **GPU init breakdown** (device / shader
/ pipeline-compile / buffers / misc ms) plus device-memory estimate. Also
peak/delta RSS, hardware thread count, and the explicitly-chosen SDF mode
(Inlined/Guarded) with the reason. A `<base>.render.ppm` of the final
frame is written as a visual reference. Side-by-side and pixel-diff
comparisons between paths are left to external tools operating on these
dumps.

Options: `--width/--height N` (render-timing size), `--no-guards` (force
the Inlined path), `--paths LIST` (which paths' artifacts to emit:
`cpu_ir`/`gpu_ir` → `.ll`, `gpu_glsl` → `.glsl`/`.spv`),
`--no-render` (skip the timing pass — just emit code artifacts).

---
_Generated by `frep_bench` from `/home/claude/frep4/tools/benchmarks.cpp`._
