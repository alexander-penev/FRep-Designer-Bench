# Benchmark run — sandbox environment

Run of the FRep-Designer-Bench suite. frep4 source is the public repo
(github.com/alexander-penev/FRep-Designer, v4.53.0), auto-fetched.

## Systems runnable here

| system   | backend         | runs? | note                                  |
|----------|-----------------|-------|---------------------------------------|
| frep4    | cpu_ir-simd8    | yes   | LLVM JIT + 8-wide SIMD                 |
| frep4    | render paths    | yes   | cpu_ir, gpu_glsl, gpu_rtx (see below) |
| libfive  | cpu-array       | yes   | C++ API (no guile frontend needed)    |
| hyperfun | interpreter     | yes   | reference tree-walking interpreter    |
| mpr      | cuda            | no    | CUDA toolkit + NVIDIA GPU required     |

## Render paths (bench_render, 256², software Vulkan = lavapipe/llvmpipe)

| path      | runs here? | why                                           |
|-----------|------------|-----------------------------------------------|
| cpu_ir    | yes        | LLVM JIT on the host                          |
| gpu_glsl  | yes        | GLSL compute via llvmpipe                     |
| gpu_rtx   | yes        | lavapipe exposes VK_KHR_ray_tracing_pipeline  |
| gpu_ir    | no         | needs llvm-spirv (SPIR-V translator) — absent |

Software-Vulkan render times are NOT representative of real-GPU performance —
they establish correctness and relative behaviour. On real hardware the GPU
paths (glsl / ir / rtx) win by a large margin; here gpu_glsl leads only because
llvmpipe runs the same CPU cores the JIT does, without the launch overhead the
RT pipeline adds.

## Results

See results.csv. metric=grid is SDF evaluation (Mvox/s throughput);
metric=render is the render pipeline (ms/frame, Mpixel/s throughput).

## To run the paths this environment can't

- **gpu_ir**: install SPIRV-LLVM-Translator (`llvm-spirv`) matching the LLVM
  version, then it appears automatically.
- **mpr** and **real GPU numbers**: run on the CUDA host (vv-nuc). Point
  `FREP4_ROOT` at the working tree if you want to benchmark a local build:
  `FREP4_ROOT=/build/ap/FRep/frep4 ./run_all.sh frep4`.
- **libfive render**: the `--render` step segfaults headless here (needs a
  display/GL context); the `--grid` numbers are unaffected.

## Selective runs (new in run_all.sh)

    ./run_all.sh                 # all four systems (default)
    ./run_all.sh frep4           # just frep4
    ./run_all.sh frep4 libfive   # a subset
