# Vendored source

This `frep4/` tree is **vendored** into FRep-Designer-Bench: it is tracked
in-tree rather than fetched from GitHub and patched at build time. The benchmark
build (`frep4_bench/build.sh`) uses it directly; there is no patch step.

- **Upstream:** https://github.com/alexander-penev/FRep-Designer.git
- **Base commit:** `cb8a6988809b864a8f7aae63d36fafc8d5505ea6`
  (`Merge pull request #9 from alexander-penev/FRep-Designer-4.0`)

Local modifications on top of that base (previously carried as
`frep4_bench/patches/0001..0007` + `.add` files, now applied directly):

- LLVM 22 codegen fix, nth-root all backends, NVPTX pow/exp/log lowering,
  adaptive SIMD width, vector render stage 1.
- Template functions, RTX multi-BLAS groups, smart import (validate + factor
  instances).
- RTX render fixes: HOST_CACHED framebuffer readback, cached per-frame output
  image + readback buffer (`acquire_frame_rt`), full-frame crop by move,
  `VK_KHR_ray_query` support, and the ray-query compute path (`core/gpu/rtx_query.*`).

To build against a different clone instead, set `FREP4_ROOT=/path/to/clone`.
