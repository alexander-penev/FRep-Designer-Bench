# MPR benchmark (CUDA host)

Driver mpr_bench (added into mpr/benchmark via patches) renders one scene at fixed
resolutions, printing the suite CSV row. Same scenes as scenes/libfive/*.frep and the
same 3D render sizes as libfive --render (512, 1024).

    ./mpr_bench/build.sh     # fetch+patch mpr, build target mpr_bench (needs nvcc)
    ./mpr_bench/run.sh       # per-scene, MPR_SIZES="512 1024", MPR_TIMEOUT=120s

Env: MPR_ROOT (source tree), MPR_SIZES, MPR_TIMEOUT. GPU energy sampled by
energy_wrap.sh (nvidia-smi power.draw); numeric-only, prints avg_W/joules to stderr.

Note: upstream render_3d_table sweeps 256..2048 with 20+100 reps and writes PNGs per
step — it can run for hours on heavy scenes (prospero@2048). mpr_bench replaces it for
comparable, bounded measurements (2 warmup + median of 5, no PNG).
