# frep-designer-bench

Cross-system FRep performance suite: FRep Designer 4.0 vs HyperFun vs libfive vs MPR.
The frep4 drivers build against a FRep Designer source tree via FREP4_ROOT;
if unset, run.sh clones https://github.com/alexander-penev/FRep-Designer.git into ./frep4.


Four systems, one canonical scene set (scenes/MATH.md), two metric classes:

  A. grid-eval  : f(x,y,z) over an N^3 grid on [-1.6,1.6]^3  -> Meval/s
  B. render-3d  : depth/shaded image at RxR                  -> ms/frame, Mpix/s

| system   | grid-eval driver              | render driver                         | device |
|----------|-------------------------------|---------------------------------------|--------|
| frep4    | frep4_bench/bench_grid (JIT)  | $FREP4_ROOT tools/benchmarks (CPU_IR, GPU_GLSL) | CPU+GPU |
| hyperfun | hyperfun_bench/bench_hf (interpreter) | n/a (polygonizer only; see notes) | CPU |
| libfive  | libfive_bench (ArrayEvaluator)| libfive_bench --render (Heightmap, multithread) | CPU |
| mpr      | n/a (GPU renderer)            | mpr_bench (mpr/benchmark) scenes/libfive/*.frep, sizes 512/1024 | GPU (CUDA) |

Protocol: 1 warmup + median of 5; steady-state only (JIT/tape build reported separately
where the system exposes it). Same host, same scenes, fixed seeds. Results -> results/results.csv
(system,backend,scene,metric,size,ms,throughput,host,date).

Build order: ./run_all.sh  (skips systems whose toolchain is absent: CUDA -> mpr, LLVM -> frep4).
Fairness notes and sign conventions: scenes/MATH.md. HyperFun renders only via polygonization,
so it participates in class A and (optionally) a polygonization-time table, not class B.

##
## Scene compatibility & structured import (v2)
- MPR uses a pinned libfive; archives from the suite's newer libfive fail its deserializer.
  mpr_bench builds mpr_emit (against MPR's own libfive) into scenes/mpr/*.frep -> compatible.
- frep4 imports libfive scenes via --export-let: a let-binding CustomExpr that preserves the
  DAG (shared subexpressions), not a flattened expression. Needs frep4 >= 4.32.5 (let-bindings
  + node-identity memoized codegen). run.sh emits scenes/frep4/*.let before generating JSON.

## Uniform configuration (all four systems)
Each system resolves its source tree the same way:
  FREP4_ROOT | HYPERFUN_ROOT | LIBFIVE_ROOT | MPR_ROOT  (env var or -D on the bench cmake)
  unset/empty -> auto-fetched (latest) into ./frep4, ./hyperfun, ./libfive, ./mpr
  The four dirs ship EMPTY: default = latest upstream; to pin a version, place a
  checkout there (or point the ROOT var at one) before running the suite.
  (git clone --depth 1; HyperFun: sourceforge 7z).
Patches needed by the suite live in <sys>_bench/patches/ (*.patch + *.add with a
"// dest:" first line) and are applied idempotently before every build - an already
patched tree is detected and skipped, so pointing a ROOT at your own checkout is safe.
Pipeline per system: resolve -> patch -> build -> run -> append results/results.csv.
Entry point: ./run_all.sh (systems with missing toolchains are skipped, e.g. mpr without CUDA).

## Imported scenes (from the MPR paper / libfive)
mpr/benchmark/files/*.frep (binary libfive archives) are first-class scenes:
  - libfive + mpr consume them directly (byte-identical trees);
  - --export-hf ports the DAG to HyperFun assignment form (validated: all 5 parse+run);
  - --export-expr flattens to one infix expression for frep4 CustomExpr
    (architecture 75k / gears 61-81k / hello_world 14k chars; prospero exceeds the
    inline limit - needs let-bindings in the frep4 expression grammar).
Caveat: bear.frep fails to deserialize with current libfive master (map::at) - usable
only through mpr's pinned submodule; excluded from the cross-system set.
Grid size for imported scenes on HyperFun: 65 (interpreter, prospero ~0.07 Meval/s).

## Energy metric
CSV columns joules (per run) and uj_per_unit (uJ per eval / per pixel).
kWh/Mpix = joules_per_pixel * 1e6 / 3.6e6 = uj_per_unit / 3.6e6.
CPU: RAPL package counter (common/energy.hpp; -1 in containers without powercap).
GPU: mpr/energy_wrap.sh samples nvidia-smi power.draw around any command.

## Status on the packaging host (1-core sandbox, reference only)
built+ran: hyperfun (grid 2.3-7.5 Meval/s), libfive (grid 173-431 Meval/s; heightmap 1024^2 37-180 ms).
prepared:  frep4 (needs FREP4_ROOT+LLVM), mpr (needs CUDA; scenes/libfive/*.frep already emitted).

## ABI note (libfive)
libfive builds with -march=native; any TU including its Eigen-based headers must use the
same flags (bench CMake does: -O3 -march=native -DEIGEN_NO_DEBUG) or ArrayEvaluator::set
segfaults on layout mismatch.

## frep4 imported-scene caveat
Imported (libfive/MPR) scenes reach frep4 as one flattened infix expression, so DAG
sharing is lost and the JIT sees a much larger tree than the native systems do; those
rows measure the expression path, not the graph path. DAG-preserving import is future work.

## Energy & stable timings
- CPU (RAPL): energy_uj is root-only (kernel >=5.10). Run the suite via sudo, or
  `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`; else joules=-1.
- GPU (MPR): energy_wrap.sh samples nvidia-smi power.draw and appends a CSV
  energy row (metric=energy: ms=elapsed, joules, uj_per_unit=avg_W).
- Pin cores for low-variance timings: BENCH_CPU=2 ./run_all.sh (uses taskset -c).
- frep4 needs C++26: build.sh auto-selects clang++-22/21/20; override via CXX/CC/LLVM_DIR.
  Default LLVM_DIR targets 22; with LLVM 20 frep4 builds in compatibility mode.

## Interval pruning scope
FREP4_PRUNE=1 prunes only when interval bounds are tight enough (<=90% cells left);
true-SDF canonical scenes prune to ~7% (2-8x over SIMD). Imported analytic scenes
(gyroid/gears, flattened multi-term trig) have loose interval bounds (interval
dependency problem on non-SDF expressions), so the driver auto-falls back to the
SIMD path for them. Same limitation applies to libfive's interval evaluator.
