#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"  # pin for stable timings
# Generate the complex-scene .hf from libfive's valid SDFs (the old committed
# architecture/gear/prospero .hf came from non-SDF archives — NaN everywhere).
LB=libfive_bench/build/bench_libfive
if [ -x "$LB" ]; then
  export LD_LIBRARY_PATH="${LIBFIVE_ROOT:-$(pwd)/libfive}/build/libfive/src:$LD_LIBRARY_PATH"
  for s in c1_gear c2_colonnade; do
    [ -f "scenes/libfive/$s.frep" ] && $LB --export-hf "scenes/libfive/$s.frep" "scenes/hyperfun/$s.hf" >&2 || true
  done
fi
$TASKSET hyperfun_bench/build/bench_hf 193 scenes/hyperfun/s?_*.hf
# Complex scenes at 193, matching every other system's grid.
$TASKSET hyperfun_bench/build/bench_hf 193 scenes/hyperfun/c1_gear.hf scenes/hyperfun/c2_colonnade.hf
