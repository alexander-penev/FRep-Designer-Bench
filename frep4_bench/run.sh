#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"
mkdir -p scenes/frep4
# Structured import: libfive DAG -> frep4 CustomExpr let-form (needs libfive bench built).
LB=libfive_bench/build/bench_libfive
if [ -x "$LB" ]; then
  export LD_LIBRARY_PATH="${LIBFIVE_ROOT:-$(pwd)/libfive}/build/libfive/src:$LD_LIBRARY_PATH"
  for f in scenes/libfive/architecture.frep scenes/libfive/hello_world.frep \
           scenes/libfive/involute_gear_2d.frep scenes/libfive/involute_gear_3d.frep; do
    [ -f "$f" ] && $LB --export-let "$f" "scenes/frep4/$(basename "$f" .frep).let" || true
  done
fi
frep4_bench/build/frep4_gen_scenes scenes/frep4
$TASKSET frep4_bench/build/frep4_bench_grid 193 scenes/frep4/s?_*.json
$TASKSET frep4_bench/build/frep4_bench_grid 65 scenes/frep4/architecture.json scenes/frep4/hello_world.json \
    scenes/frep4/involute_gear_2d.json scenes/frep4/involute_gear_3d.json
