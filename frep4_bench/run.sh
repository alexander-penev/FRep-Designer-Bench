#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"
mkdir -p scenes/frep4
# Structured import: libfive DAG -> frep4 CustomExpr let-form (needs libfive bench built).
LB=libfive_bench/build/bench_libfive
if [ -x "$LB" ]; then
  export LD_LIBRARY_PATH="${LIBFIVE_ROOT:-$(pwd)/libfive}/build/libfive/src:$LD_LIBRARY_PATH"
  # Complex scenes (valid SDFs, emitted by libfive_bench --emit-frep). The old
  # imported archives were non-SDF (atan2/division roots) and evaluated to NaN
  # everywhere; see scenes/MATH.md.
  for f in scenes/libfive/c1_gear.frep scenes/libfive/c2_colonnade.frep; do
    # Scene-prep chatter goes to stderr; only CSV rows may reach stdout, which
    # run_all.sh tees straight into results.csv.
    [ -f "$f" ] && $LB --export-let "$f" "scenes/frep4/$(basename "$f" .frep).let" >&2 || true
  done
fi
frep4_bench/build/frep4_gen_scenes scenes/frep4 >&2
$TASKSET frep4_bench/build/frep4_bench_grid 193 scenes/frep4/s?_*.json
# Complex scenes at 193 too, so they match libfive_bench's grid (a smaller grid
# here would compare against libfive's 193^3 and read as a false win).
$TASKSET frep4_bench/build/frep4_bench_grid 193 scenes/frep4/c1_gear.json scenes/frep4/c2_colonnade.json
