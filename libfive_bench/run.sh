#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"  # pin for stable timings
ROOT=${LIBFIVE_ROOT:-$(pwd)/libfive}
export LD_LIBRARY_PATH="$ROOT/build/libfive/src:$LD_LIBRARY_PATH"
B=libfive_bench/build/bench_libfive
mkdir -p scenes/libfive
$B --emit-frep scenes/libfive
# Canonical (s1-s5) and complex (c1_gear, c2_colonnade) scenes are built into
# the bench and run automatically. The previously-imported archives
# (architecture/hello_world/involute_gear/prospero) are dropped: their roots
# were atan2/division, not signed distance fields, so every evaluator — libfive's
# own interval render included — draws them empty (0% surface). See scenes/MATH.md.
$TASKSET $B --grid 193
$TASKSET $B --render 512 "$(nproc)"
$TASKSET $B --render 1024 "$(nproc)"
