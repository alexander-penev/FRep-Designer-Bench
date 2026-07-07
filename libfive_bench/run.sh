#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"  # pin for stable timings
ROOT=${LIBFIVE_ROOT:-$(pwd)/libfive}
export LD_LIBRARY_PATH="$ROOT/build/libfive/src:$LD_LIBRARY_PATH"
B=libfive_bench/build/bench_libfive
mkdir -p scenes/libfive
$B --emit-frep scenes/libfive
IMP="scenes/libfive/architecture.frep scenes/libfive/hello_world.frep scenes/libfive/involute_gear_2d.frep scenes/libfive/involute_gear_3d.frep scenes/libfive/prospero.frep"
$TASKSET $B --grid 193 $IMP
$TASKSET $B --render 512 "$(nproc)" $IMP
$TASKSET $B --render 1024 "$(nproc)" $IMP
