#!/bin/sh
set -e
cd "$(dirname "$0")/.."
TASKSET=""; [ -n "$BENCH_CPU" ] && command -v taskset >/dev/null && TASKSET="taskset -c $BENCH_CPU"  # pin for stable timings
$TASKSET hyperfun_bench/build/bench_hf 193 scenes/hyperfun/s?_*.hf
$TASKSET hyperfun_bench/build/bench_hf 65 scenes/hyperfun/architecture.hf scenes/hyperfun/hello_world.hf \
    scenes/hyperfun/involute_gear_2d.hf scenes/hyperfun/involute_gear_3d.hf scenes/hyperfun/prospero.hf
