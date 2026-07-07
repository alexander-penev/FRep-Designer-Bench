#!/bin/sh
set -e
cd "$(dirname "$0")"
SUITE_DIR=$(cd .. && pwd); export SUITE_DIR
. ../common/fetch.sh
ROOT=$(resolve_root MPR_ROOT mpr)
apply_patches "$ROOT" "$(pwd)/patches"
command -v nvcc >/dev/null || { echo "mpr: CUDA toolkit required - skipping build"; exit 0; }
( cd "$ROOT" && git submodule update --init 2>/dev/null || true )
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build "$ROOT/build" --target mpr_bench mpr_emit -j"$(nproc)"
# MPR-native canonical scenes (its pinned libfive -> deserialize-compatible)
mkdir -p "$SUITE_DIR/scenes/mpr"
"$ROOT/build/benchmark/mpr_emit" "$SUITE_DIR/scenes/mpr"
