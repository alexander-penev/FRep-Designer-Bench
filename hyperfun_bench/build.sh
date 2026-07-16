#!/bin/sh
set -e
cd "$(dirname "$0")"
SUITE_DIR=$(cd .. && pwd); export SUITE_DIR
. ../common/fetch.sh
ROOT=$(resolve_root HYPERFUN_ROOT hyperfun)
apply_patches "$ROOT" "$(pwd)/patches"
HFP="$ROOT/hfp"; [ -d "$HFP" ] || HFP="$ROOT/hyperfun_source/hfp"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHFP_DIR="$HFP" -Wno-dev >/dev/null
cmake --build build -j"$(nproc)"
