#!/bin/sh
set -e
cd "$(dirname "$0")"
SUITE_DIR=$(cd .. && pwd); export SUITE_DIR
. ../common/fetch.sh
ROOT=$(resolve_root LIBFIVE_ROOT libfive)
apply_patches "$ROOT" "$(pwd)/patches"
if [ ! -f "$ROOT/build/libfive/src/libfive.so" ] && [ ! -f "$ROOT/build/libfive/src/libfive.a" ]; then
  cmake -S "$ROOT" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_STUDIO_APP=OFF -DBUILD_GUILE_BINDINGS=OFF -DBUILD_PYTHON_BINDINGS=OFF -Wno-dev
  cmake --build "$ROOT/build" --target libfive -j"$(nproc)"
fi
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIBFIVE_ROOT="$ROOT" -Wno-dev >/dev/null
cmake --build build -j"$(nproc)"
