#!/bin/sh
set -e
cd "$(dirname "$0")"
SUITE_DIR=$(cd .. && pwd); export SUITE_DIR
. ../common/fetch.sh
# frep4 is vendored in-tree (see ../frep4), not fetched + patched: resolve_root
# returns the in-tree copy unchanged (non-empty -> no fetch), and there is no
# patch step. Set FREP4_ROOT to build against an external clone instead.
ROOT=$(resolve_root FREP4_ROOT frep4)
# frep4 needs C++26 -> a recent clang; pick clang-22 else clang-20. Override with CXX/LLVM_DIR.
if [ -z "$CXX" ]; then
  for v in 22 21 20; do command -v clang++-$v >/dev/null && { CXX=clang++-$v; CC=clang-$v; LLVM_DIR=${LLVM_DIR:-/usr/lib/llvm-$v/lib/cmake/llvm}; break; }; done
fi
: "${CXX:=clang++}" "${CC:=clang}" "${LLVM_DIR:=/usr/lib/llvm-22/lib/cmake/llvm}"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFREP4_ROOT="$ROOT" \
      -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DLLVM_DIR="$LLVM_DIR" -Wno-dev >/dev/null
cmake --build build --target frep4_gen_scenes frep4_bench_grid -j"$(nproc)"
