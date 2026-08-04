#!/bin/sh
set -e
cd "$(dirname "$0")/.."
ROOT=${MPR_ROOT:-$(pwd)/mpr}
BIN="$ROOT/build/benchmark/mpr_bench"
[ -x "$BIN" ] || { echo "mpr: not built (CUDA host required)"; exit 0; }
: "${MPR_SIZES:=512 1024}"
: "${MPR_TIMEOUT:=120}"
mkdir -p results
# mpr reads .frep archives written by the libfive it vendors (8ca1b868), whose
# opcode numbering differs from the much newer libfive the suite builds -- an
# archive from the suite's libfive fails its deserializer outright. So emit the
# scenes through mpr's own libfive into mpr's own directory, exactly as every
# other system has one, rather than borrowing libfive's (which mpr cannot read).
DIR=scenes/mpr
mkdir -p "$DIR"
EMIT=mpr_bench/build/emit_scenes_mpr
if [ ! -x "$EMIT" ] && [ -d mpr/libfive/libfive/include ]; then
    mkdir -p mpr_bench/build
    g++ -std=c++17 -O2 -I. -I mpr/libfive/libfive/include \
        -o "$EMIT" mpr_bench/emit_scenes_mpr.cpp \
        -L mpr/build/libfive/libfive/src -lfive 2>/dev/null || EMIT=""
fi
if [ -n "$EMIT" ] && [ -x "$EMIT" ]; then
    LD_LIBRARY_PATH="mpr/build/libfive/libfive/src:$LD_LIBRARY_PATH" \
        "$EMIT" "$DIR" >/dev/null 2>&1 || echo "mpr: scene emit failed" >&2
fi
ls "$DIR"/*.frep >/dev/null 2>&1 || { echo "mpr: no scenes in $DIR" >&2; exit 0; }
for f in "$DIR"/s?_*.frep; do
  [ -f "$f" ] || continue
  err=results/mpr_$(basename "$f" .frep).stderr
  mpr_bench/energy_wrap.sh timeout "$MPR_TIMEOUT" "$BIN" "$f" $MPR_SIZES 2>"$err"; rc=$?
  [ $rc -eq 0 ] && rm -f "$err" || {
    [ $rc -eq 124 ] && echo "mpr: $f TIMEOUT (${MPR_TIMEOUT}s)" >&2 \
                    || echo "mpr: $f error rc=$rc -> $err" >&2; }
done
