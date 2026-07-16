#!/bin/sh
set -e
cd "$(dirname "$0")/.."
ROOT=${MPR_ROOT:-$(pwd)/mpr}
BIN="$ROOT/build/benchmark/mpr_bench"
[ -x "$BIN" ] || { echo "mpr: not built (CUDA host required)"; exit 0; }
: "${MPR_SIZES:=512 1024}"
: "${MPR_TIMEOUT:=120}"
mkdir -p results
# MPR-native canonical scenes; falls back to scenes/libfive if mpr_emit unavailable.
DIR=scenes/mpr; [ -d "$DIR" ] && ls "$DIR"/*.frep >/dev/null 2>&1 || DIR=scenes/libfive
for f in "$DIR"/s?_*.frep; do
  [ -f "$f" ] || continue
  err=results/mpr_$(basename "$f" .frep).stderr
  mpr_bench/energy_wrap.sh timeout "$MPR_TIMEOUT" "$BIN" "$f" $MPR_SIZES 2>"$err"; rc=$?
  [ $rc -eq 0 ] && rm -f "$err" || {
    [ $rc -eq 124 ] && echo "mpr: $f TIMEOUT (${MPR_TIMEOUT}s)" >&2 \
                    || echo "mpr: $f error rc=$rc -> $err" >&2; }
done
