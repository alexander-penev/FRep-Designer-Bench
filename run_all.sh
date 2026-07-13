#!/bin/sh
# One entry point: every system resolves its ROOT (<SYS>_ROOT or ./<sys>),
# auto-fetches when empty, patches idempotently, builds, runs, appends CSV.
set -e
cd "$(dirname "$0")"
mkdir -p results
OUT=results/results.csv
[ -f "$OUT" ] || echo "system,backend,scene,metric,size,ms,throughput,joules,uj_per_unit" > "$OUT"

# Проверяваме дали има подадени параметри
if [ $# -gt 0 ]; then
    SYSTEMS="$@"
    echo "Systems (by params): $SYSTEMS"
else
    SYSTEMS="hyperfun libfive frep4 mpr"
    echo "Systems (default): $SYSTEMS"
fi

for sys in $SYSTEMS; do
    echo "==== $sys ===="
    if ./${sys}_bench/build.sh; then
        ./${sys}_bench/run.sh | tee -a "$OUT" || echo "$sys: run failed"
    else
        echo "$sys: build skipped/failed"
    fi
done