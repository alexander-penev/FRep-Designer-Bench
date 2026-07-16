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

RC=$(mktemp); FAILED=''
trap 'rm -f "$RC"' EXIT
for sys in $SYSTEMS; do
    echo "==== $sys ===="
    if ! ./${sys}_bench/build.sh; then
        echo "$sys: build FAILED" >&2; FAILED="$FAILED $sys"; continue
    fi
    # `run.sh | tee` would report tee's status (always 0) and hide a crashing
    # backend, so stash the real status in $RC. set +e keeps the subshell alive
    # long enough to record a nonzero one.
    { set +e; ./${sys}_bench/run.sh; echo $? >"$RC"; } | tee -a "$OUT"
    rc=$(cat "$RC")
    [ "$rc" = 0 ] || { echo "$sys: run FAILED (rc=$rc)" >&2; FAILED="$FAILED $sys"; }
done
[ -z "$FAILED" ] || { echo "run_all: incomplete results, failed:$FAILED" >&2; exit 1; }
