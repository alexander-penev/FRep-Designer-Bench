#!/bin/sh
# Samples nvidia-smi power.draw@100ms around CMD. Passes CMD stdout through.
# Appends a CSV energy row (system=mpr,metric=energy) with avg_W/elapsed/joules.
# awk's %f honours LC_NUMERIC; a comma-decimal locale would split the CSV row.
LC_ALL=C; export LC_ALL
LOG=$(mktemp)
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 100 >"$LOG" 2>/dev/null &
SMI=$!
S=$(date +%s.%N); "$@"; RC=$?; E=$(date +%s.%N)
kill $SMI 2>/dev/null
awk -v t="$(echo "$E-$S"|bc)" \
  '/^[0-9]+(\.[0-9]+)?$/{s+=$1;n++} END{
     if(n) printf "mpr,gpu-cuda,_run,energy,0,%.2f,0,%.1f,%.4f\n",t*1000,s/n*t,s/n;
     else  printf "mpr,gpu-cuda,_run,energy,0,%.2f,0,-1,-1\n",t*1000}'  "$LOG"
rm -f "$LOG"; exit $RC
