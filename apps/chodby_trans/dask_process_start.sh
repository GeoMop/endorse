#!/usr/bin/env bash
# Usage: dask_process_start.sh <dask-bin> <sched-addr> <count>

set -euo pipefail

DASK_BIN=$1
SCHED_ADDR=$2
COUNT=$3

: "${SCRATCHDIR:?SCRATCHDIR not set}"

LOG="$SCRATCHDIR/logs/worker_${HOSTNAME}.log"

CMD=( "$DASK_BIN" worker "$SCHED_ADDR"
      --nworkers "$COUNT" --nthreads 1
      --local-directory "$SCRATCHDIR/dask"
      --memory-limit auto )

exec </dev/null

# nohup setsid "${CMD[@]}" >"$LOG" 2>&1 < /dev/null &
nohup setsid "${CMD[@]}" >"$LOG" 2>&1 &
echo $! > "$SCRATCHDIR/worker_${HOSTNAME}.pid"
echo "Started Dask worker on $HOSTNAME, procs=$COUNT, pid=$(cat "$SCRATCHDIR/worker_${HOSTNAME}.pid")"
