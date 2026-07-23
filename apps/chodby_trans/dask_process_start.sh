#!/usr/bin/env bash
# Usage: dask_process_start.sh <dask-bin> <sched-addr> <count>

set -euo pipefail

DASK_BIN=$1
SCHED_ADDR=$2
WORKER_COUNT=$3

: "${SCRATCHDIR:?SCRATCHDIR not set}"

export ENDORSE_DISABLE_MEMOIZE="${ENDORSE_DISABLE_MEMOIZE:-}"
export DASK_DISTRIBUTED__WORKER__DAEMON=False

CMD=( "$DASK_BIN" worker "$SCHED_ADDR"
      --nworkers 1 --nthreads 1
      --no-nanny
      --local-directory "$SCRATCHDIR/dask"
      --memory-limit auto )

exec </dev/null

for ((WORKER_IDX=0; WORKER_IDX<WORKER_COUNT; WORKER_IDX++)); do
  LOG="$SCRATCHDIR/logs/worker_${HOSTNAME}_${WORKER_IDX}.log"
  printf 'ENDORSE_DISABLE_MEMOIZE=%s\n' "$ENDORSE_DISABLE_MEMOIZE" > "$LOG"
  printf 'DASK_DISTRIBUTED__WORKER__DAEMON=%s\n' \
    "$DASK_DISTRIBUTED__WORKER__DAEMON" >> "$LOG"
  # nohup setsid "${CMD[@]}" >"$LOG" 2>&1 < /dev/null &
  nohup setsid "${CMD[@]}" >>"$LOG" 2>&1 &
  echo $! > "$SCRATCHDIR/worker_${HOSTNAME}_${WORKER_IDX}.pid"
  echo "Started Dask worker on $HOSTNAME, idx=$WORKER_IDX, pid=$(cat "$SCRATCHDIR/worker_${HOSTNAME}_${WORKER_IDX}.pid")"
done
