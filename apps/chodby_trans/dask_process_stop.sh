#!/usr/bin/env bash
# dask_process_stop.sh
# Usage: dask_process_stop.sh [scheduler|worker]

# pbsdsh -vh "$HEAD_NODE" -- bash /path/to/dask_process_stop.sh scheduler
# pbsdsh -vh "$HOST"      -- bash /path/to/dask_process_stop.sh worker

set -euo pipefail
: "${SCRATCHDIR:?}"

case "${1:-}" in
  scheduler)
    [[ -f "$SCRATCHDIR/scheduler.pid" ]] && kill "$(cat "$SCRATCHDIR/scheduler.pid")" || true
    rm -f "$SCRATCHDIR/scheduler.pid"
    ;;
  worker)
    for f in "$SCRATCHDIR"/worker_*.pid; do
      [[ -f "$f" ]] || continue
      kill "$(cat "$f")" || true
      rm -f "$f"
    done
    ;;
  *)
    echo "Usage: $0 scheduler|worker" >&2
    exit 1
esac

