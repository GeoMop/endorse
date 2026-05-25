#!/bin/bash
# dask_cluster.sh
# Start/stop a Dask (no MPI) cluster across all nodes in the *current PBS interactive job*.
# It stages input to each node's $SCRATCHDIR, starts scheduler + workers, and prints the scheduler address.

set -euo pipefail

# ======= EDIT THESE PATHS =======
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${1}
#WORK_DIRNAME="workdir"

APP_PY="$PROJECT_DIR/sensitivity_sampling.py"
VENV="$PROJECT_DIR/venv"
APP_CMD_DEFAULT="meta"
# =================================

UNIQ_HOSTS=$(sort -u "$PBS_NODEFILE")
NCPUS=$(wc -l < "$PBS_NODEFILE")

PYEXEC="$VENV/bin/python"
DASK_BIN="$VENV/bin/dask"

# not necessary at the end, `dask worker --no-nanny` solved it
# export DASK_DISTRIBUTED__WORKER__DAEMON=False

require_env() {
  for v in SCRATCHDIR PBS_NODEFILE; do
    [[ -n "${!v:-}" ]] || { echo "ERROR: $v is not set. Run inside a PBS interactive job."; exit 1; }
  done
}

make_venv() {
  if [[ ! -x "$PYEXEC" ]]; then
    bash setup_venv
  fi
}

stage_inputs() {
  echo "[stage] Staging inputs to each node's \$SCRATCHDIR..."
  for node in $UNIQ_HOSTS; do
    echo "--- $node ---"
    pbsdsh -vh "$node" -- mkdir -p "$SCRATCHDIR/logs" "$SCRATCHDIR/dask"
    # pbsdsh -vh "$node" -- mkdir -p "$SCRATCHDIR/$WORK_DIRNAME"
    # pbsdsh -vh "$node" -- rsync -a "$OUTPUT_DIR/" "$SCRATCHDIR/"
    pbsdsh -vh "$node" -- bash -lc "ls -la \"\$SCRATCHDIR\""
  done
  wait
}

compute_host_counts() {
  # produces lines like: "8 charon21"
  HOST_COUNTS=$(sort "$PBS_NODEFILE" | uniq -c)
}

start_scheduler() {
  HEAD_NODE=$(head -n1 "$PBS_NODEFILE")
  HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
  SCHED_ADDR="tcp://${HEAD_IP}:8786"
  DASH_ADDR=":8787"

  echo "[sched] Starting scheduler on $HEAD_NODE ($SCHED_ADDR)..."
  nohup "$DASK_BIN" scheduler --host "$HEAD_IP" --port 8786 --dashboard-address "$DASH_ADDR" > "$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &
  # nohup "$DASK_SCHED" --host "$HEAD_IP" --port 8786 --dashboard-address "$DASH_ADDR" > "$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &
  # pbsdsh -vh "$HEAD_NODE" -- bash -lc `nohup "$DASK_SCHED" --host "$HEAD_IP" --port 8786 --dashboard-address "$DASH_ADDR" > "$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &`
  # pbsdsh -vh "$HEAD_NODE" -- cat "$SCRATCHDIR/logs/scheduler.log"
  echo $! > "$SCRATCHDIR/scheduler.pid"

  # give it a moment
  sleep 3
  # cat "pid: $SCRATCHDIR/scheduler.pid"
  # cat "$SCRATCHDIR/logs/scheduler.log"
  echo "$SCHED_ADDR" > "$SCRATCHDIR/SCHED_ADDR.txt"
  echo "[sched] Scheduler: $SCHED_ADDR (dashboard on $HEAD_IP$DASH_ADDR)"
}

start_workers() {
  echo "[worker] Launching workers on each node (one process per PBS slot, 1 thread each)..."
  echo "$SCHED_ADDR"
  echo "$HOST_COUNTS"
  while read -r COUNT HOST; do
    [[ -z "${HOST:-}" ]] && continue
    # echo "--- $HOST --- "
    # pbsdsh -vh "$HOST" -- python --version
    # pbsdsh -vh "$HOST" -- bash "$PROJECT_DIR"/dask_process_start.sh "$DASK_BIN" "$SCHED_ADDR" "$COUNT"
    echo "--- $HOST ($COUNT slots) ---"
    pbsdsh -vh "$HOST" -- bash "$PROJECT_DIR/dask_process_start.sh" "$DASK_BIN" "$SCHED_ADDR" "$COUNT"
  done <<< "$HOST_COUNTS"
  echo "[worker] Workers started."
}

stop_cluster() {
  echo "[stop] Stopping workers, copying results from scratchdir..."
  for node in $(sort -u "$PBS_NODEFILE"); do
    echo "--- $node ---"
    pbsdsh -vh "$node" -- pkill -f "$DASK_BIN worker" || true &
    pbsdsh -vh "$node" -- bash "$PROJECT_DIR/dask_process_stop.sh" worker &

    pbsdsh -vh "$node" -- bash -lc "ls -la \"\$SCRATCHDIR\""
    # pbsdsh -vh "$node" -- rsync -a "$SCRATCHDIR/$WORK_DIRNAME/" "$OUTPUT_DIR/workdir_$node" &
    pbsdsh -vh "$node" -- rsync -a "$SCRATCHDIR/logs/" "$OUTPUT_DIR/logs_$node/" &
    pbsdsh -vh "$node" -- rm -r "$SCRATCHDIR/*"
  done
  wait
  
  echo "[stop] Stopping scheduler..."
  HEAD_NODE=$(head -n1 "$PBS_NODEFILE")
  pbsdsh -vh "$HEAD_NODE" -- bash "$PROJECT_DIR/dask_process_stop.sh" scheduler
  rm -r "$SCRATCHDIR/*"
  
  echo "[stop] Done."
}

status_cluster() {
  echo "---- Dask cluster status ----"
  if [[ -f "$SCRATCHDIR/SCHED_ADDR.txt" ]]; then
    echo "Scheduler: $(cat "$SCRATCHDIR/SCHED_ADDR.txt")"
  else
    echo "Scheduler: not found"
  fi
  echo "Logs on each node: \$SCRATCHDIR/logs/{scheduler.log,worker_<host>.log}"
  echo "-----------------------------"
}

run_example() {
  # Run your driver against the live scheduler (can call this many times).
  SCHED=$(cat "$SCRATCHDIR/SCHED_ADDR.txt")
  echo "[run] Running driver against $SCHED ..."

  local app_cmd=${1}
  echo "app_cmd = $app_cmd"

  "$PYEXEC" -u "$APP_PY" $OUTPUT_DIR $app_cmd "$SCHED" \
      2>&1 | tee "$SCRATCHDIR/logs/driver_$(date +%H%M%S).log"
}

cmd=${2:-help}
case "$cmd" in
  start)
    require_env
    source $PROJECT_DIR/load_modules
    make_venv
    stage_inputs
    compute_host_counts
    start_scheduler
    start_workers
    status_cluster
    echo
    echo "Run your app any time with:  bash $0 $OUTPUT_DIR run"
    echo "Print dask status with:      bash $0 $OUTPUT_DIR status"
    echo "Stop cluster with:           bash $0 $OUTPUT_DIR stop"
    ;;
  run)
    require_env
    run_example "${3:-$APP_CMD_DEFAULT}"
    ;;
  stop)
    require_env
    stop_cluster
    ;;
  status)
    require_env
    status_cluster
    ;;
  help|*)
    cat <<EOF
Usage: bash dask_cluster.sh <workdir> <start|run|status|stop>

 start  - stage inputs to \$SCRATCHDIR on all nodes and start Dask scheduler/workers
 run    - run your driver once against the live scheduler (repeat as you wish)
 status - print scheduler address and log hints
 stop   - stop workers and scheduler

EOF
    ;;
esac
