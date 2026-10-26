#!/bin/bash
# dask_cluster.sh
# Start/stop a Dask (no MPI) cluster across all nodes in the *current PBS interactive job*.
# It stages input to each node's $SCRATCHDIR, starts scheduler + workers, and prints the scheduler address.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Application wrappers define the package-specific paths and preload modules.
: "${ENDORSE_DASK_APP_DIR:?ENDORSE_DASK_APP_DIR is not set}"
: "${ENDORSE_DASK_APP_MODULE:?ENDORSE_DASK_APP_MODULE is not set}"
: "${ENDORSE_DASK_WORKER_MODULE:?ENDORSE_DASK_WORKER_MODULE is not set}"
: "${ENDORSE_DASK_SCHEDULER_MODULE:?ENDORSE_DASK_SCHEDULER_MODULE is not set}"

# ======= SETUP PATHS =======
DASK_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ENDORSE_DASK_APP_DIR"
OUTPUT_DIR=${1}
EXPECTED_WORKERS_FILE="$SCRATCHDIR/DASK_EXPECTED_WORKERS.txt"
# WORK_DIRNAME="${ENDORSE_DASK_WORK_DIRNAME:-}"

APP_PY="$PROJECT_DIR/sensitivity_sampling.py"
VENV="$PROJECT_DIR/venv"
APP_CMD_DEFAULT="meta"
# =================================

UNIQ_HOSTS=$(sort -u "$PBS_NODEFILE")
NCPUS=$(wc -l < "$PBS_NODEFILE")
HEAD_WORKER_RESERVE=${DASK_HEAD_WORKER_RESERVE:-2}

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
    bash "$PROJECT_DIR/setup_venv"
  fi
}

stage_inputs() {
  echo "[stage] Staging inputs to each node's \$SCRATCHDIR..."
  for node in $UNIQ_HOSTS; do
    echo "--- $node ---"
    pbsdsh -vh "$node" -- mkdir -p "$SCRATCHDIR/logs" "$SCRATCHDIR/dask"
#    if [[ -n "$WORK_DIRNAME" ]]; then
#      pbsdsh -vh "$node" -- mkdir -p "$SCRATCHDIR/$WORK_DIRNAME"
#    fi
#    pbsdsh -vh "$node" -- rsync -a "$OUTPUT_DIR/" "$SCRATCHDIR/"
    pbsdsh -vh "$node" -- bash -lc "ls -la \"\$SCRATCHDIR\""
  done
  wait
}

compute_host_counts() {
  # produces lines like: "8 charon21"
  HOST_COUNTS=$(sort "$PBS_NODEFILE" | uniq -c)
}

wait_for_scheduler() {
  local timeout="${DASK_SCHEDULER_START_TIMEOUT:-180}"
  local deadline=$((SECONDS + timeout))
  local scheduler_log="$SCRATCHDIR/logs/scheduler.log"
  local scheduler_pid
  scheduler_pid=$(<"$SCRATCHDIR/scheduler.pid")

  echo "[sched] Waiting up to ${timeout}s for scheduler startup..."
  while (( SECONDS < deadline )); do
    if grep -Fq "Scheduler at:" "$scheduler_log"; then
      echo "[sched] Scheduler is accepting connections."
      return 0
    fi
    if ! kill -0 "$scheduler_pid" 2>/dev/null; then
      echo "ERROR: Dask scheduler exited during startup. Last log lines:" >&2
      tail -50 "$scheduler_log" >&2
      return 1
    fi
    sleep 1
  done

  echo "ERROR: Dask scheduler did not start within ${timeout}s. Last log lines:" >&2
  tail -50 "$scheduler_log" >&2
  return 1
}

start_scheduler() {
  HEAD_NODE=$(head -n1 "$PBS_NODEFILE")
  SCHED_ADDR="tcp://${HEAD_NODE}:8786"
  DASH_ADDR=":8787"

  echo "[sched] Starting scheduler on $HEAD_NODE ($SCHED_ADDR)..."
  nohup "$DASK_BIN" scheduler \
      --host "$HEAD_NODE" \
      --port 8786 \
      --dashboard-address "$DASH_ADDR" \
      --preload "$DASK_HELPER_DIR/dask_scheduler_preload.py" \
      >"$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &
  # nohup "$DASK_SCHED" --host "$HEAD_NODE" --port 8786 --dashboard-address "$DASH_ADDR" > "$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &
  # pbsdsh -vh "$HEAD_NODE" -- bash -lc `nohup "$DASK_SCHED" --host "$HEAD_IP" --port 8786 --dashboard-address "$DASH_ADDR" > "$SCRATCHDIR/logs/scheduler.log" 2>&1 < /dev/null &`
  # pbsdsh -vh "$HEAD_NODE" -- cat "$SCRATCHDIR/logs/scheduler.log"
  echo $! > "$SCRATCHDIR/scheduler.pid"

  wait_for_scheduler
  # cat "pid: $SCRATCHDIR/scheduler.pid"
  # cat "$SCRATCHDIR/logs/scheduler.log"
  echo "$SCHED_ADDR" > "$SCRATCHDIR/SCHED_ADDR.txt"
  echo "[sched] Scheduler: $SCHED_ADDR (dashboard on $HEAD_NODE$DASH_ADDR)"
}

start_workers() {
  echo "[worker] Launching workers on each node (one process per worker slot, 1 thread each)..."
  echo "[worker] Reserving $HEAD_WORKER_RESERVE slot(s) on scheduler node $HEAD_NODE."
  echo "$SCHED_ADDR"
  echo "$HOST_COUNTS"
  EXPECTED_WORKERS=0
  while read -r COUNT HOST; do
    [[ -z "${HOST:-}" ]] && continue
    WORKER_COUNT=$COUNT
    if [[ "$HOST" == "$HEAD_NODE" ]]; then
      WORKER_COUNT=$((COUNT - HEAD_WORKER_RESERVE))
      if (( WORKER_COUNT < 0 )); then
        WORKER_COUNT=0
      fi
    fi
    if (( WORKER_COUNT == 0 )); then
      echo "--- $HOST ($COUNT slots, $WORKER_COUNT workers; scheduler reserved) ---"
      continue
    fi
    EXPECTED_WORKERS=$((EXPECTED_WORKERS + WORKER_COUNT))
    # echo "--- $HOST --- "
    # pbsdsh -vh "$HOST" -- python --version
    # pbsdsh -vh "$HOST" -- bash "$PROJECT_DIR"/dask_process_start.sh "$DASK_BIN" "$SCHED_ADDR" "$COUNT"
    echo "--- $HOST ($COUNT slots, $WORKER_COUNT workers) ---"
    pbsdsh -vh "$HOST" -- env \
      ENDORSE_DISABLE_MEMOIZE="${ENDORSE_DISABLE_MEMOIZE:-}" \
      ENDORSE_DASK_APP_MODULE="$ENDORSE_DASK_APP_MODULE" \
      ENDORSE_DASK_WORKER_MODULE="$ENDORSE_DASK_WORKER_MODULE" \
      ENDORSE_DASK_SCHEDULER_MODULE="$ENDORSE_DASK_SCHEDULER_MODULE" \
      ENDORSE_INPUT_DIR="$OUTPUT_DIR/input_data" \
      ENDORSE_OUTPUT_DIR="$OUTPUT_DIR" \
      bash "$DASK_HELPER_DIR/dask_process_start.sh" \
        "$DASK_BIN" "$SCHED_ADDR" "$WORKER_COUNT" "$DASK_HELPER_DIR/dask_worker_preload.py"
  done <<< "$HOST_COUNTS"
  printf '%s\n' "$EXPECTED_WORKERS" > "$EXPECTED_WORKERS_FILE"
  echo "[worker] Workers started; expecting $EXPECTED_WORKERS registrations."
}

stop_cluster() {
  echo "[stop] Stopping workers, copying results from scratchdir..."
  for node in $(sort -u "$PBS_NODEFILE"); do
    echo "--- $node ---"
    pbsdsh -vh "$node" -- pkill -f "$DASK_BIN worker" || true &
    pbsdsh -vh "$node" -- bash "$DASK_HELPER_DIR/dask_process_stop.sh" worker &

    pbsdsh -vh "$node" -- bash -lc "ls -la \"\$SCRATCHDIR\""
#    if [[ -n "$WORK_DIRNAME" ]]; then
#      pbsdsh -vh "$node" -- rsync -a \
#        "$SCRATCHDIR/$WORK_DIRNAME/" "$OUTPUT_DIR/${WORK_DIRNAME}_$node/" &
#    fi
    pbsdsh -vh "$node" -- rsync -a "$SCRATCHDIR/logs/" "$OUTPUT_DIR/logs_$node/"
  done
  wait
  
  echo "[stop] Stopping scheduler..."
  HEAD_NODE=$(head -n1 "$PBS_NODEFILE")
  pbsdsh -vh "$HEAD_NODE" -- bash "$DASK_HELPER_DIR/dask_process_stop.sh" scheduler
  rm -rf "$SCRATCHDIR"/*
  
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
  local expected_workers
  expected_workers=$(<"$EXPECTED_WORKERS_FILE")
  echo "app_cmd = $app_cmd"

  DASK_EXPECTED_WORKERS="$expected_workers" \
  "$PYEXEC" -u "$APP_PY" "$OUTPUT_DIR" "$app_cmd" "$SCHED" \
      2>&1 | tee "$SCRATCHDIR/logs/driver_$(date +%H%M%S).log"
}

cmd=${2:-help}
case "$cmd" in
  start)
    require_env
    source "$PROJECT_DIR/load_modules"
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
