#!/bin/bash
# dask_cluster.sh
# Start/stop a Dask (no MPI) cluster across all nodes in the *current PBS interactive job*.
# It stages input to each node's $SCRATCHDIR, starts scheduler + workers, and prints the scheduler address.

set -euo pipefail

# ======= EDIT THESE PATHS =======
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${1}

APP_PY="$PROJECT_DIR/sensitivity_sampling.py"
VENV="$PROJECT_DIR/venv"
# =================================

PYEXEC="$VENV/bin/python"
DASK_BIN="$VENV/bin/dask"

LOG_DIR="$OUTPUT_DIR/logs"
RUNTIME_DIR="$OUTPUT_DIR/dask"
mkdir -p "$LOG_DIR" "$RUNTIME_DIR"

SCHED_ADDR_FILE="$RUNTIME_DIR/SCHED_ADDR.txt"
SCHED_PID_FILE="$RUNTIME_DIR/scheduler.pid"
DASH_PORT="${DASH_PORT:-8787}"
SCHED_PORT="${SCHED_PORT:-8786}"

# not necessary at the end, `dask worker --no-nanny` solved it
# export DASK_DISTRIBUTED__WORKER__DAEMON=False

require_env() {
  if [[ ! -x "$PYEXEC" ]]; then
    echo "[venv] Python not found in $PYEXEC"
    exit 1
  fi
}


start_scheduler() {
  local host_ip="127.0.0.1"
  local sched_addr="tcp://${host_ip}:${SCHED_PORT}"
  echo "[sched] Starting scheduler at ${sched_addr} (dashboard :${DASH_PORT})"
  nohup "$DASK_BIN" scheduler \
      --host "$host_ip" \
      --port "$SCHED_PORT" \
      --dashboard-address ":${DASH_PORT}" \
      >"$LOG_DIR/scheduler.log" 2>&1 < /dev/null &

  echo $! > "$SCHED_PID_FILE"
  # wait briefly so the port opens
  sleep 2
  echo "$sched_addr" > "$SCHED_ADDR_FILE"
  echo "[sched] PID $(cat "$SCHED_PID_FILE"), address $(cat "$SCHED_ADDR_FILE")"
}

start_workers() {
  local procs="${1:-1}"   # number of worker processes (1–4)
  if (( procs < 1 || procs > 4 )); then
    echo "[worker] Invalid worker count '$procs' (try 1–4)"; exit 1
  fi
  if [[ ! -f "$SCHED_ADDR_FILE" ]]; then
    echo "[worker] Scheduler address file not found: $SCHED_ADDR_FILE"
    exit 1
  fi
  local addr
  addr="$(cat "$SCHED_ADDR_FILE")"

  echo "[worker] Starting $procs worker process(es), 1 thread each, no nanny"

  CMD=( "$DASK_BIN" worker "$addr"
      --nworkers 1 --nthreads 1
      --no-nanny
      --local-directory "$RUNTIME_DIR"
      --memory-limit auto )

  for ((WORKER_IDX=0; WORKER_IDX<procs; WORKER_IDX++)); do
    LOG="$LOG_DIR/worker_${WORKER_IDX}.log"
    nohup setsid "${CMD[@]}" >"$LOG" 2>&1 < /dev/null &
    echo $! > "$RUNTIME_DIR/worker_${WORKER_IDX}.pid"
    echo "Started Dask worker on $HOSTNAME, idx=$WORKER_IDX, pid=$(cat "$RUNTIME_DIR/worker_${WORKER_IDX}.pid")"
  done

  echo "[worker] Workers started."
}

stop_cluster() {
  echo "[stop] Stopping workers..."
  # shopt -s nullglob
  worker_pidfiles=( "$RUNTIME_DIR"/worker_*.pid )
  if (( ${#worker_pidfiles[@]} == 0 )); then
    echo "  (no worker pid files found in $RUNTIME_DIR)"
  else
    for pidfile in "${worker_pidfiles[@]}"; do
      
      pkill -TERM -P "$(cat "$pidfile")" 2>/dev/null || true
      kill "$(cat "$pidfile")" 2>/dev/null || true
      rm -f "$pidfile"
      
      # Extra safety for any leftover worker processes:
      pgrep -f "dask .*worker .*tcp://" >/dev/null && pkill -f "dask .*worker .*tcp://" || true
    done
  fi

  echo "[stop] Stopping scheduler..."
  if [[ -f "$SCHED_PID_FILE" ]]; then
    kill "$(cat "$SCHED_PID_FILE")" 2>/dev/null || true
    rm -f "$SCHED_PID_FILE"
  fi
  pgrep -f "dask .*scheduler" >/dev/null && pkill -f "dask .*scheduler" || true

  rm -f "$SCHED_ADDR_FILE"
  echo "[stop] Done."
}

status_cluster() {
  echo "---- Dask local cluster status ----"
  if [[ -f "$SCHED_ADDR_FILE" ]]; then
    echo "Scheduler: $(cat "$SCHED_ADDR_FILE")"
    echo "Dashboard: http://127.0.0.1:${DASH_PORT}/status"
  else
    echo "Scheduler: not running"
  fi
  [[ -f "$SCHED_PID_FILE" ]] && echo "Scheduler PID: $(cat "$SCHED_PID_FILE")"
  echo "-----------------------------------"
}

run_example() {
  if [[ ! -f "$SCHED_ADDR_FILE" ]]; then
    echo "[run] No scheduler found. Start it first: bash $0 start"
    exit 1
  fi
  local sched
  sched="$(cat "$SCHED_ADDR_FILE")"
  echo "[run] $APP_PY -> $sched"
  "$PYEXEC" -u "$APP_PY" $OUTPUT_DIR local "$sched" 2>&1 | tee "$LOG_DIR/driver_$(date +%H%M%S).log"
}

cmd=${2:-help}
case "$cmd" in
  start)
    require_env
    start_scheduler
    start_workers "${3:-1}"
    status_cluster
    echo
    echo "Run your app any time with:  bash $0 $OUTPUT_DIR run"
    echo "Print dask status with:      bash $0 $OUTPUT_DIR status"
    echo "Stop cluster with:           bash $0 $OUTPUT_DIR stop"
    ;;
  run)
    require_env
    run_example
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
Usage: bash dask_cluster.sh <workdir> <command>

 workdir - path to working directory

 command:
 start [N] - start local scheduler + N worker processes (default N=1)
 run       - run your driver once against the live scheduler (repeat as you wish)
 status    - print scheduler address and log hints
 stop      - stop workers and scheduler

EOF
    ;;
esac
