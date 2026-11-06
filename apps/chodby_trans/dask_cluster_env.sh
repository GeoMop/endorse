#!/bin/bash

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENDORSE_DASK_APP_DIR="$APP_DIR"
export ENDORSE_DASK_APP_MODULE="chodby_trans"
export ENDORSE_DASK_WORKER_MODULE="chodby_trans.transport_simulation"
export ENDORSE_DASK_SCHEDULER_MODULE="chodby_trans.mlmc_worker"

exec bash "$APP_DIR/../../dask/dask_cluster.sh" "$@"
