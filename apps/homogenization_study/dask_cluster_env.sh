#!/bin/bash

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENDORSE_DASK_APP_DIR="$APP_DIR"
export ENDORSE_DASK_APP_MODULE="homogenization_study"
export ENDORSE_DASK_WORKER_MODULE="homogenization_study.problem_simulation"
export ENDORSE_DASK_SCHEDULER_MODULE="homogenization_study.mlmc_worker"
export ENDORSE_DASK_WORK_DIRNAME="workdir"

exec bash "$APP_DIR/../../dask/dask_cluster.sh" "$@"
