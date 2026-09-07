#!/usr/bin/env bash
set -euo pipefail

app_dir=$(cd "$(dirname "$0")/.." && pwd)
run_dir=${1:-"$app_dir/runs"}
work_dir="$run_dir/workdir"

mkdir -p "$work_dir"
exec "$app_dir/venv/bin/python" "$app_dir/model/run_model.py" \
    --config "$app_dir/input_data/config.yaml" \
    --work-dir "$work_dir"
