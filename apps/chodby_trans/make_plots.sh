#!/usr/bin/env bash

set -euo pipefail

WORKDIR=$1

# =================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PY="$PROJECT_DIR/sensitivity_sampling.py"
VENV="$PROJECT_DIR/venv"
PYEXEC="$VENV/bin/python"
# =================================

mkdir -p "$WORKDIR/plots"
app_cmd="read"
"$PYEXEC" -u "$APP_PY" "$WORKDIR" "$app_cmd" 2>&1 | tee "$WORKDIR/plots/sensitivity_read.log"
app_cmd="plots"
"$PYEXEC" -u "$APP_PY" "$WORKDIR" "$app_cmd" 2>&1 | tee "$WORKDIR/plots/sensitivity_plots.log"