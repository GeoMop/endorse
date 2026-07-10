#!/usr/bin/env bash
set -euo pipefail

# workdirs=(
#   "workdir_40b_ot_2048"
#   "workdir_40c_ot_2048"
#   "workdir_41d_test_ot_2048"
#   "workdir_42b_test_ot_2048"
#   "workdir_42c_test_ot_2048"
#   "workdir_43a_ot_2048"
#   "workdir_43b_ot_2048"
# )
workdirs=(CASE_0_32k/job_*_*_*)
STORAGE="/storage/liberec3-tul/projects/flow123d/chodby_trans"


# Location of the script you want to call
MAKE_PLOTS="./make_plots.sh"

[[ -x "$MAKE_PLOTS" ]] || { echo "Error: $MAKE_PLOTS not found or not executable"; exit 1; }

for wd in "${workdirs[@]}"; do
  path="$STORAGE/$wd"
  echo "==> Running: $MAKE_PLOTS $path"
  "$MAKE_PLOTS" "$path"
done

echo "All done."