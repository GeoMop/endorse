#!/usr/bin/env bash
# remove_joblib_cache.sh — simple finder/remover for "joblib_cache" dirs
set -Eeuo pipefail

# Usage: ./remove_joblib_cache.sh [--delete] ROOT_DIR
DELETE=0
if [[ ${1:-} == "--delete" ]]; then DELETE=1; shift; fi

[[ $# -eq 1 ]] || { echo "Usage: $0 [--delete] ROOT_DIR" >&2; exit 2; }
ROOT=$1
[[ -d "$ROOT" ]] || { echo "No such directory: $ROOT" >&2; exit 1; }

echo "Scanning for 'joblib_cache' under: $ROOT"

# Preview matches (no deletion yet)
mapfile -t MATCHES < <(find "$ROOT" -type d -name joblib_cache -prune -print 2>/dev/null)

for d in "${MATCHES[@]}"; do echo "$d"; done
count=${#MATCHES[@]}
echo "Found $count director$( [[ $count -eq 1 ]] && echo 'y' || echo 'ies')."

if (( DELETE )); then
  if (( count == 0 )); then
    echo "Nothing to delete."
    # exit 0
  else
    #read -r -p "Type 'delete' to remove ALL listed directories: " ans
    #[[ "$ans" == "delete" ]] || { echo "Aborted."; exit 1; }

    echo "Deleting..."
    # Run the find again to pipe safely to rm (handles spaces via -print0)
    find "$ROOT" -type d -name joblib_cache -prune -print0 2>/dev/null | xargs -0 -r rm -rf --
    echo "Done."
  fi
else
  echo "Dry-run only. Re-run with --delete to actually remove them."
fi


# COMPRESS output
nproc=4
# create
# zip -r $ROOT/logs.zip $ROOT/logs_charon*
# zip -r $ROOT/workers.zip $ROOT/workdir_charon*
echo "Compressing zarr storage..."
tar cf - $ROOT/transport_sampling | pigz -9 -p $(nproc) > $ROOT/transport_sampling.tar.gz
if (( DELETE )); then
  echo "Deleting zarr storage..."
  rm -r $ROOT/transport_sampling
fi

echo "Compressing logs..."
tar cf - $ROOT/logs_charon* | pigz -9 -p $(nproc) > $ROOT/logs.tar.gz
if (( DELETE )); then
  echo "Deleting logs..."
  rm -r $ROOT/logs_charon*
fi

echo "Compressing nodes workdirs..."
tar cf - $ROOT/workdir_charon* | pigz -9 -p $(nproc) > $ROOT/workers.tar.gz
if (( DELETE )); then
  echo "Deleting nodes workdirs..."
  rm -r $ROOT/workdir_charon*
fi

# list
# tar -I pigz -tf workers.tar.gz | head
# tar -tzf workers.tar.gz | head
# extract
# tar -I pigz -xf workers.tar.gz
# tar -xzf workers.tar.gz

