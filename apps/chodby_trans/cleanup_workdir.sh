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
tar_cd() {
  local root=$1
  local delete=${2:-0}   # 0 = keep sources, 1 = delete sources
  local archive=$3
  local nproc=4
  shift 3                # now "$@" = source paths/patterns
  ( cd "$root" || exit 1
    tar -I "pigz -9 -p $(nproc)" -cf "$archive" $@
    if (( delete )); then
      echo "Deleting sources in '$root': $*"
      rm -rf -- $@
    fi
  )
}

echo "Compressing zarr storage..."
tar_cd $ROOT 0 transport_sampling.tar.gz transport_sampling

echo "Compressing logs..."
tar_cd $ROOT $DELETE logs.tar.gz logs_charon*

# echo "Compressing nodes workdirs..."
# tar_cd $ROOT $DELETE workers.tar.gz workdir_charon*

# list
# tar -I pigz -tf workers.tar.gz | head
# tar -tzf workers.tar.gz | head
# extract
# tar -I pigz -xf workers.tar.gz
# tar -xzf workers.tar.gz

