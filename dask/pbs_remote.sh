#!/usr/bin/env bash
# pbsdsh may report a remote exit status without returning it to the caller.
# Require an acknowledgement emitted only after the remote command succeeds.
pbs_remote() {
  local node=$1
  shift
  local output status=0
  output=$(pbsdsh -vh "$node" -- bash -c '
    set -euo pipefail
    "$@"
    printf "\nENDORSE_PBS_REMOTE_OK\n"
  ' bash "$@") || status=$?
  printf '%s\n' "$output" | sed '/^ENDORSE_PBS_REMOTE_OK$/d'
  if (( status != 0 )); then
    echo "ERROR: pbsdsh failed on $node (status $status): $*" >&2
    return "$status"
  fi
  if ! grep -Fxq 'ENDORSE_PBS_REMOTE_OK' <<< "$output"; then
    echo "ERROR: Remote command failed on $node: $*" >&2
    return 1
  fi
}
