#!/usr/bin/env bash
# pbsdsh may report a remote exit status without returning it to the caller.
# Check its verbose completion diagnostic; remote task stdout may bypass pbsdsh.
pbs_remote() {
  local node=$1
  shift
  local output command status=0
  # Keep multiline argv out of PBS launch hooks. Transport shell-quoted arguments
  # on one line; remote Bash restores them before exec, including literal newlines.
  printf -v command '%q ' "$@"
  output=$(pbsdsh -vh "$node" -- bash -c "exec $command" 2>&1) || status=$?
  printf '%s\n' "$output"
  if (( status != 0 )); then
    echo "ERROR: pbsdsh failed on $node (status $status): $*" >&2
    return "$status"
  fi
  local line remote_status='' completions=0
  local completion_pattern='^pbsdsh: task 0x[[:xdigit:]]+ exit status ([0-9]+)$'
  while IFS= read -r line; do
    if [[ "$line" == 'pbsdsh: task '*' exit status '* ]]; then
      (( completions += 1 ))
      if [[ "$line" =~ $completion_pattern ]]; then
        remote_status=${BASH_REMATCH[1]}
      fi
    fi
  done <<< "$output"
  if (( completions != 1 )) || [[ -z "$remote_status" ]]; then
    echo "ERROR: Expected one valid PBS task completion on $node, got $completions: $*" >&2
    return 1
  fi
  remote_status=$((10#$remote_status))
  if (( remote_status != 0 )); then
    echo "ERROR: Remote command failed on $node (status $remote_status): $*" >&2
    if (( remote_status > 255 )); then return 1; fi
    return "$remote_status"
  fi
  return 0
}
