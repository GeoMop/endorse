#!/bin/bash
set -x

function safe_mkdir() {
  DIR_PATH="$1"
  if [ ! -d "$1" ]
  then
      mkdir -p "$1"
  fi
}

# Setup paths
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/../../../.."
ENDORSE_SRC_BAYES="${SCRIPTPATH}"

# Define config and workdir
config=$SCRIPTPATH/config_bayes_min_test.yaml
workdir=$ENDORSE_SRC_ROOT/tests/sandbox/bayes_min_test

# copy config to workdir
rm -r $workdir
safe_mkdir $workdir
cp "${config}" "${workdir}/config.yaml"

# run simulation
${ENDORSE_SRC_ROOT}/bin/endorse-bayes -t run -n 19 -o $workdir