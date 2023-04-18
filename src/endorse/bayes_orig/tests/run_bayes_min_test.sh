#!/bin/bash
set -x

TESTNAME=bayes_min_test

# Setup paths
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/../../../.."
ENDORSE_SRC_BAYES="${SCRIPTPATH}"

# Define config and workdir
config=$SCRIPTPATH/config_$TESTNAME.yaml
workdir=$ENDORSE_SRC_ROOT/tests/sandbox/$TESTNAME

# copy config to workdir
rm -r $workdir
mkdir -p $workdir
cp "${config}" "${workdir}/config.yaml"

# run simulation
${ENDORSE_SRC_ROOT}/bin/endorse-bayes -t run -n 2 -o $workdir