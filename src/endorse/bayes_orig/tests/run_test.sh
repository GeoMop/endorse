#!/bin/bash
# set -x

testname=$1
config=$2

# Setup paths
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/../../../.."
ENDORSE_SRC_BAYES="${SCRIPTPATH}"

# Define config and workdir
config=$SCRIPTPATH/$config.yaml
workdir=$ENDORSE_SRC_ROOT/tests/sandbox/$testname

# copy config to workdir
if [ -d "$workdir" ];
then
    echo "$workdir directory already exists!"
    exit 1
    # rm -r $workdir
else
    mkdir -p $workdir
fi

cp "${config}" "${workdir}/config.yaml"

# run simulation
${ENDORSE_SRC_ROOT}/bin/endorse-bayes -t run -n 19 -o $workdir
