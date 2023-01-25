#!/bin/bash
set -x

data_dir=${1:-`pwd`}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
paraview  --state "$SCRIPT_DIR/transport_vis.pvsm" --data-directory "$data_dir"
