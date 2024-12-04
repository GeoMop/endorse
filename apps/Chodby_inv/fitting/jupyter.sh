#!/bin/bash
set -x
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


# Extract the name field from the YAML file
env_yaml="$SCRIPTPATH/conda-requirements.yml"
env_name=$(cat "$env_yaml" | grep '^name:'  | awk '{print $2}')

$SCRIPTPATH/create_env.sh
source /opt/miniconda/etc/profile.d/conda.sh

conda activate $env_name
jupyter-lab --notebook-dir=${SCRIPTPATH}
