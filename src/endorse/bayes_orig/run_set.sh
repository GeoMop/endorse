#!/bin/bash
set -x

# output directory
output_dir=$1

# number of Markov chains
csv_data=$2

if [ "$csv_data" == "0" ]; then
  exit 0
fi

# sing == true => use singularity
# sing == false => use docker
sing=false
if [ "$3" == "sing" ]; then
  sing=true
fi


# Development: root of the sources
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/../../.."
ENDORSE_SRC_BAYES="${SCRIPTPATH}"
ENDORSE_VENV_BAYES="${ENDORSE_SRC_ROOT}/venv_bayes"

# visualize
command="source ${ENDORSE_VENV_BAYES}/bin/activate && python3 ${ENDORSE_SRC_BAYES}/run_set_flow123d.py $output_dir $csv_data"


if [ "$sing" == true ]; then

  sif_image="${ENDORSE_SRC_ROOT}/endorse.sif"
  sing_command="singularity exec -B ${ENDORSE_SRC_ROOT}:${ENDORSE_SRC_ROOT} $sif_image"

  final_command="${sing_command} bash -c \"${command}\""
  echo ${final_command}
  eval "${final_command}"
else

  final_command="bash -c \"${command}\""
  echo ${final_command}

  if [ "$debug" == false ]; then
    ./endorse_fterm exec ${final_command}
  fi
fi
