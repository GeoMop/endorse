#!/bin/bash

# set -x

debug=false
# sing == true => use singularity
# sing == false => use docker
sing=false

obligatory_param=0
while getopts ":hdn:o:t:cs" opt; do
  case $opt in
    h)
      # help
      echo "Usage: ./run_all_local.sh -n <N_CHAINS> -o <OUTPUT_DIR> -s -d"
      echo "-s ... runs the program in Singularity container"
      echo "-d ... only print the container command"
      exit 0
      ;;
    d)
      # debug
      debug=true
      ;;
    n)
      # number of Markov chains
      n_chains=$OPTARG
      ((obligatory_param=obligatory_param+1))
      ;;
    o)
      # output directory
      output_dir=$OPTARG
      ((obligatory_param=obligatory_param+1))
      ;;
    s)
      # output directory
      sing=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ "$debug" == true ]; then
  echo "n_chains = $n_chains"
  echo "output_dir = $output_dir"
  echo "visualize = $visualize"
  echo "sing = $sing"
fi

if [[ $obligatory_param -lt 2 ]]; then
  echo "Not all obligatory parameters set!"
  exit 1
fi


# Development: root of the sources
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/../../.."
ENDORSE_SRC_BAYES="${SCRIPTPATH}"
ENDORSE_VENV_BAYES="${ENDORSE_SRC_ROOT}/venv_bayes"

# visualize
command="source ${ENDORSE_VENV_BAYES}/bin/activate && python3 ${ENDORSE_SRC_BAYES}/run_all.py $output_dir $n_chains '' visualize"


if [ "$sing" == true ]; then

  sif_image="${ENDORSE_SRC_ROOT}/endorse.sif"
  sing_command="singularity exec -B ${ENDORSE_SRC_ROOT}:${ENDORSE_SRC_ROOT} $sif_image"

  final_command="${sing_command} bash -c \"${command}\""
  echo ${final_command}

  if [ "$debug" == false ]; then
    eval ${final_command}
  fi
else

  final_command="bash -c \"${command}\""
  echo ${final_command}

  if [ "$debug" == false ]; then
    ./endorse_fterm exec ${final_command}
  fi
fi
