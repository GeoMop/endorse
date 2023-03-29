#!/bin/bash
set -x

# set running on metacentrum to False
sed -i '/run_on_metacentrum:/c\run_on_metacentrum: False' config.yaml

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


command="source ./venv/bin/activate && python3 run_set_flow123d.py $output_dir $csv_data"

if [ "$sing" == true ]; then

  # command for running correct docker image
  rep_dir=$(pwd)
  # image=$(./endorse_fterm image)
  # sing_command="singularity exec -B $rep_dir:$rep_dir docker://$image"
  image=$(./sif_image)
  sing_command="singularity exec -B $rep_dir:$rep_dir $image"

  # auxiliary command for opening Python environment inside docker image
#  bash_py="bash -c 'source ./venv/bin/activate &&"

  # run setup, prepare PBS script (locally, single proc)
#  command="$sing_command $bash_py python3 -m mpi4py run_all.py $output_dir $n_chains'"
  command="$sing_command bash -c \"$command\""
  echo $command
  eval $command

else
    ./endorse_fterm exec "bash -c \"$command\""
fi
