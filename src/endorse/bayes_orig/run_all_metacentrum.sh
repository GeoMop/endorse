#!/bin/bash

# set -x

debug=false
clean=""

obligatory_param=0
while getopts ":hdn:o:c" opt; do
  case $opt in
    h)
      # help
      echo "Usage: ./run_all_metacentrum.sh -n <N_CHAINS> -o <OUTPUT_DIR> -c -d"
      echo "-c ... cleans the <OUTPUT_DIR> at first"
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
    c)
      # output directory
      clean="clean"
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
  echo "clean = $clean"
fi

if [[ $obligatory_param -lt 2 ]]; then
  echo "Not all obligatory parameters set!"
  exit 1
fi

# set running on metacentrum to True
sed -i '/run_on_metacentrum:/c\run_on_metacentrum: True' config.yaml

rep_dir=$(pwd)

# command for running correct docker image
#image_name=$(./endorse_fterm image)
#echo "Docker image name: '$image_name'"
#
## possibly create SIF image file
#image_sif_file=$( echo "$image_name.sif" | tr /: _ )
#if [ -f "$image_sif_file" ]; then
#  echo "Using SIF image '$image_sif_file'"
#  image=$image_sif_file
#else
#  echo "SIF does not exist. Building SIF image '$image_sif_file'"
#  singularity build "$image_sif_file" "docker://$image_name"
#  image=$image_sif_file
#fi

image=$(./sif_image)

sing_command="singularity exec -B $rep_dir:$rep_dir $image"

# auxiliary command for opening Python environment inside docker image
bash_py="bash -c 'source ./venv/bin/activate &&"


# run setup, prepare PBS script (locally, single proc)
command="$sing_command $bash_py python3 -m mpi4py run_all.py $output_dir $n_chains $clean'"
echo "$command"

if [ "$debug" == false ]; then
  eval "$command"
  qsub "$output_dir/pbs_job.sh"
fi