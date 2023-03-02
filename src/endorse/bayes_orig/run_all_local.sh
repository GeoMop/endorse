#!/bin/bash

# set -x

debug=false
run=false
visualize=false
clean=""
# sing == true => use singularity
# sing == false => use docker
sing=false

obligatory_param=0
while getopts ":hdn:o:t:cs" opt; do
  case $opt in
    h)
      # help
      echo "Usage: ./run_all_local.sh -n <N_CHAINS> -o <OUTPUT_DIR> -t <TASK> -c -s -d"
      echo "-t ... 'run' or 'visualize'"
      echo "-c ... cleans the <OUTPUT_DIR> at first"
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
    t)
      # task
      comm=$OPTARG
      if [ "$comm" == "visualize" ]; then
        visualize=true
      elif [ "$comm" == "run" ]; then
        run=true
      else
        echo "Unknown command '$comm'!" >&2
        exit 1
      fi
      ((obligatory_param=obligatory_param+1))
      ;;
    c)
      # output directory
      clean="clean"
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
  echo "run = $run"
  echo "clean = $clean"
  echo "sing = $sing"
fi

if [[ $obligatory_param -lt 3 ]]; then
  echo "Not all obligatory parameters set!"
  exit 1
fi

# set running on metacentrum to False
sed -i '/run_on_metacentrum:/c\run_on_metacentrum: False' config.yaml

#command='./endorse_fterm exec bash -c \"source ./venv/bin/activate && python3 -m mpi4py run_all.py $mcmc_config $output_dir 4\"'
#echo $command
#eval $command
#./endorse_fterm exec bash -c "\"source ./venv/bin/activate ; python3 -m mpi4py run_all.py $mcmc_config $output_dir 4\""
#./endorse_fterm exec bash -c "\"which python\""
#command="source ./venv/bin/activate && python --version"
#command="which python"

#docker run --rm -it -euid=1000 -egid=1000 -etheme=light -ewho=flow -ehome=/mnt//home/paulie -v //home/paulie:/mnt//home/paulie -w //home/paulie/Workspace/Endorse-2Dtest-Bayes -v //home/paulie/Workspace/Endorse-2Dtest-Bayes://home/paulie/Workspace/Endorse-2Dtest-Bayes -v //home/paulie/Workspace://home/paulie/Workspace flow123d/geomop:master_8c1b58980 bash -c "$command"

# run sampling
if [ "$run" == true ]; then
  command="source ./venv/bin/activate && export OMP_NUM_THREADS=2 && python3 -m mpi4py run_all.py $output_dir $n_chains $clean"
#  command="source ./venv/bin/activate && python3 -m mpi4py run_all.py $output_dir $n_chains"
fi

# visualize
if [ "$visualize" == true ]; then
  command="source ./venv/bin/activate && python3 run_all.py $output_dir $n_chains '' visualize"
fi


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

  if [ "$debug" == false ]; then
    eval $command
  fi
else
  echo "bash -c \"$command\""
  if [ "$debug" == false ]; then
    ./endorse_fterm exec "bash -c \"$command\""
  fi
fi
