#!/bin/bash
# 
# Run Flow123d in a production docker container.
#     mini_fterm.sh <image_name:tag> [...] -- <flow_arguments>

# get CWD as realtive path to current directory, which enables to build image on Windows
OLD_PWD="$(pwd)"
ABS_FLOW_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
REL_FLOW_DIR=$(realpath --relative-to=$(pwd) $ABS_FLOW_DIR)


function print_usage() {
  cat << EOF
  
Usage:  $bold fterm [tag] [options] [action] [arguments]$reset
  
Start docker container with realease Flow123d version available.
   
tag     image tag in form <branch>_<commit_short_hash>
        Images automatically downloaded from docker hub, see:  ${bblue} https://hub.docker.com/r/flow123d/flow-dev-gnu-dbg/tags ${reset}
        for avalilable tags. Default tag is ${bblue} ${image_tag} ${reset}
    
${bold}Options:${reset}
  ${bold}-h|--help ${reset}       fterm documentation
  ${bold}-x|--set-x ${reset}      turn on script debuging mode (set -x)
  ${bold}-T|--no-term ${reset}    Turn off terminal for 'run' and 'exec' actions. Necessary when called from another process.
    
${bold}Evironment variables:${reset}
  ${bold}nocolor${reset}          turn off terminal colors.
  ${bold}FLOW123D_WORK${reset}    explicit working directory to mount
  
${bold}Examples:${reset}
  ${bblue}bin/mini_fterm.sh -x -i ci-gnu: -- input.yaml
  Start make of the Flow123d using the debug environment at version 2.2.0.
    
  ${bblue}bin/fterm run --help$reset
  Run flow123d --help command.
EOF
#  ${bblue}bin/fterm dbg --detach dbg_run$reset
#      docker run <opts> ... flow123d/flow-dev-gnu-dbg:2.2.0

}


function get_mount_dir() {
  if [[ -n $FLOW123D_WORK ]]; then
    echo $FLOW123D_WORK;
  else
    # try to find the first dir in $HOME, in which
    # flow123d is located
    local workdir=$1
    local workdir_tmp=
    for (( i = 0; i < 10; i++ )); do
      workdir_tmp=$(dirname $workdir)
      if [[ "$workdir_tmp" == "$HOME" ]]; then
        # we found what we were looking for
        echo $workdir
        return
      fi
      workdir=$workdir_tmp
    done
    # the folder is too deep or not in HOME
    # we return entire $HOME
    echo $(dirname $1)
  fi
}


# Will check whether given image exists and return 0 if it does
function image_exist() {
  did=$(docker images $1 -q)
  if [[ -z $did ]]; then
    return 1
  else
    return 0
  fi
}


# Will pull all the images to newest
function update_images() {
  for image in $flow123d_images
  do
    docker pull flow123d/$image:$image_tag
  done
}


# grab user's id
gid=$(id -g)
uid=$(id -u)

# not using $(whoami) so there are no collisions with $HOME
uname=flow
autopull=${autopull:-1}
theme=${theme:-light}


# default settings
verbose=1
work=$(get_mount_dir $ABS_FLOW_DIR)
docker_terminal="-it"
image="flow123d-gnu:3.9.1"

while [[ $# -gt 0 ]]
  do
  key="$1"
  case $key in
    -x|--set-x)
      set -x
      shift
    ;;
    --)
      shift
      action=raw
      rest="$@"
      break
    ;;
    -T|--no-term)
      docker_terminal=""
      shift
    ;;
    -h|--help)
      print_usage
      exit 1
    ;;
    -i|--image)
      shift
      image=$1
      shift
    ;;
    *)
      echo -e "${bred}ERROR:$reset ${red} Invalid argument '$1'!$reset"
      print_usage
      echo -e "${bred}ERROR:$reset ${red} Invalid argument '$1'!$reset"
      exit 1
    ;;
  esac
done

full_image="flow123d/${image}"
docker pull $full_image

# env variables which will be passed as well
envarg="-euid=$uid -egid=$gid -etheme=$theme -ewho=$uname -ehome=/mnt/$HOME -v /$HOME:/mnt/$HOME"
mountargs="-w /${OLD_PWD} -v /${OLD_PWD}:/${OLD_PWD} -v /${work}:/${work}"
if [[ $privileged == "1" ]]; then
  priv_true="--privileged=true"
fi


if [[ $autopull == "1" ]]; then
    update_images
fi
docker run --rm ${docker_terminal} $envarg $mountargs $priv_true $base_image "flow123d" "$rest"

exit $?
