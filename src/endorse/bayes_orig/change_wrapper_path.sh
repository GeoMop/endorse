#!/bin/bash
#set -x

path=$1

charon_path="/auto/liberec3-tul/home/pavel_exner/workspace/endorse/src/endorse/bayes_orig/flow_wrapper.py"
my_path="/home/paulie/Workspace/endorse/src/endorse/bayes_orig/flow_wrapper.py"
find $path -iname config_mcmc_bayes.yaml
sed -i -e "s|$charon_path|$my_path|g" $(find $path -iname config_mcmc_bayes.yaml)
