#!/bin/bash

set -x

py_script=$1
pbs_script=$1.pbs
script_path=${py_script%/*}
work_dir=$2
mlmc_lib=$3
singularity_path=$4
endorse_repository=$5


cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=16:cgroups=cpuacct:mem=16Gb:scratch_ssd=32gb
#PBS -l walltime=2:00:00
#PBS -q charon_2h
#PBS -N endorse_main
#PBS -j oe

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

cd $SCRATCHDIR

module load gcc
module load python/3.8.0-gcc
module load openblas-0.3.15

python3.8 -m venv env --clear
source env/bin/activate


python3.8 -m pip install ${mlmc_lib}
python3.8 -m pip install scipy

python3.8 -m pip install pyyaml-include
python3.8 -m pip install scikit-learn
python3.8 -m pip install matplotlib


cd ${script_path}
#cp -R $SCRATCHDIR/env .
#source env/bin/activate


python3.8 ${py_script} run ${work_dir} ${singularity_path} ${endorse_repository} --clean --debug
deactivate

EOF
qsub $pbs_script
