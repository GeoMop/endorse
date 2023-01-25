#!/bin/bash

set -x

py_script=`pwd`/$1
pbs_script=`pwd`/$1.pbs
script_path=${py_script%/*}
work_dir=$2

#ENDORSE_DOCKER="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc/endorse.sif"
#ENDORSE_REPOSITORY="/storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport"
#RUN_SCRIPT_DIR="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc"
mlmc=/storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport/submodules/MLMC

cat >$pbs_script <<EOF
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=16:cgroups=cpuacct:mem=16Gb:scratch_ssd=32gb
#PBS -l walltime=2:00:00
#PBS -q charon_2h
#PBS -N endorse_main
#PBS -j oe

#export TMPDIR=$SCRATCHDIR
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

cd $SCRATCHDIR

module load gcc
module load python/3.8.0-gcc
module load openblas-0.3.15

python3.8 -m venv env --clear
source env/bin/activate

#python3.8 -m pip install wheel

#python3.8 -m pip install numpy==1.21.0

python3.8 -m pip install ${mlmc}
python3.8 -m pip install scipy

python3.8 -m pip install pyyaml-include
python3.8 -m pip install scikit-learn
python3.8 -m pip install matplotlib
#deactivate


cd ${script_path}
#cp -R $SCRATCHDIR/env .
#source env/bin/activate

#python3.8 -m pip install attrs numpy h5py gstools ruamel.yaml sklearn memoization matplotlib

#cd ${script_path}

#cd ${script_path}



python3.8 ${py_script} run ${work_dir} --clean --debug
deactivate

EOF
qsub $pbs_script

#export SINGULARITY_TMPDIR=$SCRATCHDIR
#
#
##cd $ENDORSE_REPOSITORY
##singularity exec $ENDORSE_DOCKER ./setup.sh
##
##cd $RUN_SCRIPT_DIR
#singularity exec -B /usr/bin/qsub,/usr/bin/qstat $ENDORSE_DOCKER python3 fullscale_transport_pbs.py run ../ --clean

