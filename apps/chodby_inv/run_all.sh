#!/bin/bash
#PBS -N chodby_inv_run_all
#PBS -l select=1:ncpus=20:mem=50gb
#PBS -l walltime=24:00:00
#PBS -q charon
#PBS -W umask=0002

cd $PBS_O_WORKDIR


# Exit immediately on errors
set -e

# Upper limit
n=${N:-50}

# 1) Load Python 3.11
module load python/3.11

# 2) Activate virtual environment
source venv/bin/activate

export MPLBACKEND=Agg

# 3) Loop from 0 to n
for i in $(seq 0 $n); do
    echo "Running script with parameter $i"
    python -m chodby_inv.piezo.wpt_bayes "$i"
done