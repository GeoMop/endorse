#!/bin/bash
#PBS -N chodby_inv_run_all        # Job name
#PBS -l select=1:ncpus=20:mem=10gb   # Resources (adjust to your needs)
#PBS -l walltime=06:00:00    # Max runtime

# Exit immediately on errors
set -e

# Check if n was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

# Upper limit
n=$1

# 1) Load Python 3
module load python/3

# 2) Activate virtual environment
source /venv/scripts/activate

# 3) Loop from 0 to n
for i in $(seq 0 $n); do
    echo "Running script with parameter $i"
    python -m chodby_inv.piezo.wpt_bayes "$i"
done