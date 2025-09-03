#!/bin/bash
# interactive_pbs.sh
# Request an interactive PBS job. Adjust resources to taste.

# ---- resources (edit) ----
NODES=2
PPN=8                    # cores per node
MEM_PER_NODE=20gb
SCRATCH_PER_NODE=20gb
QUEUE=charon_2h
WALLTIME=02:00:00

# ---- request interactive shell ----
qsub -I -q "$QUEUE" -l "select=${NODES}:ncpus=${PPN}:mem=${MEM_PER_NODE}:scratch_local=${SCRATCH_PER_NODE}" \
    -l place=scatter -l walltime="$WALLTIME" -N dask-debug
