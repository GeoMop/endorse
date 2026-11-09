#!/bin/bash
# interactive_pbs.sh
# Request an interactive PBS job. Adjust resources to taste.

# ---- resources (edit) ----
# NODES=1
# PPN=2                    # cores per node
# MEM_PER_NODE=60gb
# SCRATCH_PER_NODE=50gb
# QUEUE=charon
# WALLTIME=08:00:00

NODES=1
PPN=2                    # cores per node
MEM_PER_NODE=60gb
SCRATCH_PER_NODE=50gb
QUEUE=charon
WALLTIME=02:00:00

# ---- request interactive shell ----
# qsub -I -q "$QUEUE" -l "select=${NODES}:ncpus=${PPN}:mem=${MEM_PER_NODE}:scratch_local=${SCRATCH_PER_NODE}" \
#     -l place=scatter -l walltime="$WALLTIME" -N chodby_trans_int

# no scratch
qsub -I -q "$QUEUE" -l "select=${NODES}:ncpus=${PPN}:mem=${MEM_PER_NODE}" \
    -l place=scatter -l walltime="$WALLTIME" -N chodby_trans_int
