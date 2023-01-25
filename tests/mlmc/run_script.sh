#!/bin/bash

export SINGULARITY_TMPDIR=$SCRATCHDIR

ENDORSE_DOCKER="docker://flow123d/endorse:latest"
ENDORSE_REPOSITORY="/storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport"
RUN_SCRIPT_DIR="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc"

cd $ENDORSE_REPOSITORY
singularity exec $ENDORSE_DOCKER ./setup.sh

cd $RUN_SCRIPT_DIR
singularity exec $ENDORSE_DOCKER python3 fullscale_transport.py run ../ --clean
