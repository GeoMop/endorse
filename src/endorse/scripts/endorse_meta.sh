#!/bin/bash

# Main endorse application script.
# Example of the starting script for the Metacentrum PBS system.
# Configure by editatation.
# Purpose is to record right starting of the endorse PBS execution script (endorse_swrap)
# using qsub.


set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"



#QUEUE="charon"
QUEUE="charon_2h"

WORKDIR_BASE="${HOME}/endorse_work"
stdoe="${WORKDIR_BASE}/"

# Restrict to particular cluster (in case of using generic queue)
cluster_restrict=":cluster=${CLUSTER}"

if [ "${QUEUE}" == "q_1d" ]
then
    queue=
    walltime="-l walltime=3:00:00"
else
    queue="-q ${QUEUE}"
fi

select="-l select=1:ncpus=8:mem=8gb${cluster_restrict}"

endorse_cmd= 
qsub ${queue} -S /bin/bash -N endorse-main -j oe -o ${stdoe} ${select} ${walltime} -- ${SCRIPTPATH}/endorse_swrap "$WORKDIR_REL" "$endorse_cmd" 

