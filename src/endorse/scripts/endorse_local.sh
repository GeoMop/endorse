#!/bin/bash

# Main endorse application script.
# Example of the starting script for local execution.
# endorse_swrap PBS script is run directly.

set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
${SCRIPTPATH}/endorse_swrap "$WORKDIR_REL" "$@"
