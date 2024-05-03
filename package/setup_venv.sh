#!/bin/bash

set -x
# !! Needs redis server installed.
# sudo apt install redis

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/.."
cd "$ENDORSE_SRC_ROOT"

# Setup virtual environment for development

python3 -m venv venv
#python3 -m venv --system-site-packages venv
source venv/bin/activate
pip3 install wheel
pip3 install --upgrade pip
python3 -m pip install -r requirements.txt


# Following included into requirements.txt
python3 -m pip install -e ../bgem
python3 -m pip install -e ../bih
python3 -m pip install -e submodules/MLMC

python3 -m pip install -e .


# attrs are broken after all installs, force reinstallation
python3 -m pip uninstall --yes attrs
python3 -m pip install attrs
python3 -m pip install pytest

#python3 -m pip uninstall --yes pyyaml-include
#python3 -m pip install pyyaml-include

# check it
python3 -c "import attrs, yaml_include, mlmc, endorse, bgem, bih, pytest"

