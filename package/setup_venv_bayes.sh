#!/bin/bash

echo "Creating python environment for Bayes."

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/.."
cd "$ENDORSE_SRC_ROOT"

python3 -m venv --system-site-packages venv_bayes

source venv_bayes/bin/activate
python3 --version
which python
which pip

pip3 install wheel
pip3 install --upgrade pip
python3 -m pip install -r requirements.txt
pip -V

# already installed in image: numpy
pip install pandas statsmodels
# ruamel keeps comments and order when loading and dumping yaml around
pip install ruamel.yaml

# if not included in docker image
# pip install mpi4py

pip install -e submodules/bgem
pip install -e submodules/surrDAMH
pip install -e submodules/redis-cache
pip install -e . # install endorse

#pip freeze
deactivate

