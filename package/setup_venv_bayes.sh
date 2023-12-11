#!/bin/bash

echo "Creating python environment for Bayes."

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ENDORSE_SRC_ROOT="$SCRIPTPATH/.."
cd "$ENDORSE_SRC_ROOT"

rm -r venv_bayes
python3 -m venv --system-site-packages venv_bayes

venv_pip=${ENDORSE_SRC_ROOT}/venv_bayes/bin/pip

${ENDORSE_SRC_ROOT}/venv_bayes/bin/python3 --version

$venv_pip install -r requirements.txt

$venv_pip install -e submodules/bgem
$venv_pip install --force-reinstall --upgrade attrs

$venv_pip install -e submodules/surrDAMH
#$venv_pip install -e submodules/redis-cache
$venv_pip install -e . # install endorse
