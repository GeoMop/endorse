#!/bin/bash

# !! Needs redis server installed.
# sudo apt install redis


# Setup virtual environment for development

python3 -m venv venv
#python3 -m venv --system-site-packages venv
source venv/bin/activate
pip3 install wheel
pip3 install --upgrade pip
python3 -m pip install -r requirements.txt

# Following included into requirements.txt
#python3 -m pip install -e submodules/bgem
#python3 -m pip install -e submodules/bih
#python3 -m pip install -e submodules/redis-cache
#python3 -m pip install -e submodules/MLMC

python3 -m pip install -e .
