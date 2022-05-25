#!/bin/sh
# assumption:
# * virtualenv is avaliable and working

export SIF_PATH=$(pwd)
export PIP_REQUIRE_VIRTUALENV=true
export PIP_RESPECT_VIRTUALENV=true
virtualenv -p /usr/bin/python3 --clear --no-site-packages $SIF_PATH/.venv/sif_env
source $SIF_PATH/.venv/sif_env/bin/activate
pip install -r $SIF_PATH/requirements.txt --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu113
