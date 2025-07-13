#!/bin/bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt