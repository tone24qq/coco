#!/bin/bash
# Quick sanity training run
python -m src.training.train --config configs/small.yaml "$@"
