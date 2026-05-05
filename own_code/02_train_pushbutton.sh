#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$TAPAS_ROOT"

python ../own_code/train_pushbutton_gmm.py
