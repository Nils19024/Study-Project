#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$TAPAS_ROOT"

python "$OWN_CODE_ROOT/unimanual/train_pushbutton_gmm.py"
