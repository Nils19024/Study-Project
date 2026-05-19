#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
cd "$TAPAS_ROOT"
tapas-collect-rlbench --config conf/collect_data/rlbench_expert.py -t PushButton -o n_episodes=5 data_naming.data_root="$OWN_CODE_OUTPUTS/unimanual/tapas_observations"
