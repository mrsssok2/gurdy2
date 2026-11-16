#!/bin/bash
# Usage (sim container): scripts/gz_start_check_run.sh OUT.json LIN ANG [model]   (headless, policy, open world)
set -eo pipefail
OUT=$1; LIN=$2; ANG=$3; MODEL=${4:-}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=true controller:=policy ${MODEL:+model:=$MODEL} > ${OUT%.json}.launch.log 2>&1 &
L=$!
for _ in $(seq 180); do [ "$(ros2 control list_controllers 2>/dev/null | grep -c " active")" = 2 ] && break; sleep 1; done
sleep 5
python3 scripts/gz_start_check.py $OUT $LIN $ANG
kill -TERM $L; sleep 4
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|ros2 launch" || true
