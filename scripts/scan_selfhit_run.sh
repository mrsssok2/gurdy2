#!/bin/bash
# Usage (sim container): scripts/scan_selfhit_run.sh OUT_PREFIX controller [launch args...]   (lidar on, open world, headless)
set -eo pipefail
OUT=$1; CTL=$2; shift 2
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=true lidar:=true controller:=$CTL "$@" > $OUT.launch.log 2>&1 &
L=$!
ros2 run gurdy_navigation scan_self_filter_node > $OUT.filter.log 2>&1 &
for _ in $(seq 180); do [ "$(ros2 control list_controllers 2>/dev/null | grep -c " active")" = 2 ] && break; sleep 1; done
sleep 5
python3 scripts/scan_selfhit.py $OUT.json | tee $OUT.txt
kill -TERM $L; sleep 4
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|gait_node|scan_self_filter|ros2 launch" || true
