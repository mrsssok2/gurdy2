set -eo pipefail
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
M=$1; OUT=$2
ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=true controller:=policy model:=$M > /dev/null 2>&1 &
L=$!
for _ in $(seq 120); do [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" = 2 ] && break; sleep 1; done
sleep 2
python3 scripts/policy_diag_gz.py $OUT.gz.npz
kill -TERM $L; sleep 3; pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|ros2 launch" || true
python3 scripts/policy_track_mujoco.py --model $M --dump $OUT.mj.npz > /dev/null
