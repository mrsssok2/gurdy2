#!/bin/bash
# Run the shared protocol in Gazebo (headless) and MuJoCo, one after the other (RAM rule).
# Usage (inside the sim container):  scripts/parity_experiments.sh step|stand OUT_DIR [robot.yaml]
# Writes OUT_DIR/{gazebo,mujoco}_<mode>.json and prints the comparison table.
set -euo pipefail
MODE=${1:?step|stand}; OUT=${2:?out dir}; YAML=${3:-}
mkdir -p "$OUT"
cd ~/gurdy2
set +u; source ros2_ws/install/setup.bash; set -u
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
[ -n "$YAML" ] && export GURDY_ROBOT_YAML=$(realpath "$YAML")
FIXED=$([ "$MODE" = step ] && echo true || echo false)

ros2 launch gurdy_gazebo sim.launch.py gui:=false js_rate:=500 fixed_base:=$FIXED ${YAML:+robot_yaml:=$GURDY_ROBOT_YAML} \
  > "$OUT/gazebo_launch_$MODE.log" 2>&1 &
LAUNCH=$!
for _ in $(seq 120); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" = 2 ] && break; sleep 1
done
python3 scripts/gz_experiment.py "$MODE" > "$OUT/gazebo_$MODE.json"
# Background jobs in scripts ignore SIGINT, so stop launch with SIGTERM, then sweep leftovers.
kill -TERM $LAUNCH 2>/dev/null || true
for _ in $(seq 20); do kill -0 $LAUNCH 2>/dev/null || break; sleep 0.5; done
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|ros2 launch" 2>/dev/null || true
sleep 1

python3 -m sim_mujoco.experiments "$MODE" > "$OUT/mujoco_$MODE.json"
python3 scripts/compare_parity.py "$OUT" "$MODE"
