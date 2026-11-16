#!/bin/bash
# Phase 3 navigation run (headless): localization on a saved map + Nav2, goals via scripts/nav_eval.py.
# Usage (sim container, nothing else running): scripts/phase3_nav_run.sh NAME MAP controller odom_source [model]
set -eo pipefail
NAME=${1:?}; MAP=${2:?}; CTL=${3:-policy}; ODOM=${4:-ground_truth}; MODEL=${5:-}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
OUT=docs/results/data/phase3/$NAME; mkdir -p "$OUT"
ros2 launch gurdy_bringup sim.launch.py gui:=false nav:=localization map:=$MAP controller:=$CTL odom_source:=$ODOM ${MODEL:+model:=$MODEL} \
  > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
for _ in $(seq 180); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c " active")" = 2 ] && break; sleep 1
done
for _ in $(seq 120); do ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q "^active" && break; sleep 1; done
for _ in $(seq 60); do ros2 lifecycle get /controller_server 2>/dev/null | grep -q "^active" && ros2 lifecycle get /planner_server 2>/dev/null | grep -q "^active" && break; sleep 1; done
sleep 5
timeout 15 ros2 topic echo /amcl_pose --once > "$OUT/amcl_initial.txt" 2>&1 || true
(python3 scripts/measure_rtf.py > "$OUT/rtf.txt" 2>&1 &)
python3 scripts/nav_eval.py "$OUT/goals.json" "1.5,0.5,0.0" "4.0,0.5,0.0" "4.0,-2.2,1.57" "5.5,1.0,0.0" "1.5,-0.5,3.14" "0.0,0.0,0.0" | tee "$OUT/goals.txt"
ps -eo rss,cmd --sort=-rss | head -8 > "$OUT/ps_top.txt"
timeout 10 ros2 topic echo /amcl_pose --once > "$OUT/amcl_final.txt" 2>&1 || true
grep -iE "warn|error" "$OUT/launch.log" | grep -v "gz_ros_control\|controller_manager\|Utils.cc" | cut -c1-220 | sort | uniq -c | sort -rn | head -20 > "$OUT/warnings.txt" || true
kill -TERM $LAUNCH 2>/dev/null || true
sleep 4
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|gait_node|nav2|amcl|ros2 launch|measure_rtf|component_container" 2>/dev/null || true
