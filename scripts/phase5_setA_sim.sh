#!/bin/bash
# Milestone 5.1 set A: sim container side = Gazebo (semantic_lab, camera+depth+lidar) + SLAM + Nav2 + policy + frontier exploration, NO perception
# node (the GPU detector runs in brain/bench/load_gen.py). Usage (sim container): scripts/phase5_setA_sim.sh OUT_DIR SECONDS [gui:=true|false]
set -eo pipefail
OUT=${1:?}; SECS=${2:-300}; GUI=${3:-false}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
rm -rf "$OUT"; mkdir -p "$OUT"; source scripts/lib_nav_start.sh
start_nav_stack "$OUT" gui:=$GUI world:=semantic_lab nav:=slam navigation:=true controller:=policy odom_source:=ground_truth \
  || { echo "Nav2 never became ready" | tee "$OUT/FAILED.txt"; stop_nav_stack; exit 1; }
echo ready > "$OUT/nav_ready_flag"
(python3 scripts/measure_rtf.py > "$OUT/rtf.txt" 2>&1 &)
ros2 run gurdy_navigation frontier_explorer_node --ros-args -p log_path:="$OUT/explore_log.json" -p done_path:="$OUT/explore.done" > "$OUT/explorer.log" 2>&1 &
EXP=$!
sleep "$SECS"
ps -eo rss,cmd --sort=-rss | head -10 > "$OUT/ps_top.txt"
kill -TERM $EXP 2>/dev/null || true
stop_nav_stack
