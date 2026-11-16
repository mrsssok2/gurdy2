#!/bin/bash
# Phase 3 navigation run (headless): localization on a saved map + Nav2, goals via scripts/nav_eval.py.
# Usage (sim container, nothing else running): scripts/phase3_nav_run.sh NAME MAP controller odom_source [model]
set -eo pipefail
NAME=${1:?}; MAP=${2:?}; CTL=${3:-policy}; ODOM=${4:-ground_truth}; MODEL=${5:-}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
OUT=docs/results/data/phase3/$NAME; mkdir -p "$OUT"
source scripts/lib_nav_start.sh
start_nav_stack "$OUT" gui:=false nav:=localization map:=$MAP controller:=$CTL odom_source:=$ODOM ${MODEL:+model:=$MODEL} \
  || { echo "Nav2 never became ready" | tee "$OUT/FAILED.txt"; stop_nav_stack; exit 1; }
sleep 5
timeout 15 ros2 topic echo /amcl_pose --once > "$OUT/amcl_initial.txt" 2>&1 || true
(python3 scripts/measure_rtf.py > "$OUT/rtf.txt" 2>&1 &)
(python3 scripts/nav_probe.py 3000 > "$OUT/probe.txt" 2>&1 &)      # velocity chain + GT motion per 10 s window
python3 scripts/nav_eval.py "$OUT/goals.json" "1.5,0.5,0.0" "4.0,0.5,0.0" "4.0,-2.2,1.57" "1.5,-0.5,3.14" "0.0,0.0,0.0" | tee "$OUT/goals.txt"
ps -eo rss,cmd --sort=-rss | head -8 > "$OUT/ps_top.txt"
timeout 10 ros2 topic echo /amcl_pose --once > "$OUT/amcl_final.txt" 2>&1 || true
grep -iE "warn|error" "$OUT/launch.log" | grep -v "gz_ros_control\|controller_manager\|Utils.cc" | cut -c1-220 | sort | uniq -c | sort -rn | head -20 > "$OUT/warnings.txt" || true
stop_nav_stack
pkill -f nav_probe 2>/dev/null || true
