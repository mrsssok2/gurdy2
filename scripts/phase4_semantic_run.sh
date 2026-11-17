#!/bin/bash
# Phase 4: exploration in semantic_lab with the detector on; saves the semantic map + annotated frames
# Usage (sim container, nothing else running): scripts/phase4_semantic_run.sh NAME [detector.onnx]
# Output: docs/results/data/phase4/NAME/{semantic.json,frames/,{explore_log.json,timeseries.csv,monitor.json,snaps/,rtf.txt,...}, maps/NAME.*
set -eo pipefail
NAME=${1:?}; DET=${2:-det_v1_s.onnx}; CTL=policy; ODOM=ground_truth; MODEL=
MAXWALL=${MAXWALL:-3600}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
OUT=docs/results/data/phase4/$NAME; rm -rf "$OUT"; mkdir -p "$OUT"
DONE="$OUT/explore.done"
source scripts/lib_nav_start.sh
start_nav_stack "$OUT" gui:=false world:=semantic_lab perception:=true detector_model:=$DET target_frame:=map nav:=slam navigation:=true controller:=$CTL odom_source:=$ODOM ${MODEL:+model:=$MODEL} \
  || { echo "Nav2 never became ready" | tee "$OUT/FAILED.txt"; stop_nav_stack; exit 1; }
python3 vision/save_semantic_map.py --frames "$OUT/frames" models/detector/$DET > "$OUT/frames.log" 2>&1 &
FR=$!
(python3 scripts/measure_rtf.py > "$OUT/rtf.txt" 2>&1 &)
python3 scripts/explore_monitor.py "$OUT" "$DONE" > "$OUT/monitor.log" 2>&1 &
MON=$!
ros2 run gurdy_navigation frontier_explorer_node --ros-args -p log_path:="$OUT/explore_log.json" -p done_path:="$DONE" \
  > "$OUT/explorer.log" 2>&1 &
EXP=$!
for _ in $(seq $((MAXWALL / 5))); do [ -f "$DONE" ] && break; sleep 5; done
[ -f "$DONE" ] || echo "WALL TIMEOUT ${MAXWALL}s" | tee "$OUT/wall_timeout.txt"
sleep 8
python3 vision/save_semantic_map.py "$OUT/semantic.json" > "$OUT/semantic_save.log" 2>&1 || true
scripts/map_tools.sh save "$NAME" > "$OUT/map_save.log" 2>&1 || true
scripts/map_tools.sh serialize "$NAME" > "$OUT/serialize.log" 2>&1 || true
ps -eo rss,cmd --sort=-rss | head -8 > "$OUT/ps_top.txt"
grep -iE "warn|error" "$OUT/launch.log" | grep -v "gz_ros_control\|controller_manager\|Utils.cc" | cut -c1-220 | sort | uniq -c | sort -rn | head -20 > "$OUT/warnings.txt" || true
kill -TERM $FR $EXP $MON 2>/dev/null || true
stop_nav_stack
ls -la maps/$NAME.* 2>/dev/null || true
