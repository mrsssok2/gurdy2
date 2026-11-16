#!/bin/bash
# Phase 3 mapping run (headless, SLAM-only): scripted route in flat_lab, save the map, RTF/RAM, score.
# Usage (inside the sim container, nothing else running): scripts/phase3_map_run.sh NAME [controller=tripod] [odom=ground_truth] [model]
set -eo pipefail
NAME=${1:?map name}; CTL=${2:-tripod}; ODOM=${3:-ground_truth}; MODEL=${4:-}
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
OUT=docs/results/data/phase3/$NAME; mkdir -p "$OUT"
ros2 launch gurdy_bringup sim.launch.py gui:=false nav:=slam navigation:=false controller:=$CTL odom_source:=$ODOM ${MODEL:+model:=$MODEL} \
  > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
for _ in $(seq 180); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" = 2 ] && break; sleep 1
done
for _ in $(seq 60); do ros2 topic list 2>/dev/null | grep -q "^/map$" && break; sleep 1; done
sleep 3
timeout 20 ros2 topic hz /scan 2>&1 | grep "average rate" | tail -1 > "$OUT/scan_hz.txt" || true
(python3 scripts/measure_rtf.py > "$OUT/rtf.txt" 2>&1 &)
if [ -n "$BAG" ]; then      # BAG=1: TF + scans + all odometry for offline analysis (scripts/bag_export.py, bag_overlay.py)
  rm -rf "$OUT/bag"
  ros2 bag record -o "$OUT/bag" /tf /tf_static /scan /clock /imu /odom/ground_truth /odom/legs /imu/ekf /odometry/filtered \
    /cmd_vel > "$OUT/bag_record.log" 2>&1 &
  BAGPID=$!; sleep 3
fi
python3 scripts/drive_route.py "$OUT/route.json" | tee "$OUT/route_summary.txt"
if [ -n "$BAGPID" ]; then kill -INT $BAGPID; sleep 5; fi
scripts/map_tools.sh save "$NAME" > "$OUT/map_save.log" 2>&1
scripts/map_tools.sh serialize "$NAME" > "$OUT/serialize.log" 2>&1 || true
ps -eo rss,cmd --sort=-rss | head -8 > "$OUT/ps_top.txt"
grep -iE "warn|error" "$OUT/launch.log" | sort | uniq -c | sort -rn | head -15 > "$OUT/warnings.txt" || true
kill -TERM $LAUNCH 2>/dev/null || true
sleep 4
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|gait_node|slam_toolbox|ros2 launch|measure_rtf" 2>/dev/null || true
ls -la maps/$NAME.* 
