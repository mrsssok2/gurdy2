#!/usr/bin/env bash
# Send a Nav2 goal in the map frame:  scripts/nav_goal.sh X Y [YAW_RAD]   (inside the container)
set -euo pipefail
x=${1:?x}; y=${2:?y}; yaw=${3:-0.0}
qz=$(python3 -c "import math;print(math.sin($yaw/2))"); qw=$(python3 -c "import math;print(math.cos($yaw/2))")
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: $x, y: $y, z: 0.0}, orientation: {z: $qz, w: $qw}}}}" --feedback
