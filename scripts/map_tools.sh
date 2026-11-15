#!/usr/bin/env bash
# Map helpers (run inside the container while the nav stack is up).
#   scripts/map_tools.sh save NAME        # maps/NAME.yaml + .pgm  (for localization mode, nav2 map_server)
#   scripts/map_tools.sh serialize NAME   # maps/NAME.posegraph + .data (slam_toolbox, resume mapping later)
set -euo pipefail
cmd=${1:?usage: map_tools.sh save|serialize NAME}
name=${2:?map name}
dir="$(cd "$(dirname "$0")/.." && pwd)/maps"
mkdir -p "$dir"
case "$cmd" in
  save)
    ros2 run nav2_map_server map_saver_cli -f "$dir/$name" --ros-args -p save_map_timeout:=10.0 -p free_thresh_default:=0.25 -p occupied_thresh_default:=0.65 ;;
  serialize)
    ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$dir/$name'}" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
