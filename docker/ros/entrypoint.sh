#!/bin/bash
# Source ROS 2 and (if built) the workspace overlay, then run the given command.
set -e
# Isolate gz-transport per container: without this, two Gazebo instances on the same Docker
# network share /clock and /stats (mixed sim time). Override GZ_PARTITION to join on purpose.
export GZ_PARTITION="${GZ_PARTITION:-$(hostname)}"
source /opt/ros/jazzy/setup.bash
if [ -f "$HOME/gurdy2/ros2_ws/install/setup.bash" ]; then
  source "$HOME/gurdy2/ros2_ws/install/setup.bash"
fi
exec "$@"
