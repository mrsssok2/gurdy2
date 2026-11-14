#!/bin/bash
# Source ROS 2 and (if built) the workspace overlay, then run the given command.
set -e
source /opt/ros/jazzy/setup.bash
if [ -f "$HOME/gurdy2/ros2_ws/install/setup.bash" ]; then
  source "$HOME/gurdy2/ros2_ws/install/setup.bash"
fi
exec "$@"
