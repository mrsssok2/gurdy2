#!/bin/bash
# Default command of the `sim` compose service: build the workspace if needed, then start the
# walking robot in Gazebo (GUI). Extra args go to the launch file, e.g. sensors:=false rviz:=true
set -e
cd ~/gurdy2/ros2_ws
if [ ! -f install/setup.bash ]; then
  echo "[start_sim] first run: colcon build"; colcon build --symlink-install
fi
set +u; source install/setup.bash
exec ros2 launch gurdy_bringup sim.launch.py "$@"
