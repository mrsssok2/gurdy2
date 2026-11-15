# Phase 3 — mapping + navigation: test plan (to run, nothing below has been run in Gazebo yet)

Prepared 2026-11-15. Built and checked **without** any simulation: unit tests (`gurdy_navigation/test`:
footprint, Nav2 params vs robot.yaml, deadband/limits, world layout, launch files), xacro expansion with
and without the lidar (`check_urdf` ok), launch `OpaqueFunction`s evaluated for slam / localization /
depth. Design and open risks: `docs/phase3_navigation.md`.

## 0. Preconditions (CLAUDE.md rules 2 and 8: one heavy workload at a time)
```bash
docker ps                                   # must be empty (wait for gurdy_train to finish)
docker compose build sim                    # image already has Nav2 / slam_toolbox / depthimage_to_laserscan
docker compose run --rm --name gurdy_nav sim bash        # shell #1 (the sim container)
# inside:
cd ros2_ws && colcon build --symlink-install --parallel-workers 2 && source install/setup.bash && cd ..
python3 -m pytest -q ros2_ws/src/gurdy_navigation/test   # expect 8 passed
```

## 1. Mapping run (teleop, tripod gait)
```bash
# shell #1: Gazebo GUI + lidar + flat_lab + odom TF + slam_toolbox + Nav2 + RViz (spawn at origin, room A)
ros2 launch gurdy_bringup sim.launch.py nav:=slam rviz:=true
# shell #2:  docker exec -it gurdy_nav /entrypoint.sh bash
ros2 topic hz /scan                          # expect ~10 Hz (lazy bridge starts when slam subscribes)
ros2 run tf2_ros tf2_echo map base_link      # exists once slam_toolbox has processed the first scans
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
```
Drive: room A wall-following, through the doorway (x = 2.5, y 0..1), around room B, back through the
door to the start (loop closure). Watch in RViz: walls stay straight, no doubled walls after the loop.
SLAM-only (no Nav2, less CPU): add `navigation:=false`. Depth-camera instead of lidar:
`scan_source:=depth` (60° FOV, 5 m: expect a worse map — useful as the comparison).

## 2. Save the map
```bash
# shell #2, while mapping is still running:
scripts/map_tools.sh save flat_lab           # maps/flat_lab.yaml + flat_lab.pgm
scripts/map_tools.sh serialize flat_lab      # maps/flat_lab.posegraph + .data (resume mapping later)
```
Record: map quality (walls straight? door width ≈ 1.0 m? compare the pgm with the world), run duration,
`docker stats --no-stream` RAM, RTF (`python3 scripts/measure_rtf.py`), lidar scan rate.
Stop shell #1 (Ctrl-C), then `docker rm -f gurdy_nav` before the next run.

## 3. Navigation with the saved map (AMCL) — send goals
```bash
# shell #1 (fresh container, nothing else running):
ros2 launch gurdy_bringup sim.launch.py nav:=localization map:=flat_lab rviz:=true
# AMCL starts at the spawn pose (0,0,0) from nav2_params.yaml (set_initial_pose). Check localisation:
ros2 topic echo /amcl_pose --once
# shell #2 — goals in the map frame (x y yaw); A -> B room through the door, then back:
scripts/nav_goal.sh 1.5 0.5 0.0              # inside room A, near the red box
scripts/nav_goal.sh 4.0 0.5 0.0              # room B (through the doorway)
scripts/nav_goal.sh 4.0 -2.2 1.57            # room B, south, past the yellow box
scripts/nav_goal.sh 0.0 0.0 0.0              # back to the start
# or in RViz: "2D Goal Pose"
```
Per goal record: reached (result status SUCCEEDED, final position error < 0.25 m), collision (any contact:
`gz topic -e -t /stats` is not enough — watch the robot in the GUI / check `min` distance in the costmap),
time, recoveries. Acceptance (ROADMAP Phase 3): ≥ 8/10 goals without collision, reported for the tripod gait
and for the policy.

## 4. Policy instead of the tripod gait
Same as 1–3 with `controller:=policy` appended to the launch (needs `sensors:=true`, which the nav profile
sets). Run only after step 2.9 (Gazebo deployment test) passed.

## 5. Mapping on its own from Nav2 goals (no teleop)
slam_toolbox + Nav2 work together in `nav:=slam`: send the goals of step 3 to unexplored places inside the
known map; frontier exploration (`explore`) is still to be written (ROADMAP Phase 3, last item).

## What to look at first if something is off
| Symptom | Check |
|---|---|
| no `/scan` | `ros2 topic list \| grep scan`; Gazebo log for gpu_lidar/ogre2 errors; `sensors` plugin loaded (`lidar:=true`) |
| scan shows the ground ring / flickering walls | body pitch/roll; lower `max_laser_range`, see docs/phase3_navigation.md "lidar tilt" |
| `map` frame missing | slam_toolbox lifecycle (`ros2 lifecycle get /slam_toolbox`), `/odom` + TF `odom->base_link` (odom_tf_node) |
| Nav2 never becomes active | `ros2 lifecycle get /controller_server`, TF `map->odom->base_link` chain, `use_sim_time` |
| robot does not move on Nav2 goals | `ros2 topic echo /cmd_vel` (collision_monitor output); `/cmd_vel_nav`, `/cmd_vel_smoothed` upstream; deadband 0.03 m/s / 0.10 rad/s |
| blocked in the doorway | footprint radius 0.34 m, doorway 1.0 m: raise `inflation_radius` down / `cost_scaling_factor` up |
