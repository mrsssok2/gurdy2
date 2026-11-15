# Step 2.9 — Gazebo deployment test (to run after `terrain_v1` finishes)

Prepared 2026-11-15 with `dr_energy_ft` 6M. **Nothing below has been run in Gazebo yet.** Already measured
without Gazebo: ONNX vs torch max abs diff 3.0e-7 (300 recorded observations, limit 1e-5); deployment
controller observation/targets equal the MuJoCo env (`rl/tests/test_policy_runtime.py`); ROS node logic
with fake messages (`ros2_ws/src/gurdy_policy/test`); the same protocol through the deployment path in
MuJoCo (no ROS): all segments within 2–9 %, no fall, tick 0.13 ms mean.

## 0. Preconditions
```bash
docker ps                                   # only gurdy_train may exist -> wait for it to finish, then:
docker ps                                   # must be empty (CLAUDE.md rule 8)
docker compose build sim                    # one-off: adds onnxruntime to the ROS image
docker compose run --rm sim bash
# inside:
cd ros2_ws && colcon build --symlink-install --parallel-workers 2 && source install/setup.bash && cd ..
(cd gurdy_core && python3 -m pytest -q); python3 -m pytest -q ros2_ws/src/gurdy_policy/test
```

## 1. Smoke test (GUI optional) and safety behaviour
```bash
# terminal 1 (container A):
ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=true controller:=policy
# terminal 2 (docker compose exec sim bash):
ros2 topic echo /policy/status                                  # expect stand:cmd_vel timeout
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.08}}"   # -> status: policy, walks
# Ctrl-C the pub: after 0.5 s -> stand:cmd_vel timeout, joints ramp to stance (<= 1 rad/s)
ros2 param set /policy_node mode tripod                         # A/B: same node, tripod gait (switch while stationary)
ros2 param set /policy_node mode policy
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
```
Log line `policy tick (obs+ONNX+mapping): mean … ms` every 500 ticks = inference cost.

## 2. Tracking + sim-to-sim gap (headless, one command, ~6 min)
```bash
scripts/policy_gazebo_test.sh /tmp/policy_deploy      # inside the sim container, nothing else running
```
Runs the shared 123 s `/cmd_vel` protocol in Gazebo with `controller:=policy` (IMU on), then the same
policy in MuJoCo through the deployment path (`scripts/policy_track_mujoco.py`), and prints the table
(per segment: Gazebo vs MuJoCo measured velocity, error %, MuJoCo-vs-Gazebo gap; falls, min body z, tilt).
Acceptance: no fall; report tracking error per sim and the gap. Baseline to compare: tripod in Gazebo,
`docs/results/phase1_tracking.md` (vx −8.3 %, wz −11.3 %); policy in MuJoCo (above).

## 3. Record
Save the table to `docs/results/phase2_deploy.md` (RTF, RAM via `docker stats`, tick time), tick the
ROADMAP 2.9 boxes. If the policy falls or drifts in Gazebo: first check `/policy/status`, then the
IMU frame (orientation/gyro), then the PD gains (`kp 20, kd 0.05`), before any retraining.
