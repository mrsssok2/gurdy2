# Phase 3 Gazebo test (headless, flat_lab, 2026-11-16)

Scripts: `scripts/phase3_map_run.sh` (SLAM-only mapping run, scripted route `scripts/drive_route.py` = closed loop on ground-truth
odom: 18 waypoints, both rooms, doorway there and back, 37.9 m), `scripts/phase3_nav_run.sh` + `scripts/nav_eval.py` (Nav2 goals),
`scripts/map_quality.py`, `scripts/plot_map.py`. Raw logs/JSON: `docs/results/data/phase3/`. Maps: `maps/flat_lab_*.{yaml,pgm}`
(posegraphs gitignored, 15–23 MB). Images: `docs/results/phase3_map_*.png` (black = map, red = true outline, blue = GT path).
RViz screenshots were not taken (headless runs, no GUI); the PNGs are rendered from the saved maps.

## 1–2, 4. Mapping (lidar + slam_toolbox, `navigation:=false`)
| run | odometry | route | RTF | precision mean / p95 | recall < 10 cm | result |
|---|---|---|---|---|---|---|
| tripod | ground truth | 18/18, 416 s, 0 stalls, end 0.14 m from start | 0.99 | **0.024 m / 0.064 m** | 95.4 % | clean map, no doubled walls, door ≈ 1 m |
| policy (terrain_v1 60M) | ground truth | 18/18, 482 s, 0 stalls, end 0.15 m from start | 0.99 | 0.031 m / 0.123 m | 97.7 % | clean, thicker walls (2336 vs 1523 cells) |
| tripod | leg odometry + IMU EKF | 18/18, 416 s, 0 stalls | 0.57 (host benchmark 69 vs 126: host was slower) | **1.28 m / 3.95 m** | 38.9 % | **map unusable** (west wall at x ≈ −6 m and −5.3 m instead of −2 m) |

Scan rate 9.94 Hz (7.4 Hz in the EKF run). RAM (RSS): gz sim ≈ 0.5 GiB, each python ROS node 60–85 MiB, whole container 0.5–0.84 GiB.
The policy walks the same route 16 % slower (482 vs 416 s) with 25 % more path (47.4 vs 37.9 m: more lateral corrections); its map is
thicker because body pitch/roll sweeps the single lidar plane (walls p95 12 cm vs 6 cm), not misaligned.

### Finding: EKF-odometry map is bad although the odometry is good
- Open-loop probe (tripod, `scripts/ekf_probe.py`): forward 20 s: GT 1.823 m / EKF 1.794 m (−1.6 %); turn 15 s: yaw GT 46.7° / EKF 46.3° (mod 360);
  strafe 0.636 vs 0.663 m; second forward 1.393 vs 1.378 m (−1.1 %). Heading from the IMU works.
- Full route (`scripts/ekf_track.py`; that run hung at 133 s of sim time, `/clock` stopped, cause unknown): EKF vs GT within 0.03 m at every
  logged point up to 125 s, i.e. while the map's west wall was already being mapped wrongly.
- So the distortion is **not** odometry drift. Untested suspects: slam_toolbox with the EKF's TF vs the GT TF node; load/timing at RTF 0.57;
  scan matching in a bad minimum after the first 180° turn. Next: bag + offline analysis (`docs/results/phase3_retest.md`).

## 3. Nav2 goals with the policy: **failed (0/1 goals reached, run stopped)**
Localization on `flat_lab_tripod_gt` + Nav2, `controller:=policy`, `odom_source:=ground_truth`. (First attempt: all goals aborted at once because my
wait script matched "inactive" as "active"; fixed in the script, not a robot problem.) Goal 1 (1.5, 0.5): ABORTED after 347 s sim time,
21 recoveries, 84 s stall, robot ended 1.9 m from the goal (start was 1.6 m away). Probe (`scripts/nav_probe.py`, 10 s windows):

| window | /cmd_vel_nav (MPPI) mean (vx, vy, wz) | /cmd_vel_smoothed | /cmd_vel | robot moved |
|---|---|---|---|---|
| 0–80 s | (+0.024, +0.004, +0.010) | (0, 0, 0) | none | 0.00–0.01 m |
| 80–90 s (recovery spin) | (0.002, 0.001, +0.450) | (0, 0, +0.446) | wz ≈ 0.04 | 0.05 m |

**Cause: start deadlock between MPPI and the minimum speeds.** MPPI plans from the *measured* velocity (0 at standstill); with `ax_max 0.4` m/s²
and `model_dt 0.1` s its first command is at most 0.04 m/s (az_max·dt = 0.15 rad/s), below the `velocity_smoother.deadband_velocity`
0.05 m/s / 0.15 rad/s then in force, so the smoother sent zero, the robot stayed at 0, and MPPI never got above the deadband.
`SimpleProgressChecker` aborted ("Failed to make progress"); only the recovery spins passed. Goals 2–6 were not run, so success rate / path length /
final error are **not available**; AMCL localisation (initial pose 0,0) was fine.

## Follow-up (2026-11-16, code only): fixes prepared, not yet run in Gazebo
- Option 1 implemented: `cmd_vel_min_speed_node` (smoother → node → collision_monitor) lifts small commands to the minimum; smoother deadband 0;
  goal tolerance xy 0.12 / yaw 0.2.
- EKF wiring reviewed statically, no defect found; bag-based analysis tooling added. Commands: `docs/results/phase3_retest.md`.
