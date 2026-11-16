# Lidar self-hits (collision monitor) and start check at the Nav2 minimum speed — 2026-11-16

## 1. The lidar sees the robot's own legs (hypothesis confirmed)
`scripts/scan_selfhit_run.sh` (open world, policy `terrain_dr2`, headless): `/scan_raw` recorded while standing 20 s and walking
(vx 0.1) 40 s; each return < 0.5 m is compared with the leg chains (coxa→femur→tibia→foot from TF) and the body box.
- Standing: 1–2 scans with a return < 0.5 m (a swung femur at rest).
- Walking: 91 % of scans (321/351) have returns at **0.15–0.29 m** (median 0.21; 4 % exactly at range_min), all around the circle,
  median distance to the nearest **femur** segment 1.9 cm (max 4.5 cm), 7–12 cm from the body box → legs, not the body. In swing the
  femurs rise to the lidar plane (0.05 m above base_link). The footprint hexagon (circumradius 0.34 m) contains all of them,
  so FootprintApproach saw obstacles inside the robot and throttled /cmd_vel (0.04 → 0.002 m/s).
- Why not only `range_min`: it would have to be ≥ 0.30 m, blinding the robot to everything closer than 30 cm.

Fix: `scan_self_filter_node` (gurdy_navigation): Gazebo bridge now publishes `/scan_raw`; the node sets every return inside the Nav2
footprint (+2 cm) to +inf and publishes `/scan` (slam_toolbox, AMCL, costmaps, collision_monitor unchanged). `laser_filters` is not in the image;
the node is a numpy polygon mask (beam exit radius precomputed per angle). Footprint-approach polygon **re-enabled**.
Filtered scan, same protocol: 0 returns < 0.5 m while standing and walking, minimum range 2.1 m (real walls).

Nav2, 5 goals, policy terrain_dr2 15M, GT odometry, headless, **collision monitor ON** (`docs/results/data/phase3/nav_dr2_cm_filtered*`):
| goal | min speed 0.05/0.15: result, time, final error, recoveries | min speed 0.09/0.23 (final) |
|---|---|---|
| (1.5, 0.5) | OK, 54.9 s, 0.098 m, 0 | OK, 45.0 s, 0.110 m, 0 |
| (4.0, 0.5) doorway | OK, 81.6 s, 0.110 m, 1 | OK, 78.1 s, 0.053 m, 0 |
| (4.0, −2.2) | OK, 40.1 s, 0.072 m, 0 | OK, 80.0 s, 0.097 m, 0 |
| (1.5, −0.5) back | OK, 157.9 s, 0.216 m, 0 | OK, 94.0 s, 0.077 m, 0 |
| (0, 0) | OK, 76.2 s, 0.091 m, 0 | OK, 43.3 s, 0.038 m, 0 |
Success 5/5 and 5/5 (before: 5/5 only with the monitor off; with it on 0/2). RTF 0.57 / 0.48 (sim seconds). One earlier launch of the first
configuration aborted all goals at 0 s (bt_navigator timed out waiting for the planner action server during start-up, unrelated to
the scan; not repeated — rerun is the table). Times are sim seconds.

## 2. Start check at the operating point (`rl/tools/min_speed_start.py`, MuJoCo, terrain_dr2 15M)
6 commands × 8 start timings (0–0.7 s after 2 s rest); failed = mean speed 3–11 s after the command < 20 % of it.
| lin m/s / ang rad/s | failed starts /48 | failing cells |
|---|---|---|
| 0.05 / 0.15 (previous minimum) | **38** | vx+ 6/8, vx−, vy±, wz− 8/8 each (only wz+ passes) |
| 0.06 / 0.18 | 32 | vx−, vy±, wz− |
| 0.07 / 0.21 | 24 | vx−, vy−, wz− |
| 0.08 / 0.24 | 8 | vy− |
| 0.085 / 0.22 | 13 | vy− 5/8, wz− 8/8 |
| **0.09 / 0.23** | **0** (speed ratios 0.96–1.10) | – |
| 0.095 / 0.24, 0.10 / 0.30 | 0, 0 | – |
The cliff is sharp (success is all or nothing per direction). Smallest passing at 0.005 m/s / 0.01 rad/s resolution: **0.09 m/s, 0.23 rad/s**
→ `MIN_LINEAR_SPEED`/`MIN_ANGULAR_SPEED` in `gurdy_core/rl_command.py`. The margin is thin (0.085 fails); 0.095 / 0.24 also passes if more margin is wanted.

Gazebo, each of the 6 directions once (`scripts/gz_start_check_run.sh`, sim-time windows 6 s after 4 s start-up, ratio = measured / commanded):
| level | vx+ | vx− | vy+ | vy− | wz+ | wz− |
|---|---|---|---|---|---|---|
| 0.05 / 0.15 | 1.04 | 1.26 | 0.95 | 1.23 | 1.01 | 1.12 |
| 0.09 / 0.23 | 1.07 | 1.11 | 0.96 | 1.08 | 0.99 | 1.09 |
Gazebo starts at both levels (the sim-to-sim difference is in the start behaviour, as for the earlier gate); MuJoCo is the stricter check, so its level is used.
Consequence: Nav2 cannot creep below 0.09 m/s (vy limit is 0.10) and 0.23 rad/s; goal tolerance (xy 0.12 m, yaw 0.2 rad) held in the 5-goal run (final errors ≤ 0.11 m).
