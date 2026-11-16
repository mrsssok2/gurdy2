# Step 2.9b + Phase 3 re-test results (2026-11-16)

## terrain_dr2 (15M steps from terrain_v1 60M) vs terrain_v1 60M — MuJoCo (`rl/runs/terrain_dr2/eval_*.log`)
| | terrain_v1 60M | terrain_dr2 15M | terrain_dr2 10M |
|---|---|---|---|
| nominal window error | 11.2 ± 0.5 % | **10.6 ± 0.3 %** | 11.1 ± 0.5 % |
| target_offset error | 46.6 % | **28.1 %** | 29.4 % |
| soft_contact error | 15.0 % | **13.4 %** | 13.5 % |
| all_extremes error | 14.2 % | **11.2 %** | 11.8 % |
| latency 18 ms error | 17.2 % | **14.7 %** | – |
| falls (12 scenarios) | 0 | 0 | 0 |
| CoT | 5.76 | 5.77 | 5.94 |
| start gate (144) | 40 fail | 65 fail | 64 fail |
| slopes 12° / 15° success | 100 / 93 % | 93 / 97 % | 97 / 83 % |
| steps 4 / 5 cm success | 80 / 53 % | 83 / 63 % | 90 / 63 % |
Selected: **15M** (`models/terrain_dr2_15M.onnx`). Weaker: start gate (from-rest small commands; handled at deployment by the
min-speed lift) and 12° slope (93 %). **Eval bug found:** scenario `latency_20ms` (10 substeps = a whole control step) never applies
a target, so every controller incl. tripod scored 100 % error; replaced by `latency_18ms` (9 substeps). Training range [0,10] substeps
has the same flaw only at exactly 10 (negligible probability).

## Gazebo sim-to-sim (`scripts/policy_gazebo_test.sh`, data `docs/results/data/policy_dr2*`, `policy_tv1*`)
Gap = |MuJoCo − Gazebo| error, main axis. Gazebo error itself: dr2 −10.7…+12.7 %, v1 −22.5…+8.1 %.
| segment | v1 gap (pts) | dr2 gap (pts) |
|---|---|---|
| forward | 24.3 | 12.2 |
| backward | 2.9 | 1.3 |
| strafe left | 23.6 | 13.7 |
| strafe right | 5.9 | 0.9 |
| turn left / right | 6.4 / 5.6 | 7.1 / 8.3 |
| combined | 26.9 | 31.3 |
Worst single-axis gap 24 → 14 pts, Gazebo tracking error of the robot ≤ 12.7 % (v1 22.5 %), 0 falls. NOT within 10 pts everywhere
(forward, strafe-left 12–14; combined 31, MuJoCo overshoots there). Partly closed.

## Nav2, 5 goals, policy terrain_dr2 15M, GT odometry (`docs/results/data/phase3/nav_dr2_*`)
Collision monitor ON: goals 1–2 aborted (21 recoveries, errors 2.0 / 2.7 m): `cmd_vel_min` ≈ 0.04 m/s but `/cmd_vel` ≈ 0.002 m/s —
the FootprintApproach polygon throttles the robot to a crawl. Disabled in `nav2_params.yaml` (needs a proper fix: scan self-hits / footprint).
Collision monitor OFF (RTF 0.60):
| goal | status | time s | path m | final err m | recoveries |
|---|---|---|---|---|---|
| (1.5, 0.5) | ok | 59.7 | 4.31 | 0.098 | 0 |
| (4.0, 0.5) | ok | 95.3 | 6.84 | 0.127 | 0 |
| (4.0, −2.2) | ok | 105.0 | 8.09 | 0.083 | 0 |
| (1.5, −0.5) | ok | 172.0 | 12.70 | 0.137 | 1 |
| (0, 0) | ok | 142.3 | 10.73 | 0.134 | 0 |
Without the min-speed deadlock fix earlier: 0/5. The first-window check passes (cmd_vel_min 0.048 from cmd_vel_nav 0.025).

## EKF mapping run (tripod, bag) — `docs/results/data/phase3/flat_lab_tripod_ekf2`
18/18 waypoints, 416 sim s, no hang, RTF 0.56–0.82. **The map is NOT distorted this time:** precision 0.033 m, p95 0.083 m, recall
99.8 % (GT-odom map: 0.024 / 0.064 / 95.4 %). Pose error vs ground truth: EKF 0.059 m mean (0.137 max), legs 0.075 (0.186), slam
0.037 (0.092); yaw ≤ 1.9°; TF == /odometry/filtered. `bag_overlay` dead-reckoned scan maps score 0.37–0.44 m for ALL sources incl. ground
truth, so that metric is dominated by scan/pose timing and cannot rank sources. Conclusion: no pose source draws the walls wrong; the first
EKF run's distortion did not reproduce (it ran at RTF 0.57 and hung; cause stays unexplained, plausibly load/timing). Bag was not finalized (killed
recorder, no metadata.yaml): `scripts/bag_export.py` now reads the bare mcap.
