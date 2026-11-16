# Phase 2 step 2.8 — terrain curriculum (`terrain_v1`)

`terrain_v1` = `dr_energy_ft` 6M fine-tuned 60M steps (6.05 h) on a per-env terrain curriculum
(`rl/configs/terrain.yaml`), deadband + small-command sampling. Measured 2026-11-16 in the `rl` container
(MuJoCo). Raw data: `phase2_terrain_eval.json` (terrain grid, 30 episodes/cell), `phase2_terrain_scenarios.json`
(9 DR scenarios x 100 episodes + 144-start gate). Candidates: ft-6M, terrain_v1 at 20M, 40M (terrain grid only), 60M.

## Terrain grid (success % / fall % / window error % / CoT), selected rows
| terrain | tripod | ft-6M | tv1 20M | tv1 40M | **tv1 60M** |
|---|---|---|---|---|---|
| rough 4 cm | 100/0/25.7 | 100/0/13.7 | 100/0/13.3 | 100/0/16.6 | 100/0/15.0 |
| slope 8° | 100/0/19.3 | 87/0/29.1 | 97/0/14.1 | 97/0/17.3 | **100/0/13.0** |
| slope 10° | 100/0/21.0 | 67/3/44.0 | 93/3/16.3 | 97/0/17.7 | **100/0/14.1** |
| slope 12° | 100/0/22.5 | 43/3/63.4 | 100/0/16.0 | 97/0/17.8 | **100/0/15.1** |
| slope 15° | 100/0/25.4 | 33/7/75.7 | 93/7/20.1 | 97/0/19.5 | 93/7/17.5 |
| step 3 cm | 83/0/36.2 | 97/0/26.5 | 97/0/23.5 | 90/0/31.1 | 93/0/28.5 |
| step 4 cm | 57/0/50.0 | 77/0/40.1 | 83/0/37.0 | 77/0/38.7 | 80/0/38.3 |
| step 5 cm | 43/0/59.8 | 43/0/53.3 | 73/0/44.3 | 57/0/47.7 | 53/0/47.9 |

Slopes: ft-6M collapses beyond 8°; tv1 60M holds 100 % to 12°. Rough/steps: better than ft-6M or equal; 20M is
best on steps. CoT 5.6–6.2 (ft-6M 6.1–7.1, tripod 5.9–6.6).

## Flat DR scenarios (window error % ± SE; power W)
| | tripod | ft-6M | tv1 20M | tv1 60M |
|---|---|---|---|---|
| nominal | 15.9 ± 0.8 | **9.2 ± 0.6** | 11.2 ± 0.3 | 11.2 ± 0.5 |
| all_extremes | 13.7 | 10.3 | 11.5 | 14.2 |
| CoT (protocol) | 7.35 | 7.34 | 6.23 | **5.76** |
| mean power | 13.1 | 15.5 | 13.6 | **11.6** |

0 % falls in all 9 scenarios for every controller. **Flat regression**: nominal tracking 9.2 → 11.2 % (+2 pt,
still well under tripod 15.9 %); all_extremes 10.3 → 14.2 %. Energy better (CoT −21 %).

## 144-start gate (large / small from rest / small while walking)
| controller | failed | large | small rest | small walking |
|---|---|---|---|---|
| tripod | 0 | 0 | 0 | 0 |
| ft-6M | 91 | 0 | 46 | 45 |
| tv1 20M | 66 | 0 | 38 | 28 |
| **tv1 60M** | **40** | 0 | 40 | **0** |

## Selection: `terrain_v1` **60M (latest)**
Best slopes (100 % to 12°), best efficiency, small-while-walking commands fixed (0/48), large starts 0 fails.
Costs: +2 pt flat tracking error. **Not fixed: small-command starts from rest** (vx, −vx, vy, −vy, −wz: 8/8
failed each; +wz passes) — the gate still FAILS (40/144); the robot stays standing for |v| ≈ 0.03 m/s from rest.
**Known limitation (accepted 2026-11-16, no retraining):** `terrain_v1` 60M does not start from rest for
|v| ≈ 0.03 m/s (gate 40/144 fails, all in the small-from-rest group). Handled at deployment: Nav2
minimum speed 0.09 m/s (vx, vy) / 0.23 rad/s (wz) (was 0.05 / 0.15 until 2026-11-16, which `terrain_dr2` does not start at, `phase3_selfhit_and_minspeed.md`): since 2026-11-16 `cmd_vel_min_speed_node` LIFTS smaller
commands to it (the first version zeroed them via the smoother deadband and deadlocked MPPI's start, `phase3_gazebo_results.md`), above the dead zone (±0.03 / ±0.10); `rl_command` deadband stays below that
(0.015 / 0.05). Consequence: Nav2 can never command a creep below 0.09 m/s. Not covered: teleop or other
sources that publish small /cmd_vel directly (they hit the policy's dead zone and just stand).
ONNX: `models/terrain_v1_60M.{onnx,json,_parity.npz}` (default of `policy_node` since 2026-11-16; Gazebo transfer: `phase2_deploy_gazebo.md`).
