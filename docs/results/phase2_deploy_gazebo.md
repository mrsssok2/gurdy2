# Step 2.9: RL policy in Gazebo (sim-to-sim transfer), 2026-11-16

Protocol: `scripts/policy_gazebo_test.sh` (Gazebo Harmonic headless, `controller:=policy`, IMU on, the shared
123 s /cmd_vel protocol at 20 Hz) vs the same ONNX through the same deployment code in MuJoCo
(`scripts/policy_track_mujoco.py`). Errors = main-axis error of the steady part of each segment (same analysis in
both sims). Raw JSON: `docs/results/data/policy_{ft6M,tv1_60M}/`. Clean `colcon build` from `ros2_ws` was needed
first (stale build dir was created for another mount path); 7 packages build, navigation tests 12/12.

## Result
| | ft-6M | terrain_v1 60M |
|---|---|---|
| falls (Gazebo / MuJoCo) | 0 / 0 | 0 / 0 |
| min body z Gazebo / MuJoCo (m) | 0.084 / 0.099 | 0.079 / 0.091 |
| max tilt Gazebo / MuJoCo (rad) | 0.078 / 0.075 | 0.085 / 0.088 |
| policy tick (obs + ONNX + mapping) | mean 0.31–0.37 ms, max ≤ 1.8 ms | mean 0.37–0.38 ms, max ≤ 2.0 ms |
| Gazebo RTF (headless, policy node running) | 0.94 | 0.87 |
| segments within ±25 % (Gazebo) | **no** | yes |

| segment | cmd | ft-6M Gz % | ft-6M MJ % | tv1 Gz % | tv1 MJ % | tv1 gap (pts) |
|---|---|---|---|---|---|---|
| forward | vx 0.10 | −62.8 | +9.0 | −21.5 | −3.7 | 17.8 |
| backward | vx −0.10 | +9.2 | +6.8 | −0.7 | −2.9 | 2.2 |
| strafe_left | vy 0.08 | −69.0 | +8.0 | −17.5 | −0.2 | 17.3 |
| strafe_right | vy −0.08 | −41.7 | +2.3 | +6.9 | +14.5 | 7.6 |
| turn_left | wz 0.3 | −2.0 | +2.2 | −11.2 | −5.6 | 5.6 |
| turn_right | wz −0.3 | −104.9 | +4.7 | −0.5 | +4.8 | 5.3 |
| combined | (0.08, 0.04, 0.2) | −20.3 | +7.4 | −15.4 | +6.7 | 22.1 |

## Verdict: transfer does NOT meet the target (no falls, but not within ~5 pts of MuJoCo)
- **ft-6M fails**: it stalls (stands or barely moves) on forward, strafe_left, strafe_right and turn_right in
  Gazebo, no falls. Not usable in Gazebo.
- **terrain_v1 60M is usable but off target**: 0 falls, all segments within ±25 %, but forward, strafe_left and
  combined are 17–22 pts below MuJoCo (backward, turn_left and turn_right are within 6 pts, strafe_right 7.6). The open-loop tripod has a Gazebo-vs-MuJoCo gap of 3–11 pts
  (`phase1_tracking.md`); the policy roughly doubles the worst case.
- `policy_node` default model switched to `terrain_v1_60M.onnx`.

## Diagnosis (nothing in the robot/policy was changed; diagnostic scripts only)
Recorded joint states, IMU and targets in both sims (`scripts/policy_diag.sh`, `policy_diag_analyze.py`,
`policy_diag_latency.py`, `policy_diag_sweep.sh`; terrain_v1).
| hypothesis | evidence | verdict |
|---|---|---|
| obs mismatch / wiring | standing IMU gravity/gyro sane, gyro noise ~0.03–0.07 rad/s (training noise 0.05); stride amplitudes per joint type and gyro rms in walking segments match MuJoCo within ~15 %; ONNX/obs parity tests exist | not the cause |
| tick timing / jitter | commands 50.0 Hz, gaps p99 23 ms, max 25 ms (rerun; first run had one 217 ms gap); command arrives ≈1 ms after its /joint_states stamp; IMU/joint_states receipt−stamp ≈ 0 ms | not the cause, but total loop latency ≈ 3–8 ms (state ≤ 10 ms old at 100 Hz joint_states, IMU ≤ 5 ms) |
| latency sensitivity (MuJoCo, extra delay on targets) | forward / strafe_left / turn_left error: 0 ms −4/0/−6, 10 ms −9/−4/−11, 20 ms −15/−18/−17, 30 ms −17/−16/−18 | **contributes**: the policy loses speed with delay (training DR covered 0–10 ms); Gazebo's ~5 ms explains only ~5 of the 17–22 pts |
| foot friction | MuJoCo friction ×0.6: forward −1 (was −4), others ±1 | not the cause |
| PD gains | same kp/kd/effort/damping in both (Phase 1 step parity 4 ms) | not the cause, but see next row |
| contact / load distribution | policy stands with **asymmetric knees** (targets 1.2–1.9 rad); in Gazebo the more extended legs sag 45–95 mrad below target, MuJoCo ≤ 20 mrad; joint-target rms error 0.037 vs 0.022 rad standing, 0.063 vs 0.035 walking | **main suspect**: stance-load distribution differs between DART and MuJoCo soft contacts (known from Phase 1, `phase1_parity.md`); the tripod's symmetric stance hides it, the policy's asymmetric stance exposes it |
Not isolated (would need more runs): contribution of the foot contact model alone (e.g. `impratio`/solref in
MuJoCo vs DART contact) and of Gazebo's hard joint friction.

## Options (not done, need a decision)
1. Randomise what differs: wider latency DR (to ~20 ms), per-leg joint-target offsets / kp scale, contact softness
   (`solref`/`impratio`) in training, then fine-tune terrain_v1 (cheap, ~hours).
2. Tune MuJoCo contacts/limits to match DART's stance sag first (measure per-joint standing error in both).
3. Accept terrain_v1 in Gazebo as is (all segments ±25 %, no falls) and move on to Phase 3 with it.
