# Phase 3 — leg odometry: offline drift test (MuJoCo, no Gazebo)

Measured 2026-11-15 with `python3 -m sim_mujoco.leg_odometry_eval --out docs/results/phase3_leg_odometry.json`
(inside the `rl` image: `docker run --rm -v $PWD:/w -w /w -e PYTHONPATH=/w/gurdy_core:/w/sim_mujoco gurdy2/rl:cpu ...`).
One single-process MuJoCo walk with the **tripod gait** on flat ground (one run, one noise seed): four 12 s
segments, each followed by 2 s of standing. Joint states + IMU sampled at 100 Hz (as Gazebo's `/joint_states`);
`gurdy_core.leg_odometry` run on the recording; pose dead-reckoned per segment from the ground-truth pose at the
segment start. The policy was not tested here (training was running).

How it works (`gurdy_core/leg_odometry.py`): stance feet = feet within 1.5 cm of the lowest one (tilt-corrected
with the IMU orientation), sliding outliers dropped (> 5 cm/s residual); stance-foot velocity from the analytic
leg Jacobian; the foot sphere's **rolling** is compensated (contact point instead of centre); body velocity =
−(foot velocity + ω × foot position), averaged over the stance feet. Without a gyro, wz comes from a planar
least squares over the stance feet.

| variant / segment | path (GT) | position error | heading (GT) | heading error | velocity RMSE | wz RMSE |
|---|---|---|---|---|---|---|
| **legs only** forward 0.10 m/s | 1.16 m | 0.197 m = **17.0 %** | +1.1° | −0.4° | 0.039 m/s | 0.023 rad/s |
| legs only strafe 0.08 m/s | 0.99 m | 0.024 m = 2.4 % | +0.1° | −1.0° | 0.016 | 0.012 |
| legs only turn 0.3 rad/s | (0.06 m) | — (rotation in place) | −174.5° | **−13.0°** (7.5 %) | 0.008 | 0.085 |
| legs only arc 0.08 m/s + 0.2 rad/s | 0.93 m | 0.218 m = **23.5 %** | +123.1° | **−28.7°** (23 %) | 0.034 | 0.121 |
| **legs + gyro** forward | 1.16 m | 0.008 m = **0.7 %** | +1.1° | 0.0° | 0.009 | 0.010 |
| legs + gyro strafe | 0.99 m | 0.017 m = 1.7 % | +0.1° | 0.0° | 0.016 | 0.003 |
| legs + gyro turn | (0.06 m) | — | −174.5° | +1.3° | 0.009 | 0.019 |
| legs + gyro arc | 0.93 m | 0.010 m = 1.1 % | +123.1° | +0.7° | 0.009 | 0.015 |

"legs + gyro" = vx, vy from the legs using the gyro as angular rate, heading from the gyro — what the EKF
effectively does when it fuses the IMU yaw rate with the leg twist. Mean stance feet: 3.4–3.6 (tripod: 3).

Reading it:
- **Translation from the legs is good (≈ 1–2 % of path) once the rotation comes from the gyro.** Without the gyro
  the planar least squares couples noisy leg-derived rotation into vx, vy and gives 17–24 % on forward/arc.
- **Leg-only heading is not usable** (13° over a 174° turn, 29° over a 123° arc): the EKF must take yaw from the IMU.
  `ekf.yaml` does: yaw (differential) + yaw rate from the IMU, only vx, vy, wz from the legs.
- The covariance model (`LegOdomConfig`: σ = √((σ0² + (k·speed)²)/n_stance)) is set from these velocity errors
  (v: 0.012 m/s + 20 % of speed; wz: 0.03 rad/s + 20 % of |wz|). It is a first calibration from one flat-ground run.
- Not covered: long paths (segments are ~1 m), terrain, the policy gait, Gazebo contact differences
  (MuJoCo foot splay ~2 mm, see CLAUDE.md), real-robot slip. Per-percent numbers over longer distances are unknown.
- The ground-truth/EKF comparison in Gazebo is step 6 of `phase3_test_plan.md`.
