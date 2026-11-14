# Phase 1 — Tripod gait velocity tracking (Gazebo and MuJoCo)

Measured 2026-11-14 in the `sim` container. Same `/cmd_vel` protocol and analysis code in both
sims (`gurdy_core/gurdy_core/experiments.py`), same gait code (`gurdy_core/gurdy_core/gait.py`,
50 Hz, cycle 0.8 s, step height 4 cm). Gazebo: `gurdy_bringup` headless, `sensors:=false`,
ground truth from `/odom/ground_truth`. MuJoCo: generated MJCF, free base. Raw JSON in
`phase1_tracking/`. Reproduce:

```bash
docker compose run --rm --no-deps sim scripts/tracking_experiments.sh docs/results/phase1_tracking
```

Protocol (123 s): 3 s stand, then seven 12 s segments separated by 2 s stops, then 20 s of random
commands (new command every 2 s, within limits). Mean body-frame velocity per segment, skipping the
first 2 s (acceleration limit + start-up half-speed step). Error on the segment's main axis.

RTF gazebo 0.77, mujoco 35 | fell: gazebo False, mujoco False | min body z 0.105 / 0.104 m | max tilt 0.008 / 0.011 rad

| segment | command (vx, vy, wz) | gazebo measured | err % | mujoco measured | err % | mj vs gz (main axis) |
|---|---|---|---|---|---|---|
| forward | (+0.100, +0.000, +0.000) | (+0.092, -0.000, +0.000) | -8.2 | (+0.098, -0.000, +0.002) | -2.3 | +6.4 |
| backward | (-0.100, +0.000, +0.000) | (-0.092, -0.000, -0.000) | -8.3 | (-0.097, -0.000, -0.001) | -2.6 | +6.2 |
| strafe_left | (+0.000, +0.080, +0.000) | (-0.000, +0.075, +0.000) | -5.9 | (+0.000, +0.084, +0.000) | +4.5 | +11.0 |
| strafe_right | (+0.000, -0.080, +0.000) | (+0.000, -0.075, -0.000) | -5.7 | (+0.000, -0.083, +0.000) | +4.2 | +10.5 |
| turn_left | (+0.000, +0.000, +0.300) | (+0.000, -0.000, +0.266) | -11.3 | (+0.000, +0.000, +0.274) | -8.7 | +2.9 |
| turn_right | (+0.000, +0.000, -0.300) | (-0.000, -0.000, -0.266) | -11.4 | (-0.000, +0.000, -0.273) | -8.8 | +2.9 |
| combined | (+0.080, +0.040, +0.200) | (+0.074, +0.038, +0.179) | -10.7 | (+0.077, +0.042, +0.180) | -10.0 | +0.8 |

Gazebo all within ±25 % and no fall: True | MuJoCo: True

**Phase 1 AC check:** vx = 0.10 m/s → Gazebo −8.2 % and yaw 0.3 rad/s → −11.3 % (limit ±25 %);
no fall in the 2-minute session (min body height 0.105 m, max tilt 0.008 rad); MuJoCo forward
speed within +6.4 % of Gazebo (limit ±30 %). Gazebo undershoots slightly more than MuJoCo, likely
foot contact (DART stiction vs MuJoCo soft contacts, see `phase1_parity.md`). RTF values here are
provisional (host speed varied this session; see `phase1_parity.md`).
