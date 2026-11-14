# Phase 1 — Gazebo ↔ MuJoCo actuator parity and chatter test

Measured 2026-11-14 in the `sim` container (Gazebo Harmonic gz-sim 8.15.0 / DART, MuJoCo 3.3.0),
headless, one simulator at a time. Raw JSON in `phase1_parity/`. Reproduce:

```bash
docker compose run --rm --no-deps sim scripts/parity_experiments.sh step  docs/results/phase1_parity
docker compose run --rm --no-deps sim scripts/parity_experiments.sh stand docs/results/phase1_parity
```

**Setup (robot.yaml):** PD `tau = 20·(q_ref − q) − 0.05·q̇`, clipped ±2.5 N·m, 500 Hz in both sims
(Gazebo: `gurdy_control/JointPDController` on the effort interface, physics 1 ms; MuJoCo: affine
actuator, 2 ms Euler step). Joint damping 0.42 N·m·s/rad, Coulomb friction 0.02 N·m, **no armature**.
Protocol and metrics: `gurdy_core/gurdy_core/experiments.py` (identical code for both sims).

## Step response (base fixed in the air, leg `lf`, from nominal stance)

Rise time 10–90 % of the final value; steady error = target − mean of last 0.3 s.
RTF: gazebo 1.00 (with 500 Hz recorder), mujoco 125

| joint | step rad | rise ms gz / mj | overshoot % gz / mj | steady err mrad gz / mj | peak vel rad/s gz / mj | parity |
|---|---|---|---|---|---|---|
| lf_hip_yaw | 0.05 | 40 / 44 | 0.0 / 0.0 | 1.19 / 0.02 | 1.60 / 1.56 | PASS |
| lf_hip_pitch | 0.05 | 46 / 46 | 0.0 / 0.0 | 6.25 / 4.70 | 1.72 / 1.74 | PASS |
| lf_knee | 0.05 | 48 / 44 | 0.0 / 0.0 | 2.29 / -0.11 | 1.87 / 1.89 | PASS |
| lf_hip_yaw | 0.30 | 50 / 52 | 0.0 / 0.0 | 1.19 / 0.02 | 5.88 / 5.88 | PASS |
| lf_hip_pitch | 0.30 | 55 / 56 | 0.0 / 0.0 | 5.98 / 4.58 | 5.68 / 5.68 | PASS |
| lf_knee | 0.30 | 57 / 56 | 0.0 / 0.0 | 2.87 / 0.33 | 5.85 / 5.91 | PASS |

Tolerances: {'rise_time_rel': 0.15, 'rise_time_abs_s': 0.005, 'overshoot_pp': 3.0, 'steady_error_rad': 0.003, 'body_height_m': 0.002}
STEP PARITY: PASS

## Standing 60 s on flat ground (after 5 s settling) — chatter test

RTF: gazebo 0.96 (with 500 Hz recorder), mujoco 77

| metric | gazebo | mujoco |
|---|---|---|
| worst qd_rms (rm_knee / rm_knee) | 4.49e-06 | 5.30e-14 |
| worst q_p2p (rm_knee / lm_knee) | 2.69e-04 | 2.98e-14 |
| worst tau_std (rm_hip_pitch / rm_hip_pitch) | 6.27e-04 | 2.81e-13 |
| chatter pass (limits {'qd_rms': 0.01, 'q_p2p': 0.005, 'tau_std': 0.05}) | True | True |
| sag hip_yaw mrad | 0.00 | -0.00 |
| sag hip_pitch mrad | 25.04 | 18.51 |
| sag knee mrad | 20.22 | -3.89 |
| body height (FK) m | 0.1080 | 0.1083 |
| samples / duration s | 29935 / 60.0 | 29999 / 60.0 |

STAND: chatter both pass = True, body height diff 0.22 mm (tol 2 mm) -> PASS

## Findings

1. **Actuator dynamics match.** With joint friction set to 0 (diagnostic run), the linear 0.05 rad
   steps matched to the millisecond (rise 22/22, 24/24, 30/30 ms; steady error equal to 0.01 mrad):
   inertia, PD sampling and implicit joint damping are identical in both sims.
2. **Velocity limit.** DART enforces the URDF joint velocity limit (6 rad/s) as a hard constraint;
   MuJoCo has none. With damping 0.25 the saturated steps reached 10 rad/s in MuJoCo but were capped at
   6 rad/s in Gazebo (rise 40–45 vs 30–34 ms). Fix: damping = effort / velocity = 0.42, which is the
   DC-servo torque-speed line; both sims now peak at 5.7–5.9 rad/s and rise times agree within 4 ms.
3. **Coulomb friction.** DART = hard stiction, MuJoCo `frictionloss` = soft constraint. Gives up to
   2.5 mrad steady-error difference (deadband 0.02/20 = 1 mrad). Accepted (tolerance 3 mrad).
4. **Standing joint sag distribution differs, body height does not** (0.1080 vs 0.1083 m).
   MuJoCo's soft contacts let the feet splay 2.2 mm outward while the robot settles (first 0.5 s,
   independent of spawn height); DART feet stick. A 6-foot stance is statically indeterminate, so
   this shifts sag between hip_pitch and knee. `cone=elliptic, impratio=100` reduces the initial splay
   but the feet then creep (140.0 → 141.8 mm in 60 s), so the default contact model is kept for now;
   revisit for RL in Phase 2.
5. **No chatter in either sim** (worst joint RMS velocity 4.5e-6 rad/s in Gazebo, limit 1e-2).
   Worst-case explicit-PD stability ratios: kd·dt/I ≤ 0.35, ω·dt ≤ 0.53 (limit 2).

## Performance / resources

Re-measured after isolating containers (`GZ_PARTITION`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`).
An earlier RTF measurement (1.02 headless / 0.85 GUI) was contaminated by a second Gazebo instance
on the same Docker network sharing `/stats` and `/clock`; it is withdrawn. The parity metrics above
were re-run after isolation and are identical to the earlier run.

| | Gazebo headless | Gazebo GUI (WSLg, d3d12) | MuJoCo |
|---|---|---|---|
| RTF, no recorder (30 s, 300 `/stats` samples) | 0.96 | 0.55 (shadows on or off) | — |
| RTF, with 500 Hz Python recorder | 0.54–0.98 | — | 47–125 |
| RAM | container 290–325 MiB | +~450 MiB GUI process | 62 MiB (step), 101 MiB (stand) peak RSS |

`/joint_states` now publishes at 100 Hz by default (`js_rate`, 500 Hz only for these tests);
`robot_state_publisher` CPU dropped from ~27–36 % to ~10–12 %.
Open: GUI RTF 0.55 — the GUI process uses ~100 % CPU; investigate before demos.

**Update (sensors step):** averaging instantaneous `/stats` RTF values proved unreliable; RTF is now
measured as sim/wall time over 10 s windows (`scripts/measure_rtf.py`). During this session the
host's speed varied ~3x (pure MuJoCo stepping benchmark 34–170× real time with nothing else running;
likely Windows power/thermal management), so the RTF values above are provisional and will be
re-measured on a stable host. Parity metrics use sim time and are unaffected.
