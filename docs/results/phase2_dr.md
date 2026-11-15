# Phase 2 step 2.7 — Domain randomisation, asymmetric critic, energy/smoothness curriculum

Policy **`dr_energy_v1`, BC-initialised from the tripod gait** (via `flat_bc_v1`), then PPO with domain
randomisation (DR), an asymmetric critic and an energy/smoothness curriculum. MuJoCo, measured
2026-11-15 in the `rl` container. Raw data: `phase2_dr_checkpoints.json`, `phase2_dr_scenarios.json`,
`phase2_dr_smoothness.json`. Video (tripod | flat_bc_v1 | dr_energy_v1 40M):
[`docs/videos/phase2_3way_tripod_flat_dr.mp4`](../videos/phase2_3way_tripod_flat_dr.mp4).

**Selected checkpoint: 40M steps** (not the final one — see "late-training degeneration" below).

## Setup (`rl/configs/dr_energy.yaml`)
* Start: `flat_bc_v1` 60M, transferred into an asymmetric actor-critic (actor copied exactly; critic copied
  with zero weights on 17 new privileged inputs), critic warm-up 0.4M steps.
* Actor input unchanged (195 proprioceptive values); critic also gets true body velocity, angular velocity,
  foot contacts, friction, payload, mass scale, motor strength, latency.
* DR (ranges widened 0 → full over 10M steps): friction 0.5–1.25, body mass ±15 % + payload 0–0.3 kg
  ±2 cm off-centre, link masses ±10 %, effort ±15 %, kp ±10 %, damping ±20 %, joint friction 0–0.04 N·m,
  latency 0–10 ms, sensor noise (gyro 0.05, gravity 0.02, joint pos 0.005, joint vel 0.1), pushes
  0.1–0.2 m/s every 5–8 s. Leg failure implemented but off (Experiment B).
* Efficiency curriculum (final weights reached at 20M): energy −2e-3 → −2e-2 per W, torque −5e-4 →
  −2.5e-3, action rate −0.01 → −0.5, action 2nd difference 0 → −2.0.
* 60,006,400 steps in **5.03 h** (3.2–5.5k samples/s), RAM 1.58–1.66 GiB.

## Training curves (stochastic policy, DR active)

| metric | 0.04M | 7.54M | 15.03M | 22.53M | 30.02M | 37.52M | 45.02M | 52.51M | 60.01M |
|---|---|---|---|---|---|---|---|---|---|
| return | 318 | 877 | 726 | 709 | 737 | 783 | 795 | 816 | 835 |
| fall | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| trk_err | 0.673 | 0.168 | 0.201 | 0.182 | 0.2 | 0.215 | 0.215 | 0.222 | 0.178 |
| r_track | 0.422 | 1.18 | 1.16 | 1.16 | 1.17 | 1.11 | 1.13 | 1.12 | 1.19 |
| r_energy | -0.069 | -0.251 | -0.319 | -0.381 | -0.324 | -0.293 | -0.286 | -0.26 | -0.249 |
| r_arate | -0.00151 | -0.0254 | -0.0338 | -0.0442 | -0.0361 | -0.0352 | -0.0354 | -0.0319 | -0.0305 |
| r_torque | -0.00837 | -0.0171 | -0.0215 | -0.0247 | -0.0215 | -0.0205 | -0.0203 | -0.0193 | -0.0187 |
| r_a2nd | -0.000244 | -0.0512 | -0.0685 | -0.0814 | -0.0709 | -0.0663 | -0.0652 | -0.062 | -0.0613 |
| eff_std | 0.0406 | 0.0395 | 0.0384 | 0.0374 | 0.0366 | 0.0359 | 0.0354 | 0.035 | 0.0346 |
| kl | 0 | 0.0206 | 0.0209 | 0.0197 | 0.0219 | 0.021 | 0.0232 | 0.0146 | 0.0239 |
| clipfrac | 0 | 0.243 | 0.239 | 0.254 | 0.248 | 0.233 | 0.244 | 0.199 | 0.169 |
| expl_var | 0.396 | 0.625 | 0.877 | 0.863 | 0.871 | 0.826 | 0.856 | 0.81 | 0.77 |
| clipped | 0.414 | 0.0662 | 0.119 | 0.125 | 0.106 | 0.127 | 0.103 | 0.126 | 0.0713 |
| eff_prog | 0.00102 | 0.376 | 0.751 | 1 | 1 | 1 | 1 | 1 | 1 |
| dr_scale | 0.00205 | 0.752 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| fps | 5.55e+03 | 3.72e+03 | 3.42e+03 | 3.45e+03 | 3.33e+03 | 3.24e+03 | 3.17e+03 | 3.2e+03 | 3.31e+03 |
| ram_MiB | 1.58e+03 | 1.66e+03 | 1.64e+03 | 1.66e+03 | 1.65e+03 | 1.64e+03 | 1.62e+03 | 1.6e+03 | 1.66e+03 |

Mean power implied by r_energy at final weight: ~19 W at 22.5M → ~12.5 W at 60M (DR + training commands).
Asymmetric critic: explained variance 0.77–0.88 (flat_bc_v1: 0.57–0.67). No falls in 60M steps.

## Checkpoint selection (nominal physics, deterministic)
Phase 1 tracking protocol (123 s) + 100 training-distribution episodes, same `rl/eval.py` run for all:

| controller | forward | backward | strafe_left | strafe_right | turn_left | turn_right | combined | falls | energy J | CoT | mean power W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod | -1.5 % | -1.5 % | +4.3 % | +4.1 % | -8.6 % | -8.5 % | -8.5 % | 0 | 1614 | 7.35 | 13.1 |
| flat_bc_v1:latest | +2.9 % | +1.7 % | +1.9 % | +8.8 % | +4.4 % | +3.2 % | +8.0 % | 0 | 4602 | 18.29 | 37.4 |
| dr_energy_v1:20000000 | +4.8 % | +7.3 % | +11.4 % | -100.0 % | +0.5 % | -0.0 % | +6.0 % | 0 | 2541 | 10.96 | 20.7 |
| dr_energy_v1:30000000 | +6.9 % | +9.4 % | +10.8 % | +10.6 % | -2.0 % | -4.7 % | +8.2 % | 0 | 2297 | 8.69 | 18.7 |
| dr_energy_v1:40000000 | +3.0 % | +4.1 % | +9.2 % | +5.0 % | -1.7 % | -2.5 % | +5.2 % | 0 | 1963 | 7.70 | 16.0 |
| dr_energy_v1:50000000 | +4.3 % | +13.2 % | -100.0 % | +6.0 % | -0.6 % | -5.0 % | +2.4 % | 0 | 1606 | 6.84 | 13.1 |
| dr_energy_v1:latest | +6.3 % | +8.7 % | -100.0 % | +4.8 % | +1.7 % | +1.3 % | -1.8 % | 0 | 1615 | 7.06 | 13.1 |


| scenario | tripod | flat_bc_v1:latest | dr_energy_v1:20000000 | dr_energy_v1:30000000 | dr_energy_v1:40000000 | dr_energy_v1:50000000 | dr_energy_v1:latest |
|---|---|---|---|---|---|---|---|
| nominal (n=100) | 0 % / 15.9 % / 13.2 W | 0 % / 4.9 % / 39.2 W | 0 % / 10.3 % / 23.7 W | 0 % / 10.6 % / 19.9 W | 0 % / 10.9 % / 17.3 W | 0 % / 9.4 % / 16.3 W | 0 % / 9.2 % / 16.2 W |

**−100 % = the robot did not move at all** on that segment (measured 0.000 m/s on every axis, no fall).
A start-from-rest test (`rl/tools/start_test.py`: stand 2 s + 0–0.7 s, then a pure command for 8 s, at
8 gait-clock phases; failure = < 20 % of the commanded speed) confirms it:

| controller | +vx | -vx | +vy | -vy | +wz | -wz | total failed |
|---|---|---|---|---|---|---|---|
| tripod | 0/8 (0.97) | 0/8 (0.97) | 0/8 (1.01) | 0/8 (1.01) | 0/8 (0.88) | 0/8 (0.88) | 0/48 |
| flat_bc_v1:latest | 0/8 (1.01) | 0/8 (1.00) | 0/8 (0.98) | 0/8 (1.06) | 0/8 (0.99) | 0/8 (0.99) | 0/48 |
| dr_energy_v1:30000000 | 0/8 (1.06) | 0/8 (1.09) | 0/8 (1.10) | 0/8 (1.08) | 0/8 (0.94) | 0/8 (0.92) | 0/48 |
| dr_energy_v1:40000000 | 0/8 (1.02) | 0/8 (1.03) | 0/8 (1.06) | 0/8 (1.03) | 0/8 (0.93) | 0/8 (0.94) | 0/48 |
| dr_energy_v1:50000000 | 0/8 (1.03) | 0/8 (1.11) | 4/8 (0.50) | 0/8 (1.03) | 0/8 (0.95) | 0/8 (0.92) | 4/48 |
| dr_energy_v1:latest | 0/8 (1.05) | 0/8 (1.07) | 7/8 (0.13) | 0/8 (1.03) | 0/8 (0.98) | 0/8 (0.98) | 7/48 |

**Late-training degeneration:** after ~45M steps the policy learned to *not start* a left strafe from
standing (4/8 failed starts at 50M, 7/8 at 60M); all other commands start fine and the 100-episode
average hides it (pure lateral commands from rest are rare under independent uniform sampling of
vx, vy, wz). Energy pressure + a rare situation = a local "stand" optimum (the Gurdy 1.0 trap,
narrowly). 30M and 40M start every command from rest (0/48 failures). **40M is selected**: CoT 7.70,
window tracking error 10.9 % (user target < 10 % narrowly missed; 50M/60M meet it only by being
unreliable at starting). Fix proposed for a follow-up (2.7b): axis-aligned and start-from-rest commands
in the training distribution, start_test as a regression gate.

## Robustness: DR extreme scenarios (fall rate / window tracking error / mean power)
50 episodes per extreme (100 nominal), deterministic, one parameter fixed at its extreme, rest nominal;
`all_extremes` = all hard settings at once (friction 0.5, heavy, weak motors, damping +20 %, joint
friction 0.04, latency 10 ms, sensor noise, pushes).

| scenario | tripod | flat_bc_v1:latest | dr_energy_v1:40000000 | dr_energy_v1:latest |
|---|---|---|---|---|
| nominal (n=100) | 0 % / 15.9 % / 13.2 W | 0 % / 4.9 % / 39.2 W | 0 % / 10.9 % / 17.3 W | 0 % / 9.2 % / 16.2 W |
| friction_low (n=50) | 0 % / 14.4 % / 13.6 W | 0 % / 9.6 % / 40.0 W | 0 % / 11.1 % / 17.5 W | 0 % / 8.9 % / 16.3 W |
| friction_high (n=50) | 0 % / 16.7 % / 13.6 W | 0 % / 4.9 % / 40.0 W | 0 % / 11.3 % / 17.8 W | 0 % / 9.4 % / 16.4 W |
| heavy (n=50) | 0 % / 15.4 % / 13.7 W | 0 % / 4.9 % / 40.2 W | 0 % / 10.6 % / 17.8 W | 0 % / 8.8 % / 16.6 W |
| weak_motors (n=50) | 0 % / 14.8 % / 13.1 W | 0 % / 5.5 % / 34.6 W | 0 % / 11.1 % / 16.6 W | 0 % / 9.2 % / 15.4 W |
| latency_10ms (n=50) | 0 % / 12.2 % / 13.6 W | 0 % / 8.8 % / 38.4 W | 0 % / 10.0 % / 17.1 W | 0 % / 8.1 % / 15.4 W |
| sensor_noise (n=50) | 0 % / 16.2 % / 13.6 W | 0 % / 99.2 % / 34.6 W | 0 % / 11.1 % / 17.8 W | 0 % / 9.0 % / 16.7 W |
| pushes (n=50) | 0 % / 16.1 % / 14.1 W | 0 % / 5.3 % / 42.2 W | 0 % / 11.7 % / 18.8 W | 0 % / 9.4 % / 17.5 W |
| all_extremes (n=50) | 0 % / 13.8 % / 14.9 W | 0 % / 35.1 % / 33.3 W | 0 % / 12.2 % / 17.3 W | 0 % / 10.4 % / 15.9 W |

* **Fall rate 0 % for every controller in every scenario** (AC: < 5 %).
* `flat_bc_v1` is accurate only in a clean world: **sensor noise breaks it (99.2 % error)** and all
  extremes give 35.1 %. The DR policy stays at **10.0–12.2 %** in every scenario.
* The open-loop tripod is unaffected by sensor noise but cannot correct anything (12–17 %).

## Smoothness and energy (nominal protocol, 500 Hz analysis, `rl/tools/smoothness.py`)

| metric | tripod | flat_bc_v1 | dr_energy_v1 40M |
|---|---|---|---|
| protocol energy / CoT | 1614 J / 7.35 | 4602 J / 18.29 | **1963 J / 7.70** |
| energy, moving segments | 1020 J | 2950 J | 1250 J |
| energy NOT explained by < 5 Hz motion | 5.6 % | 7.4 % | 6.7 % |
| hip_yaw velocity power above 5 Hz | 1.9 % | 14.9 % | 6.4 % |
| action rate Σ(Δa)² | 0.035 | 0.113 | 0.046 |
| action 2nd difference | 0.0026 | 0.0225 | 0.0047 |
| joint speed RMS | 1.30 rad/s | 2.19 rad/s | 1.40 rad/s |
| stepping frequency (forward) | 2.50 Hz | 2.80 Hz | 2.10 Hz |

flat_bc_v1's extra energy came mostly from larger/faster gait motion, with hip_yaw jitter on top; the
curriculum removed most of both: **2.6× less energy than flat_bc_v1, CoT within 5 % of the tripod**.
The DR policy now steps slower than the tripod (2.1 vs 2.5 Hz); some hip_yaw jitter remains (6.4 %).

## Summary vs step 2.7 targets
| target | result (40M) |
|---|---|
| ROADMAP AC: flat AC still met with DR (tracking < 20 %, falls < 5 %) | met: 10.9 % nominal, 0 falls |
| ROADMAP AC: fall rate < 5 % at DR extremes | met: 0 % in all 9 scenarios |
| CoT ≤ 1.5× tripod (≤ ~11) | met: 7.70 (tripod 7.35) |
| mean tracking error < 10 % | **narrowly missed: 10.9 %** (flat_bc_v1 4.9 %, tripod 15.9 %) |
| jitter check | done: jitter was a minor energy share; hip_yaw HF 14.9 → 6.4 % |
| start-from-rest (added) | 40M: 0/48 failures; 50M/60M fail left-strafe starts |
