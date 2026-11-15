# Phase 2 step 2.7 — Domain randomisation, asymmetric critic, energy/smoothness curriculum

Policy **`dr_energy_v1`, BC-initialised from the tripod gait** (via `flat_bc_v1`), then PPO with domain
randomisation (DR), an asymmetric critic and an energy/smoothness curriculum. MuJoCo, measured
2026-11-15 in the `rl` container. Raw data: `phase2_dr_checkpoints.json`, `phase2_dr_scenarios.json`,
`phase2_dr_smoothness.json`. Video (tripod | flat_bc_v1 | dr_energy_v1 40M):
[`docs/videos/phase2_3way_tripod_flat_dr.mp4`](../videos/phase2_3way_tripod_flat_dr.mp4).

**Selected in 2.7: 40M steps** (not the final one — see "late-training degeneration" below). **Superseded by 2.7b: final selection `dr_energy_ft` 6M** (fine-tuned from 40M; selected 2026-11-15 after the corrected evaluation — see the end of this file).

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
| stride frequency, forward (touchdowns/leg/s)¹ | 1.23 | 1.40 | 1.08 |

flat_bc_v1's extra energy came mostly from larger/faster gait motion, with hip_yaw jitter on top; the
curriculum removed most of both: **2.6× less energy than flat_bc_v1, CoT within 5 % of the tripod**.
The DR policy strides slower than the tripod (1.08 vs 1.23 touchdowns/leg/s); some hip_yaw jitter
remains (6.4 %).

¹ Corrected 2026-11-15: earlier versions reported the strongest hip_pitch-velocity spectral peak
(2.50 / 2.80 / 2.10 Hz), which picks the stride's 2nd harmonic for some gaits and the fundamental for
others; touchdown counting from foot contacts is unambiguous (ratios between policies were unchanged).

## Summary vs step 2.7 targets
| target | result (40M) |
|---|---|
| ROADMAP AC: flat AC still met with DR (tracking < 20 %, falls < 5 %) | met: 10.9 % nominal, 0 falls |
| ROADMAP AC: fall rate < 5 % at DR extremes | met: 0 % in all 9 scenarios |
| CoT ≤ 1.5× tripod (≤ ~11) | met: 7.70 (tripod 7.35) |
| mean tracking error < 10 % | **narrowly missed: 10.9 %** (flat_bc_v1 4.9 %, tripod 15.9 %) |
| jitter check | done: jitter was a minor energy share; hip_yaw HF 14.9 → 6.4 % |
| start-from-rest (added) | 40M: 0/48 failures; 50M/60M fail left-strafe starts |
| stride frequency | 1.08 touchdowns/leg/s (tripod 1.23) |

---

# Step 2.7b — start-from-rest gate + fine-tune (`dr_energy_ft`)

**Selected policy: `dr_energy_ft` 10M** — fine-tuned from `dr_energy_v1` 40M (BC-initialised from the
tripod gait). Raw data: `phase2_dr_ft_checkpoints.json`, `phase2_dr_ft_scenarios.json`,
`phase2_dr_ft_smoothness.json`.

**Changes.**
* Training commands (`rl/configs/dr_energy_ft.yaml`): 25 % axis-aligned commands (pure ±vx/±vy/±wz,
  30–100 % of the limit) and 25 % of command switches preceded by 1–2 s of zero command (start from rest).
  Energy/smoothness weights fixed at their final values, DR at full scale, no critic warm-up.
  10,014,720 steps in **0.96 h** (2.7–3.0k samples/s), 1.6 GiB, no falls.
* **Start gate** (now in every `rl/eval.py` run): ±vx 0.10, ±vy 0.08, ±wz 0.3 from rest at 8 gait-clock
  phases (48 starts); fail if < 20 % of the commanded speed; a checkpoint qualifies only at 0/48.
* `rl/eval.py` reports the standard error of the tracking error across episodes.

## Checkpoint selection (nominal, deterministic, same eval run)

| controller | forward | backward | strafe_left | strafe_right | turn_left | turn_right | combined | falls | energy J | CoT | mean power W | start gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod | -1.5 % | -1.5 % | +4.3 % | +4.1 % | -8.6 % | -8.5 % | -8.5 % | 0 | 1614 | 7.35 | 13.1 | PASS 0/48 |
| dr_energy_v1:40000000 | +3.0 % | +4.1 % | +9.2 % | +5.0 % | -1.7 % | -2.5 % | +5.2 % | 0 | 1963 | 7.70 | 16.0 | PASS 0/48 |
| dr_energy_ft:2000000 | +6.7 % | +4.2 % | +13.5 % | +6.1 % | -1.0 % | -5.1 % | +2.4 % | 0 | 1962 | 7.48 | 16.0 | PASS 0/48 |
| dr_energy_ft:4000000 | +8.5 % | +4.6 % | +8.9 % | +0.5 % | +0.3 % | -3.5 % | +3.8 % | 0 | 1903 | 7.34 | 15.5 | PASS 0/48 |
| dr_energy_ft:6000000 | +9.1 % | +6.8 % | +8.0 % | +2.4 % | +2.4 % | +4.7 % | +7.2 % | 0 | 1909 | 7.34 | 15.5 | PASS 0/48 |
| dr_energy_ft:8000000 | +4.4 % | +4.5 % | +8.0 % | +6.6 % | +2.1 % | -2.3 % | +5.8 % | 0 | 1835 | 7.13 | 14.9 | PASS 0/48 |
| dr_energy_ft:10000000 | +2.4 % | +3.5 % | +10.6 % | -0.1 % | +1.8 % | +0.1 % | +6.0 % | 0 | 1776 | 6.93 | 14.4 | PASS 0/48 |


| checkpoint | window tracking error ± SE (100 episodes) | mean power |
|---|---|---|
| tripod | 15.9 ± 0.8 % | 13.2 W |
| dr_energy_v1 40M | 10.9 ± 1.0 % | 17.3 W |
| dr_energy_ft 6M | 10.2 ± 1.2 % | 15.9 W |
| dr_energy_ft 8M | 11.0 ± 1.3 % | 15.5 W |
| **dr_energy_ft 10M** | **10.9 ± 1.0 %** | **15.2 W** |

Every fine-tune checkpoint passes the gate (0/48; mean start speed 0.97–1.09 of the command). Tracking is
statistically the same for all (SE ±1.0–1.3 points), so **10M is selected for the lowest energy**
(CoT 6.93). **The < 10 % tracking target is not met** (10.9 ± 1.0 %; best point estimate 10.2 % at 6M).

## DR scenarios (fall rate / window error ± SE / mean power) with start gate

> **Superseded — evaluation bug.** In this first 2.7b evaluation `dr_energy_ft` was graded on its own
> training command distribution (25 % axis-aligned + 25 % rest starts) while the other controllers used
> the standard one. Corrected numbers: section "2.7b corrected evaluation" below.

| controller | forward | backward | strafe_left | strafe_right | turn_left | turn_right | combined | falls | energy J | CoT | mean power W | start gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod | -1.5 % | -1.5 % | +4.3 % | +4.1 % | -8.6 % | -8.5 % | -8.5 % | 0 | 1614 | 7.35 | 13.1 | PASS 0/48 |
| flat_bc_v1:latest | +2.9 % | +1.7 % | +1.9 % | +8.8 % | +4.4 % | +3.2 % | +8.0 % | 0 | 4602 | 18.29 | 37.4 | PASS 0/48 |
| dr_energy_v1:40000000 | +3.0 % | +4.1 % | +9.2 % | +5.0 % | -1.7 % | -2.5 % | +5.2 % | 0 | 1963 | 7.70 | 16.0 | PASS 0/48 |
| dr_energy_ft:10000000 | +2.4 % | +3.5 % | +10.6 % | -0.1 % | +1.8 % | +0.1 % | +6.0 % | 0 | 1776 | 6.93 | 14.4 | PASS 0/48 |


| scenario | tripod | flat_bc_v1:latest | dr_energy_v1:40000000 | dr_energy_ft:10000000 |
|---|---|---|---|---|
| nominal (n=100) | 0 % / 15.9 ± 0.8 % / 13.2 W | 0 % / 4.9 ± 0.2 % / 39.2 W | 0 % / 10.9 ± 1.0 % / 17.3 W | 0 % / 10.9 ± 1.0 % / 15.2 W |
| friction_low (n=50) | 0 % / 14.4 ± 1.2 % / 13.6 W | 0 % / 9.6 ± 0.5 % / 40.0 W | 0 % / 11.1 ± 1.5 % / 17.5 W | 0 % / 10.7 ± 1.6 % / 15.2 W |
| friction_high (n=50) | 0 % / 16.7 ± 1.2 % / 13.6 W | 0 % / 4.9 ± 0.2 % / 40.0 W | 0 % / 11.3 ± 1.5 % / 17.8 W | 0 % / 12.0 ± 1.5 % / 15.5 W |
| heavy (n=50) | 0 % / 15.4 ± 1.2 % / 13.7 W | 0 % / 4.9 ± 0.2 % / 40.2 W | 0 % / 10.6 ± 1.5 % / 17.8 W | 0 % / 10.0 ± 1.3 % / 15.6 W |
| weak_motors (n=50) | 0 % / 14.8 ± 1.2 % / 13.1 W | 0 % / 5.5 ± 0.2 % / 34.6 W | 0 % / 11.1 ± 1.5 % / 16.6 W | 0 % / 11.6 ± 1.4 % / 14.5 W |
| latency_10ms (n=50) | 0 % / 12.2 ± 1.0 % / 13.6 W | 0 % / 8.8 ± 0.3 % / 38.4 W | 0 % / 10.0 ± 1.1 % / 17.1 W | 0 % / 10.5 ± 1.3 % / 14.7 W |
| sensor_noise (n=50) | 0 % / 16.2 ± 1.1 % / 13.6 W | 0 % / 99.2 ± 3.2 % / 34.6 W | 0 % / 11.1 ± 1.0 % / 17.8 W | 0 % / 11.3 ± 1.5 % / 14.8 W |
| pushes (n=50) | 0 % / 16.1 ± 1.1 % / 14.1 W | 0 % / 5.3 ± 0.3 % / 42.2 W | 0 % / 11.7 ± 1.3 % / 18.8 W | 0 % / 10.3 ± 1.1 % / 14.9 W |
| all_extremes (n=50) | 0 % / 13.8 ± 0.9 % / 14.9 W | 0 % / 35.1 ± 0.6 % / 33.3 W | 0 % / 12.2 ± 0.9 % / 17.3 W | 0 % / 11.8 ± 0.8 % / 15.2 W |

## Smoothness vs 40M (nominal protocol, 500 Hz)

| metric | tripod | dr_energy_v1 40M | **dr_energy_ft 10M** |
|---|---|---|---|
| protocol energy / CoT | 1614 J / 7.35 | 1963 J / 7.70 | **1776 J / 6.93** |
| energy, moving segments | 1020 J | 1250 J | 1110 J |
| energy NOT explained by < 5 Hz motion | 5.6 % | 6.7 % | 6.6 % |
| hip_yaw velocity power above 5 Hz | 1.9 % | 6.4 % | 6.2 % |
| action rate / 2nd difference | 0.035 / 0.0026 | 0.046 / 0.0047 | 0.043 / 0.0043 |
| joint speed RMS | 1.30 rad/s | 1.40 rad/s | 1.33 rad/s |
| stride frequency (touchdowns/leg/s) | 1.23 | 1.08 | 1.23 |

## Result vs 2.7b targets
| target | dr_energy_ft 10M |
|---|---|
| start gate 0/48 (all directions incl. strafe and turns) | **met** (also met by every fine-tune checkpoint) |
| tracking < 10 % | ~~not met: 10.9 ± 1.0 %~~ → **corrected: 9.6 ± 0.4 % (met nominally)**, see below |
| CoT ≤ 11 | **met: 6.93** (below the tripod's 7.35) |
| falls < 5 % in all 9 DR scenarios | **met: 0 %** in all |
| smoothness vs 40M | equal or slightly better on every metric; stride frequency now equals the tripod's |

The comparison video was not re-rendered: the selected policy's gait is visually the same as 40M's (same
smoothness numbers within a few %, 9 % less energy, 14 % higher stride frequency).

---

# 2.7b corrected evaluation, overshoot and dead-zone diagnosis (2026-11-15)

## Evaluation bug (fixed, no retraining)
`rl/eval.py` built each controller's environment from that run's training config. `dr_energy_ft`'s config
contains the 2.7b command sampling (25 % axis-aligned, 25 % rest starts), so it was graded on a different,
harder command mix than the other controllers. `eval.py` now always uses the standard command
distribution (`axis_aligned_prob = rest_start_prob = 0`). The fixed-command protocol, start gate and
DR scenario physics were not affected; 40M / flat_bc_v1 / tripod numbers are unchanged.

## Corrected comparison — same standard command distribution for every controller

| controller | forward | backward | strafe_left | strafe_right | turn_left | turn_right | combined | falls | energy J | CoT | mean power W | start gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod | -1.5 % | -1.5 % | +4.3 % | +4.1 % | -8.6 % | -8.5 % | -8.5 % | 0 | 1614 | 7.35 | 13.1 | PASS 0/48 |
| flat_bc_v1:latest | +2.9 % | +1.7 % | +1.9 % | +8.8 % | +4.4 % | +3.2 % | +8.0 % | 0 | 4602 | 18.29 | 37.4 | PASS 0/48 |
| dr_energy_v1:40000000 | +3.0 % | +4.1 % | +9.2 % | +5.0 % | -1.7 % | -2.5 % | +5.2 % | 0 | 1963 | 7.70 | 16.0 | PASS 0/48 |
| dr_energy_ft:6000000 | +9.1 % | +6.8 % | +8.0 % | +2.4 % | +2.4 % | +4.7 % | +7.2 % | 0 | 1909 | 7.34 | 15.5 | PASS 0/48 |
| dr_energy_ft:8000000 | +4.4 % | +4.5 % | +8.0 % | +6.6 % | +2.1 % | -2.3 % | +5.8 % | 0 | 1835 | 7.13 | 14.9 | PASS 0/48 |
| dr_energy_ft:10000000 | +2.4 % | +3.5 % | +10.6 % | -0.1 % | +1.8 % | +0.1 % | +6.0 % | 0 | 1776 | 6.93 | 14.4 | PASS 0/48 |


| scenario | tripod | flat_bc_v1:latest | dr_energy_v1:40000000 | dr_energy_ft:6000000 | dr_energy_ft:8000000 | dr_energy_ft:10000000 |
|---|---|---|---|---|---|---|
| nominal (n=100) | 0 % / 15.9 ± 0.8 % / 13.2 W | 0 % / 4.9 ± 0.2 % / 39.2 W | 0 % / 10.9 ± 1.0 % / 17.3 W | 0 % / 9.2 ± 0.6 % / 16.8 W | 0 % / 10.4 ± 0.6 % / 16.5 W | 0 % / 9.6 ± 0.4 % / 16.2 W |
| friction_low (n=50) | 0 % / 14.4 ± 1.2 % / 13.6 W | 0 % / 9.6 ± 0.5 % / 40.0 W | 0 % / 11.1 ± 1.5 % / 17.5 W | 0 % / 9.2 ± 0.7 % / 17.0 W | 0 % / 11.0 ± 1.1 % / 16.6 W | 0 % / 9.3 ± 0.6 % / 16.3 W |
| friction_high (n=50) | 0 % / 16.7 ± 1.2 % / 13.6 W | 0 % / 4.9 ± 0.2 % / 40.0 W | 0 % / 11.3 ± 1.5 % / 17.8 W | 0 % / 9.4 ± 0.9 % / 17.2 W | 0 % / 10.8 ± 0.9 % / 16.8 W | 0 % / 10.8 ± 0.7 % / 16.5 W |
| heavy (n=50) | 0 % / 15.4 ± 1.2 % / 13.7 W | 0 % / 4.9 ± 0.2 % / 40.2 W | 0 % / 10.6 ± 1.5 % / 17.8 W | 0 % / 9.1 ± 0.7 % / 17.3 W | 0 % / 10.5 ± 0.9 % / 16.9 W | 0 % / 9.6 ± 0.6 % / 16.6 W |
| weak_motors (n=50) | 0 % / 14.8 ± 1.2 % / 13.1 W | 0 % / 5.5 ± 0.2 % / 34.6 W | 0 % / 11.1 ± 1.5 % / 16.6 W | 0 % / 9.4 ± 0.8 % / 16.2 W | 0 % / 10.8 ± 0.9 % / 15.8 W | 0 % / 10.2 ± 0.7 % / 15.5 W |
| latency_10ms (n=50) | 0 % / 12.2 ± 1.0 % / 13.6 W | 0 % / 8.8 ± 0.3 % / 38.4 W | 0 % / 10.0 ± 1.1 % / 17.1 W | 0 % / 9.0 ± 0.8 % / 16.4 W | 0 % / 10.4 ± 0.9 % / 16.0 W | 0 % / 9.6 ± 0.6 % / 15.6 W |
| sensor_noise (n=50) | 0 % / 16.2 ± 1.1 % / 13.6 W | 0 % / 99.2 ± 3.2 % / 34.6 W | 0 % / 11.1 ± 1.0 % / 17.8 W | 0 % / 9.7 ± 0.7 % / 17.3 W | 0 % / 10.7 ± 0.9 % / 17.0 W | 0 % / 10.4 ± 0.7 % / 16.6 W |
| pushes (n=50) | 0 % / 16.1 ± 1.1 % / 14.1 W | 0 % / 5.3 ± 0.3 % / 42.2 W | 0 % / 11.7 ± 1.3 % / 18.8 W | 0 % / 10.0 ± 1.1 % / 18.3 W | 0 % / 11.3 ± 1.1 % / 17.8 W | 0 % / 10.1 ± 1.0 % / 17.6 W |
| all_extremes (n=50) | 0 % / 13.8 ± 0.9 % / 14.9 W | 0 % / 35.1 ± 0.6 % / 33.3 W | 0 % / 12.2 ± 0.9 % / 17.3 W | 0 % / 10.3 ± 0.5 % / 16.7 W | 0 % / 11.7 ± 0.7 % / 16.3 W | 0 % / 11.5 ± 0.6 % / 16.0 W |

* **ft-10M meets < 10 % nominally: 9.6 ± 0.4 %** (below 10 % in 4/9 DR scenarios; 9.3–11.5 % overall),
  CoT 6.93, 0 falls, start gate 0/48.
* **ft-6M tracks better in all 9 scenarios** (9.2 ± 0.6 % nominal, 9.0–10.3 % overall, < 10 % in 8/9),
  CoT 7.34, ~4 % more power.
* **Final selection (user, 2026-11-15): `dr_energy_ft` 6M** — better tracking under DR in all 9 scenarios
  (9.0–10.3 %, < 10 % in 8/9), CoT 7.34 (≤ 11 and at the tripod's 7.35), 0 falls, start gate 0/48.
  Checkpoint: `rl/runs/dr_energy_ft/checkpoints/ppo_6000000_steps.zip`. Known limitation: small-command
  dead zone (below), fix planned in 2.8.

## Overshoot diagnosis — signed error per axis and speed (`rl/tools/tracking_bias.py`)
Single-axis commands, 12 s from rest, mean over the last 6 s, deterministic, nominal physics.
**Reward and eval compute velocity identically**: the reward's 0.2 s EMA and the eval's body-frame mean
differ by ≤ 0.0014 (m/s or rad/s) for every command; both use the command active during the step and the
velocity at its end. The Phase 1 *protocol* uses a heading (yaw-only) frame from world poses; it differs by
≤ 0.003 m/s and ≤ 0.024 rad/s (on a wobbling body the z-gyro rate ≠ heading-change rate) — a different,
valid definition, not a bug.

Linear fit measured = slope × command + offset (body frame):

| controller | vx slope / offset | vy slope / offset | wz slope / offset |
|---|---|---|---|
| tripod | 0.965 / +0.0000 m/s | 1.010 / +0.0001 m/s | 0.878 / −0.0000 rad/s |
| flat_bc_v1 | 0.984 / +0.0006 | 1.006 / −0.0030 | 0.969 / −0.0004 |
| dr_energy_v1 40M | 0.900 / −0.0071 | 0.898 / −0.0052 | 0.861 / −0.0028 |
| dr_energy_ft 10M | 0.942 / −0.0001 | 0.952 / +0.0030 | 0.869 / −0.0027 |

(40M/ft fits include the dead-zone points, which pull the slope down.) **There is no general overshoot**:
on single-axis commands the policies slightly *undershoot* at the extremes (turning ~13 % at 0.6 rad/s,
like the tripod's 12 %) and ft-10M has a small leftward drift (+3 mm/s on vy: strafe left +8 %, strafe right
−3 %). The "+2…+10 %" seen on protocol segments comes from that lateral offset plus the heading-frame
definition. The real tracking problem is a dead zone at small commands.

## Dead zone at small commands
Commands that give < 20 % of the commanded speed (F = from rest, W = after 4 s walking at the 3rd speed):

| controller | vx | vy | wz |
|---|---|---|---|
| tripod | none | none | none |
| flat_bc_v1 | none (+0.03 W: 0.71 of command) | −0.03 W | none |
| dr_energy_v1 40M | ±0.03 F and W; +0.06 F | ±0.03 F and W; +0.05 F | ±0.10 F and W |
| dr_energy_ft 10M | ±0.03 F and W | ±0.03 F and W; −0.05 W | ±0.10 F and W |

(tested: vx 0.03/0.06/0.10/0.15, vy 0.03/0.05/0.08/0.10, wz 0.10/0.20/0.30/0.60, both signs.) ft-10M
follows vx ≥ 0.045, vy ≥ 0.05 (+), wz ≥ 0.15. The energy curriculum widened a dead zone that the flat
policy barely had. In the random-command windows the effect is small (components < 20 % of the limit:
ft-10M 9.3 % vs 8.8–11.7 % for larger ones, `rl/tools/window_error_breakdown.py`) because tiny components
usually come with a larger one; it matters for pure slow commands (e.g. fine Nav2 corrections).

**Is standing the reward-optimal answer?** Per-step reward with the final weights
(`rl/tools/deadzone_reward.py`, seconds 4–10 from rest):

| command | stand: reward/step (track) | tripod: reward/step (track, energy+smooth) | policy: reward/step (speed ratio) |
|---|---|---|---|
| (0.03, 0, 0) | 0.206 (0.158) | 1.384 (1.461, -0.139) | 0.203 (0.00) |
| (0.045, 0, 0) | 0.075 (0.027) | 1.360 (1.442, -0.143) | 1.162 (0.95) |
| (0.06, 0, 0) | 0.075 (0.027) | 1.344 (1.432, -0.149) | 1.239 (1.02) |
| (0, 0.03, 0) | 0.206 (0.158) | 1.398 (1.477, -0.140) | 0.200 (0.00) |
| (0, 0.05, 0) | 0.075 (0.027) | 1.389 (1.473, -0.146) | 1.195 (1.11) |
| (0, 0, 0.1) | 0.075 (0.027) | 1.286 (1.364, -0.139) | 0.070 (0.00) |
| (0, 0, 0.15) | 0.075 (0.027) | 1.280 (1.360, -0.141) | 1.189 (0.91) |
| (0, 0, 0.2) | 0.075 (0.027) | 1.275 (1.358, -0.145) | 1.256 (0.96) |

**No.** At 0.03 m/s the σ floor (0.02 m/s) raises standing's tracking term to 0.158, but walking earns ~7×
more per step even after −0.14 energy + smoothness cost; at wz 0.10 the floor is not even active
(σ = κ·0.10 = 0.05) and standing earns only 0.027 tracking — the policy still stands. **The dead zone is a
learned local optimum, not the reward optimum.** Likely causes: (1) sampling — pure small commands are
rare (under independent uniform sampling only ~2.7 % of non-zero commands have every component < 30 % of
its limit, vs 15 % exact-zero commands that train "stand still"), so "stand" generalises from zero to small
commands; (2) the energy curriculum made the stand basin deep (flat_bc_v1 barely had it); (3) low
exploration noise (std 0.035) cannot discover a slow gait starting from a standing pose.

**Proposed fix (2.8, no retraining now):**
1. Sampling: ~15 % of commands small (one or more components at 15–40 % of the limit, axis-aligned and
   mixed); zero-command share 15 % → 10 %.
2. Gates: extend the start gate with small commands (vx 0.03, vy 0.03, wz 0.10, from rest and while
   walking) — 0 failures required.
3. Explicit command deadband in the `/cmd_vel` interface (e.g. |v| < 0.015 m/s, |ω| < 0.05 rad/s → 0) so
   "too small to walk" is a specified behaviour, not a learned one; train and gate above it.
4. Keep the σ floors (measured: not the cause); optionally a slightly larger κ for small commands is
   unnecessary given the numbers above.
