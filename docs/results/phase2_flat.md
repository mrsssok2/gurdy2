# Phase 2 steps 2.4–2.5 — Flat-ground locomotion policy (MuJoCo)

**Policy `flat_bc_v1`, BC-initialised from the tripod gait**, then PPO fine-tuned on flat ground
(no domain randomisation). Measured 2026-11-14/15 in the `rl` container. Raw evaluation:
`phase2_flat_eval.json`. Video (45 s excerpt, tripod left / policy right):
[`docs/videos/phase2_flat_tripod_vs_policy.mp4`](../videos/phase2_flat_tripod_vs_policy.mp4).

## How we got here (2M-step smoke runs, reward per step; standing = 0.385, tripod ≈ 1.3)

| Run | Change | Reward/step at 2M | Outcome |
|---|---|---|---|
| smoke | first version (Gaussian std 0.35) | ≈ −1.2 | noise-driven penalties; falling early beat surviving; nothing learned |
| smoke_v1 | per-step reward clipped ≥ 0, separate fall term; penalty curriculum from 0.1× | 0.039 | critic learns (expl. var. +0.44); far below standing |
| A | gSDE (resampled every 80 ms), lr 1e-4, penalties from 0.02× | 0.065 | flat |
| B | Gaussian std 0.1, lr 1e-4, penalties from 0.02× | 0.083 | slowly rising |
| C1 | B + action EMA α 0.5 (in `gurdy_core.rl_action`) | 0.182 | **best from scratch** (kept for Experiment B) |
| C2 | C1 + BC warm start from the tripod (+ critic warm-up) | 1.10 | walks from the start; chosen |

Diagnosis: white exploration noise every 20 ms makes the body jitter; with the deliberately sharp
tracking term (σ = κ·|cmd|) even std 0.1 tracked worse than standing (0.073 vs 0.338). The action
EMA tripled the noisy policy's reward without changing the tripod's; BC removed the need to discover
walking from noise. **Results below are for a policy BC-initialised from the tripod gait.**

## Training run `flat_bc_v1` (step 2.4)

Config `rl/configs/flat_bc_long.yaml`: BC init (1M tripod samples, val MSE 0.0002 vs label variance
0.072; deterministic BC policy 1.18 reward/step vs tripod 1.34), critic warm-up 0.2M steps, PPO lr 5e-5,
target KL 0.015, std 0.05 → 0.04, penalty curriculum 0.02 → 1.0 (0–15M), κ 0.8 → 0.5 (10–40M),
40 envs (10 × 4). **60,006,400 steps in 3.84 h** (3.0–4.7k samples/s), RAM (process tree) 1.58–1.64 GiB,
container ≤ 6 GiB limit. Resume verified: a copy resumed from the 58M checkpoint continued at 58,000,000
steps in the same TensorBoard curve (return 1223 vs ~1220 before).

| metric | 0.04M | 7.54M | 15.03M | 22.53M | 30.02M | 37.52M | 45.02M | 52.51M | 60.01M |
|---|---|---|---|---|---|---|---|---|---|
| return | 1.13e+03 | 1.23e+03 | 1.19e+03 | 1.19e+03 | 1.21e+03 | 1.2e+03 | 1.21e+03 | 1.22e+03 | 1.22e+03 |
| fall | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| trk_err | 0.319 | 0.237 | 0.228 | 0.172 | 0.172 | 0.192 | 0.135 | 0.162 | 0.148 |
| r_track | 1.08 | 1.21 | 1.23 | 1.27 | 1.26 | 1.24 | 1.27 | 1.24 | 1.29 |
| r_slip | -0.000213 | -0.00909 | -0.0161 | -0.0151 | -0.0154 | -0.0172 | -0.0168 | -0.0149 | -0.0166 |
| r_energy | -0.000333 | -0.0174 | -0.0404 | -0.0446 | -0.0494 | -0.0523 | -0.0525 | -0.0465 | -0.0514 |
| eff_std | 0.05 | 0.0451 | 0.0428 | 0.0418 | 0.0412 | 0.0406 | 0.0405 | 0.0404 | 0.0406 |
| kl | 0 | 0.0306 | 0.0313 | 0.0374 | 0.037 | 0.0384 | 0.0424 | 0.039 | 0.037 |
| clipfrac | 0 | 0.32 | 0.362 | 0.364 | 0.38 | 0.399 | 0.401 | 0.382 | 0.363 |
| expl_var | -0.859 | 0.533 | 0.639 | 0.673 | 0.633 | 0.627 | 0.586 | 0.569 | 0.57 |
| value_loss | 0.786 | 0.0235 | 0.017 | 0.0283 | 0.0161 | 0.0147 | 0.0251 | 0.0213 | 0.0237 |
| pen_scale | 0.0213 | 0.511 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| clipped | 0.00742 | 0.0123 | 0.0282 | 0.0225 | 0.0318 | 0.0361 | 0.0341 | 0.0514 | 0.0326 |
| fps | 4.68e+03 | 2.98e+03 | 3.11e+03 | 3.5e+03 | 3.77e+03 | 3.97e+03 | 4.12e+03 | 4.25e+03 | 4.35e+03 |
| ram_MiB | 1.58e+03 | 1.62e+03 | 1.62e+03 | 1.62e+03 | 1.61e+03 | 1.58e+03 | 1.64e+03 | 1.59e+03 | 1.63e+03 |
| kappa | 0.8 | 0.8 | 0.75 | 0.675 | 0.6 | 0.525 | 0.5 | 0.5 | 0.5 |

(Training metrics use the stochastic policy; `trk_err` = mean relative error of the 0.2 s-filtered body
velocity vs command over moving steps; penalties grow with `pen_scale` up to 15M.)

Observations: no falls in 60M steps; tracking error 0.32 → 0.15 while κ tightened 0.8 → 0.5 (tracking
reward still rose 1.08 → 1.29); explained variance −0.86 → 0.57–0.67. KL (0.03–0.04) and clip fraction
(0.36–0.40) stayed high even at lr 5e-5 — with std 0.04 small mean shifts give large probability ratios;
most updates were stopped early by target KL. Energy rose throughout (r_energy −0.017 → −0.05).

## Deterministic evaluation (step 2.5)

`rl/eval.py`: (1) the **Phase 1 tracking protocol** (123 s, same code as Phase 1) — per-segment main-axis
error, falls, mechanical energy Σ|τ·q̇| integrated every 2 ms, cost of transport CoT = E/(m g d);
(2) **100 episodes × 20 s** with the training command distribution (seeds 0–99) — fall rate and
per-5 s-window tracking error ‖(v̄ − c)/lim‖ / ‖c/lim‖ (v̄ = mean body velocity over seconds 1–5).
Deterministic policy, checkpoint's VecNormalize statistics. 166 s for 7 controllers.

| controller | forward | backward | strafe_left | strafe_right | turn_left | turn_right | combined | falls (protocol) | energy J | CoT | 100-ep fall rate | window err mean / median / p90 | windows < 20 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tripod | -1.5 % | -1.5 % | +4.3 % | +4.1 % | -8.6 % | -8.5 % | -8.5 % | 0 | 1614 | 7.35 | 0 % | 16.1 / 11.1 / 36.6 % | 65 % |
| ppo_10000000_steps | +15.1 % | +7.3 % | +18.9 % | +10.8 % | +9.8 % | +8.5 % | +9.7 % | 0 | 3895 | 14.25 | 0 % | 13.1 / 10.3 / 19.4 % | 91 % |
| ppo_20000000_steps | +3.8 % | +8.3 % | +16.5 % | +14.1 % | +7.6 % | +10.8 % | +8.4 % | 0 | 4582 | 16.96 | 0 % | 10.5 / 8.9 / 16.4 % | 96 % |
| ppo_30000000_steps | +7.4 % | +4.9 % | +11.1 % | +9.4 % | +5.7 % | +9.3 % | +11.3 % | 0 | 4744 | 18.33 | 0 % | 8.0 / 5.9 / 12.7 % | 98 % |
| ppo_40000000_steps | +5.2 % | +3.4 % | +4.0 % | +5.2 % | +6.9 % | +6.3 % | +10.0 % | 0 | 4757 | 18.42 | 0 % | 6.6 / 4.6 / 9.9 % | 99 % |
| ppo_50000000_steps | +2.8 % | +7.6 % | +5.1 % | +6.4 % | +7.3 % | +0.5 % | +4.3 % | 0 | 4813 | 18.81 | 0 % | 5.6 / 4.6 / 8.8 % | 99 % |
| ppo_latest | +2.9 % | +1.7 % | +1.9 % | +8.8 % | +4.4 % | +3.2 % | +8.0 % | 0 | 4602 | 18.29 | 0 % | 5.7 / 4.7 / 8.8 % | 99 % |

## Result vs step 2.4 acceptance criteria — met
* Flat-ground tracking error **< 20 %**: final policy 5.7 % mean / 4.7 % median / 8.8 % p90 over 100
  episodes (99 % of command windows < 20 %); Phase 1 protocol segments +1.7 … +8.8 %.
* Fall rate **< 5 %**: 0 % (0/100 episodes, 0 in the protocol).
* Resume verified; RAM peak (1.64 GiB) and wall time (3.84 h) recorded.

**Selected checkpoint: `ppo_latest` (60M)** — tracking equal to 50M (5.7 vs 5.6 %), 4 % less energy.

## Policy vs tripod gait (flat ground, MuJoCo)
| | Tripod gait | Policy (BC-init + PPO, 60M) |
|---|---|---|
| Protocol main-axis error (range) | −8.6 … +4.3 % | +1.7 … +8.8 % |
| 100-episode window error mean / p90 | 16.1 / 36.6 % | **5.7 / 8.8 %** |
| Command windows < 20 % error | 65 % | **99 %** |
| Falls | 0 | 0 |
| Protocol energy / mean power | **1614 J / 13 W** | 4602 J / 37 W |
| Cost of transport (mechanical) | **7.35** | 18.3 |

**The policy tracks commands much better than the tripod (especially fast and combined commands,
where the tripod's stride cap and fixed cycle limit it), but uses 2.9× the energy.** The energy
penalty (−2e-3 per W ≈ −0.07/step at 37 W vs ≈ +1.29 tracking) is too weak to prevent it; the policy
traded efficiency for tracking. To address in 2.7 (stronger energy/torque weight or a CoT-style term)
and to report in Experiments A/B. Both CoT values are mechanical (|τ·q̇|, incl. negative work); servo
electrical losses are not modelled.
