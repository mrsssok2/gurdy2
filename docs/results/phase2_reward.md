# Phase 2 step 2.3 — Reward design and sanity tests

Reward: `rl/envs/reward.py`, weights in `rl/configs/flat.yaml`. Tests: `rl/tests/test_reward.py`
(64 tests in `rl/tests` pass). Raw returns and per-term sums: `phase2_reward_reference.json`.

## Tracking term (dominant)
r_track = exp(−‖v̄_xy − c_xy‖²/σ_v² − (ω̄_z − c_ω)²/σ_ω²), weight 1.5, with
σ_v = max(κ‖c_xy‖, 0.02 m/s), σ_ω = max(κ|c_ω|, 0.05 rad/s), κ = 0.8 (annealed to 0.5 in training).

* **One joint term**: standing cannot collect the yaw part under a forward command, nor the linear
  part under a pure turn.
* **Command-relative sigma**: standing under any move command scores exp(−1/κ²) — 0.21 at κ 0.8,
  0.018 at κ 0.5 — independent of command size (a fixed σ = 0.1 m/s would give standing 0.37–0.53
  under slow commands).
* **Floors** (user decision): zero and pure-turn commands are well defined; at zero command a drift of
  0.02 m/s or 0.05 rad/s costs a factor e⁻¹.
* **Filtered velocity v̄** (EMA, τ = 0.2 s): measured tripod stride surge — forward speed oscillates
  0.013–0.137 m/s around a 0.096 mean (std 0.033; EMA std 0.008). With instantaneous velocity the
  half-speed walker scored almost as much as the full-speed walker (263 vs 268) and tripod/stand was
  only 2.47× forward; filtered: 288 vs 405 and 3.74×.

Action scale raised to hip_yaw 0.3 / hip_pitch 0.7 / knee 0.6 rad: the tripod gait needed up to
0.47 / 0.56 / 0.51 rad (with 0.5 for hip_pitch it was not representable, |a| up to 1.12).

## Sanity test results (6 s episodes, fixed command, MuJoCo, seed 0)

| Command | Stand | Tripod | Half-speed tripod | Tripod / stand | Half / stand |
|---|---|---|---|---|---|
| forward 0.10 | 108.4 | 405.2 | 287.5 | 3.74 | 2.65 |
| backward −0.10 | 108.5 | 405.0 | 287.5 | 3.73 | 2.65 |
| strafe left 0.08 | 108.8 | 437.2 | 304.0 | 4.02 | 2.79 |
| strafe right −0.08 | 108.1 | 434.7 | 301.3 | 4.02 | 2.79 |
| turn left 0.3 (pure) | 108.7 | 420.5 | 269.0 | 3.87 | 2.47 |
| turn right −0.3 (pure) | 108.2 | 420.2 | 268.6 | 3.88 | 2.48 |
| combined (0.08, 0.04, 0.2) | 34.0 | 406.4 | 179.4 | 11.94 | 5.27 |
| slow 0.05 | 108.4 | 427.1 | 290.6 | 3.94 | 2.68 |
| fast turn 0.6 | 108.6 | 414.2 | 269.1 | 3.81 | 2.48 |
| zero | **460.3** | (gait stands) | (gait stands) | — | — |

Test thresholds: tripod ≥ 2.5× standing and half-speed ≥ 1.5× standing for every move command;
at zero command standing beats walking at 0.1 m/s by > 2×; nobody falls; tripod |a| ≤ 1 (max 0.82).

Tripod reference return (forward, 6 s): 405 = track 395 + alive 15 + air time 2.7 − penalties 7.5
(energy −4.4, foot slip −1.6, torque −0.7, joint acc −0.4, vz −0.2, action rate −0.1): tracking
dominates; penalties are ~2 % of the return.
