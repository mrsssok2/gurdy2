# Experiment B — leg-failure adaptation

Question: can a blind RL policy keep walking when one leg dies, if it has to infer the failure from proprioception alone?

## Failure model (implemented in `rl/envs/randomization.py`, `rl/envs/gurdy_env.py`)
- In a fraction `leg_failure_prob` (0.5 in training) of episodes one random leg (0–5) fails at a random time (uniform 1–10 s of a 20 s episode).
- **limp**: the 3 motors of that leg produce zero torque from then on (leg flops; joint damping/friction remain).
- **locked**: the PD target of the 3 joints is frozen at the joint angles of the failure moment (the servo holds that pose).
- Type chosen 50/50. The failure is applied inside `env.step` (`_maybe_fail_leg`); `reset` restores the healthy robot.
- **Actor is not told:** its 195 inputs are unchanged (joint positions/velocities, gyro, gravity, command, previous actions, 3-frame history). Test: actor obs shape unchanged, no failure flag.
- **Critic is told** (`failure_privileged: true`): +8 privileged values = one-hot failed leg (6, zero before the failure) + limp flag + locked flag. The critic input grows 212 → 220;
  `policies.extend_privileged` copies terrain_dr2 (actor identical, new critic columns start at 0; tested).
- Episodes with a failure do not move the terrain curriculum (a fall caused by the failure must not demote the terrain level).

## Training (`rl/configs/leg_failure.yaml`, run name `leg_failure`)
Fine-tune from `terrain_dr2` 15M (`rl/runs/terrain_dr2/checkpoints/ppo_latest.zip`), 30M steps, full DR (as terrain_dr2), terrain = flat 35 % / rough ≤ 2 cm / slopes ≤ 8° / steps ≤ 2.5 cm,
critic warm-up 0.41M steps with the actor frozen (new critic inputs), everything else (reward, PPO, command mix) as terrain_dr2. No reward change: tracking reward under a dead leg is simply lower.

## Baseline observation before training (smoke eval, 2 episodes per cell, so only indicative)
terrain_dr2 *without* adaptation degrades badly under a limp leg (post-failure tracking error ≈ 90 % vs the tripod's ≈ 39 %), and less under a locked leg (≈ 63 % vs 38 %); no falls anywhere
(the robot has 5 legs left, it stalls/drifts rather than falls). Falls are therefore a weak metric here; tracking error and CoT are the main ones.
Real numbers come from the full evaluation below.

## Evaluation plan (`rl/eval_failure.py`; run after training, MuJoCo, flat floor, DR off, nothing else running)
```bash
docker compose run --rm rl python3 rl/eval_failure.py --policies terrain_dr2:latest leg_failure:latest --episodes 10
```
- Controllers: tripod (no adaptation), `terrain_dr2` 15M (no adaptation), `leg_failure` (candidate checkpoints: 20M and latest; pick with the rule below).
- Cells: every failed leg (0–5) × type (limp, locked) × 10 episodes (same seeds and commands for every controller), plus 50 no-failure episodes.
- The failure happens at 5 s (end of the first command window). **Metrics on the windows after the failure only** (first second of each window skipped): fall rate (whole episode), tracking error
  (episode mean of window errors, `rl/eval.py` definition), CoT = energy / (m g path) after the failure. Pooled rows over the 6 legs per type.
- **No-failure regression check:** the no-failure row (50 episodes) vs `terrain_dr2`, plus `rl/eval.py --policies terrain_dr2:latest leg_failure:latest --scenarios all` (9 DR scenarios), `--terrain all` and the start gate;
  accept the candidate only if nominal tracking error ≤ 12 % (terrain_dr2: 10.6 %), 0 falls in the DR scenarios, and rough/slopes/steps success within 10 points of terrain_dr2.
- Success criterion for the experiment (set before seeing results): post-failure tracking error pooled over legs < 40 % for both failure types (tripod ≈ 38 %) and clearly below terrain_dr2 without adaptation; report per-leg results including the worst leg. If the criterion is missed, say so.
- Not covered (limitations): simultaneous failures, failure of a single joint, partial torque loss, failures on terrain, Gazebo transfer.
