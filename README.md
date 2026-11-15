# Gurdy 2.0

An 18-DOF hexapod robot in simulation: Hindi/Hinglish voice commands, an offline fine-tuned LLM
mission planner, an RL gait for rough terrain, mapping + navigation, object finding by name, and a
robot-ops system that clusters failures into incidents and diagnoses root causes.
ROS 2 Jazzy · Gazebo Harmonic · MuJoCo · Docker (WSL2). License: MIT.

**Status:** Phase 1 (robot + ROS 2 base) complete — see [ROADMAP.md](ROADMAP.md) and
[docs/results/](docs/results/).

## Quick start (Phase 1 demo)

```bash
docker compose build sim
docker compose up sim                 # walking robot in Gazebo (first run builds the workspace)
# second terminal:
docker compose exec sim bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
docker compose --profile '*' down     # stop everything
```

## Phase 1 results (measured)

| | Result |
|---|---|
| Velocity tracking, Gazebo (vx 0.10 m/s / yaw 0.3 rad/s) | −8.3 % / −11.3 % (limit ±25 %), no fall in 2 min |
| MuJoCo vs Gazebo forward speed | +6.5 % (limit ±30 %) |
| Joint step-response parity Gazebo ↔ MuJoCo | rise time within 4 ms |
| Chatter (60 s standing) | worst joint RMS velocity 4.5e-6 rad/s (limit 1e-2) |

**Phase 2 (in progress) — RL locomotion in MuJoCo** (policies BC-initialised from the tripod gait + PPO):

| | tripod gait | flat policy | DR + energy policy (`dr_energy_ft`) |
|---|---|---|---|
| tracking error, nominal | 15.9 % | 4.9 % | 10.9 % |
| tracking error, all DR extremes | 13.8 % | 35.1 % | 11.8 % |
| tracking error, sensor noise | 16.2 % | 99.2 % | 11.3 % |
| falls (all 9 scenarios) | 0 % | 0 % | 0 % |
| starts from rest (48) | 48/48 | 48/48 | 48/48 |
| cost of transport | 7.35 | 18.3 | 6.93 |

[flat results](docs/results/phase2_flat.md) · [DR results](docs/results/phase2_dr.md) ·
[3-way video](docs/videos/phase2_3way_tripod_flat_dr.mp4) (DR panel: 40M checkpoint)

Details: [parity + chatter](docs/results/phase1_parity.md) ·
[velocity tracking](docs/results/phase1_tracking.md). Design notes and rules: [CLAUDE.md](CLAUDE.md).

## RL training (Phase 2)

Runs live in `rl/runs/<name>/` (`train.log`, `tb/`, `checkpoints/` every 2M steps incl. VecNormalize
stats; `ppo_latest.*` is written on stop). Training runs detached, so it survives closing VS Code.

```bash
# start (example: long flat run, BC-initialised from the tripod gait)
mkdir -p rl/runs/flat_bc_v1
docker compose run -d --rm --name gurdy_train rl bash -c 'exec python3 rl/train.py \
  --config rl/configs/flat_bc_long.yaml --name flat_bc_v1 \
  --init-from rl/runs/c2_bc/checkpoints/ppo_bc.zip > rl/runs/flat_bc_v1/train.log 2>&1'

# progress
docker ps                                            # gurdy_train running?
tail -f rl/runs/flat_bc_v1/train.log                 # live log (Ctrl+C stops tail, not training)
docker compose run --rm rl python3 rl/tools/summarize_run.py flat_bc_v1   # curves as a table
docker stats --no-stream gurdy_train                 # RAM / CPU
# TensorBoard on http://localhost:6006
docker compose run -d --name gurdy_tb --service-ports rl tensorboard --logdir rl/runs --bind_all --port 6006

# stop (saves ppo_latest), then remove the container
docker stop gurdy_train && docker rm gurdy_train

# resume from the newest checkpoint (or --resume rl/runs/<name>/checkpoints/ppo_<N>_steps.zip)
docker compose run -d --rm --name gurdy_train rl bash -c 'exec python3 rl/train.py \
  --config rl/configs/flat_bc_long.yaml --name flat_bc_v1 --resume latest >> rl/runs/flat_bc_v1/train.log 2>&1'
```
Never run training together with Gazebo (RAM rule); training uses ~1.6 GiB and ~7 CPU threads.
