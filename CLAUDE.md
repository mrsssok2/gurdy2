# CLAUDE.md — Gurdy 2.0

Read this first in every session. Keep it up to date when decisions change.

## What this is
Gurdy 2.0: an 18-DOF hexapod that takes Hindi/Hinglish voice commands, plans missions with an
offline fine-tuned LLM ("Gurdy Brain"), walks with an RL gait over rough terrain, maps and navigates,
finds objects by name, and has a robot-ops system ("Robot Incident Graph") that clusters failures
into incidents and auto-diagnoses root causes. Every phase must be **independently demoable**.

Predecessor (reference only, **never modify**): github.com/mrsssok2/gurdy_rl
(ROS Noetic + Gazebo Classic, 12-DOF, DQN, Gnani voice, OpenCV, Flask). Clone it outside this
repo (e.g. into the scratchpad) if needed for reference.

Full plan with checkboxes and acceptance criteria: **ROADMAP.md**.

## The user
- Final-year B.Tech AI & Data Science student. Comfortable with Python; new to ROS 2 and Claude Code.
- **Explain ROS 2 and other new concepts briefly as we go** (1–3 sentences, not lectures).

## Machine / environment
- Windows 11 + WSL2 Ubuntu + Docker Desktop (WSL integration), VS Code.
- 16 GB RAM total; `.wslconfig` caps WSL at **8 GB RAM + 8 GB swap**. 12 logical CPUs.
- GPU (measured 2026-11-14): **NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM**, driver 621.14, visible in
  WSL (`/usr/lib/wsl/lib/nvidia-smi`); Docker Desktop has the `nvidia` runtime. Used now for OpenGL
  rendering (Mesa d3d12). GPU/CUDA plan (decided 2026-11-14):
  - Phase 2 RL: MuJoCo PPO stays on **CPU**; GPU only if a later CNN-encoder terrain policy benefits.
  - Phase 4 perception: train + run YOLO on local **CUDA**, with a CPU/ONNX fallback.
  - Phase 5 LLM: LoRA on Qwen3-0.6B locally first; Qwen3-1.7B with QLoRA (4-bit) if it fits in
    6 GB; Colab/Kaggle as fallback. Local GPU inference via Ollama.
  - Add CUDA setup (CUDA base image or a `gpu` compose profile) **only when Phase 2/4 needs it**.
  - VRAM is 6 GB: one GPU workload at a time, same as the RAM rule.
- Gazebo GUI via **WSLg** (`/mnt/wslg` + its X11 socket mounted into the container; host
  `/tmp/.X11-unix` is empty on this distro). Keep cameras small (≤ 320x240) and frame rates modest.
- Free cloud GPUs on Google Colab / Kaggle: fallback for Phase 5 LLM training (notebooks).

## Hard rules
1. **Secrets** (e.g. `GNANI_API_KEY`) come only from environment variables. Never write them to
   files, logs, notebooks, compose files or commits. `.env.example` may list variable *names* only.
2. **RAM budget:** never run Gazebo, RL training and the ops stack at the same time. One heavy
   workload at a time; use compose profiles. Report RAM usage when testing.
3. **Test live in the container** whenever possible, not just offline. Report **real measured
   numbers** (RTF, velocity tracking error, RAM, success rates) — never invented or estimated ones.
4. **Small steps.** One phase at a time; show the plan for each phase before building it.
5. **Commit after each working step** with a clear message. **Ask before pushing.**
6. If something is ambiguous or risky, **ask**.
7. Keep ROADMAP.md checkboxes and this file current as we progress.
8. **Before measuring, check `docker ps`**: no other sim containers may run (they share CPU, and
   without isolation they share /clock). Stop GUI demos when done (`docker rm -f <name>`).

## Lessons from Gurdy 1.0 (design constraints, not trivia)
| Gurdy 1.0 problem | Gurdy 2.0 rule |
|---|---|
| 2 joints/leg → couldn't turn | 3 DOF/leg: hip-yaw, hip-pitch, knee (18 DOF) |
| DQN with 25 tiny discrete actions learned slowly | PPO, continuous joint-target actions, velocity-command conditioned |
| Training in Gazebo: 450 eps ≈ 12 h, froze WSL | Train in **MuJoCo** headless with parallel envs; Gazebo only for sim-to-sim deployment |
| Tiny masses/inertias + high D-gains → chatter | Realistic masses; inertias **computed from geometry**; moderate gains; joint damping; automated chatter metric |
| Knees bent backwards | One-sided knee limits in URDF *and* MJCF *and* IK; unit test for it |
| Reward paid more for standing than walking | Reward dominated by command tracking; test that a standing policy scores lower than a walking one when commanded to move |
| Keyword parser misheard "baayen" → "haan ye" | LLM parser trained on noisy-STT variants; prefers clarification over guessing on low confidence |
| Manual debugging from logs | Phase 6 incident graph + structured telemetry from day one |

## Tech stack (decided)
- ROS 2 **Jazzy** + Gazebo **Harmonic** (`ros_gz`, `gz_ros2_control`) in Docker on Ubuntu 24.04.
- ros2_control with `joint_state_broadcaster` + `gurdy_control/JointPDController` (effort PD, 500 Hz).
- MuJoCo + Gymnasium + Stable-Baselines3 PPO (CPU, `SubprocVecEnv`; GPU only for a CNN encoder).
- SLAM: slam_toolbox (2D, light) first; RTAB-Map optional. Nav2.
- Perception: YOLO (small) or small VLM; Voice: Gnani Prisma STT / Timbre TTS, Whisper fallback.
  Spoken/text replies start with **"Main ek AI assistant hoon"**.
- LLM: Qwen3-0.6B/1.7B, LoRA/QLoRA via PEFT/TRL (SFT → DPO) on local RTX 4050 (Colab/Kaggle
  fallback), GGUF → Ollama (local GPU inference), LangGraph agent.
- Ops: Redpanda, Qdrant, HDBSCAN, PostgreSQL, Redis/Celery, FastAPI, LangGraph investigator.
- Web: Next.js/TypeScript, rosbridge, three.js. CI: GitHub Actions.

## Key design decisions
- **Single source of truth for robot geometry:** `robot.yaml` (dimensions, masses, limits, gains)
  is read by the xacro (URDF), the MJCF generator, and the IK library. Parity is checked by a test.
- **Shared pure-Python core** (`gurdy_core`): kinematics/IK and gait math, no ROS imports, used by
  the ROS gait node and the MuJoCo env alike. Unit-tested without ROS.
- **Same PD in both sims:** `tau = kp*(q_ref - q) - kd*qdot`, clipped to effort, at **500 Hz**
  (Gazebo: `gurdy_control/JointPDController` on the `gz_ros2_control` effort interface; MuJoCo:
  affine actuator, every 2 ms Euler step). Gait/RL at 50 Hz. kp 20, kd 0.05.
- **Joint damping 0.42 = effort / velocity** (servo back-EMF; DC torque-speed line), implicit in DART
  and MuJoCo. Needed because DART hard-enforces the URDF velocity limit and MuJoCo has none.
- **No armature (decided 2026-11-14):** Gazebo Harmonic (SDFormat 14 / gz-physics DART) has no
  joint armature; folding it into link inertias is non-physical (MuJoCo rejects it) or inflates body
  inertia 70–160 %. Armature = 0 in both sims (enforced by the loader). Worst-case explicit-PD
  ratios (tested): kd·dt/I ≤ 0.35, ω·dt ≤ 0.53 (hard limit 2).
- **Known sim differences (accepted, measured):** joint Coulomb friction (DART hard vs MuJoCo soft,
  ≤ 2.5 mrad steady error); MuJoCo soft foot contacts splay feet ~2 mm when standing (joint sag
  distribution differs, body height equal). Details: `docs/results/phase1_parity.md`.
- Robot geometry lives in `robot.yaml`; gait timing in `gurdy_core/gurdy_core/config/gait.yaml`.
- Gait/policy nodes publish joint targets; everything above them speaks only `/cmd_vel`
  (`geometry_msgs/Twist`: linear.x, linear.y, angular.z), so Nav2/teleop/LLM are swappable.

## Repo layout (approved)
Repo lives in the WSL filesystem: `~/projects/gurdy2` (not `/mnt/c/...` — bind mounts from the
Windows drive are slow for colcon builds). License: MIT.

```
gurdy2/
├── CLAUDE.md  ROADMAP.md  README.md  LICENSE  .env.example (variable names only)
├── compose.yaml               # one file, profiles: sim, rl, nav, voice, brain, ops, web
├── docker/
│   ├── ros/Dockerfile         # Jazzy + Gazebo Harmonic + ros2_control + MuJoCo
│   ├── rl/Dockerfile          # headless: MuJoCo + PyTorch CPU + SB3 (no ROS)
│   └── ops/ …                 # Phase 6
├── gurdy_core/                # pure Python, no ROS: leg FK/IK, gait math, robot.yaml loader
│   └── tests/
├── ros2_ws/src/               # colcon workspace
│   ├── gurdy_description/     # robot.yaml (single source of truth), xacro, RViz config
│   ├── gurdy_gazebo/          # worlds, gz bridge config, spawn launch
│   ├── gurdy_control/         # ros2_control controller YAML
│   ├── gurdy_gait/            # tripod gait node (/cmd_vel -> joint targets)
│   ├── gurdy_bringup/         # top-level launch files
│   ├── gurdy_interfaces/      # custom msgs/srvs/actions (when needed)
│   └── (later) gurdy_rl_policy, gurdy_navigation, gurdy_perception, gurdy_voice, gurdy_agent
├── sim_mujoco/                # MJCF generator (from robot.yaml) + parity test
├── rl/                        # Phase 2: envs/, configs/, train.py, eval.py
├── brain/                     # Phase 5: data_gen/, notebooks/, eval/, ollama/, agent/
├── ops/                       # Phase 6: services/, workers/, api/, fault_injection/
├── web/                       # Phase 7: Next.js app
├── docs/                      # design notes, results/, gifs/, website
├── scripts/                   # dev helpers (chatter metric, RAM monitor, ...)
└── .github/workflows/
```

## Common commands
```bash
docker compose build sim                          # build ROS/Gazebo image (gurdy2/ros:jazzy, ~5 GB)
docker compose run --rm sim bash                  # shell in the image (ROS sourced, repo at ~/gurdy2)
docker compose run --rm sim scripts/check_display.sh   # sockets, OpenGL renderer, versions, MuJoCo
docker compose exec sim /entrypoint.sh <cmd>      # run a ROS/gz command in the running container
GZ_GALLIUM_DRIVER=llvmpipe docker compose up sim  # fallback: CPU rendering if GPU path breaks
docker compose --profile '*' down                # stop ALL services (plain `down` skips profiled ones!)
# inside the container (docker compose run --rm sim bash):
cd ros2_ws && colcon build --symlink-install && source install/setup.bash && cd ..
colcon test --base-paths ros2_ws && colcon test-result          # ROS package tests (run in ros2_ws)
(cd gurdy_core && python3 -m pytest -q); (cd sim_mujoco && python3 -m pytest -q)
ros2 launch gurdy_description display.launch.py                 # RViz + joint sliders
ros2 launch gurdy_gazebo sim.launch.py [gui:=false] [fixed_base:=true] [robot_yaml:=...]
scripts/parity_experiments.sh step|stand OUT_DIR [robot.yaml]   # Gazebo vs MuJoCo (+ chatter)
python3 -m sim_mujoco.mjcf > /tmp/gurdy.xml                      # generated MuJoCo model
```
```bash
docker compose up sim                            # walking robot, Gazebo GUI (builds on first run)
docker compose exec sim bash                     # 2nd terminal, then:
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
scripts/tracking_experiments.sh OUT_DIR          # (in container) 2-min tracking test, both sims
python3 scripts/measure_rtf.py                   # (in container, sim running) RTF from sim/wall time
```
```bash
docker compose build rl                          # RL image (no ROS), ~2.9 GB
docker compose run --rm rl python3 rl/bench.py   # throughput / RAM benchmark
docker compose run --rm rl bash -c 'cd rl && python3 -m pytest -q'
```

## Current status
- **Phase 0 and Phase 1 done (2026-11-14), pushed to github.com/mrsssok2/gurdy2.** All Phase 1 AC
  met; results in `docs/results/phase1_parity.md` and `docs/results/phase1_tracking.md`.
- Built: `gurdy_core` (robot.yaml loader, FK/IK, TripodGait, shared experiment protocols),
  `gurdy_description` (xacro from robot.yaml, massless base_link root + body_link), `gurdy_control`
  (C++ JointPDController, effort interface, 500 Hz), `gurdy_gazebo` (world, sim launch, sensors +
  lazy bridge, GT odom), `gurdy_gait` (node timed by /joint_states stamps with catch-up),
  `gurdy_bringup`, `sim_mujoco` (MJCF generator, IMU/camera, experiments).
- Key numbers: tracking vx 0.10 → −8.3 %, wz 0.3 → −11.3 % (Gazebo); MuJoCo +6.5 % vs Gazebo;
  step parity within 4 ms; no chatter. RTF headless 0.76–0.82, GUI 0.52–0.72 (gz server bound);
  RAM 245–290 MiB headless, 550–576 MiB GUI.
- Lessons: plain `docker compose down` skips profiled services; containers must be isolated
  (GZ_PARTITION, ROS_AUTOMATIC_DISCOVERY_RANGE); measure RTF as sim/wall over windows; host clock
  varies under load (report ranges + host benchmark); rclpy use_sim_time costs ~40 % CPU at 1 kHz /clock.
- Open items: GUI RTF ~0.5 while walking; MuJoCo soft-contact foot splay (revisit `impratio` for RL).
- **Phase 2 in progress** (plan approved 2026-11-14, steps 2.1–2.10 in ROADMAP.md). Decisions: action
  option A (offsets around stance; B residual only as fallback), 3-frame proprioceptive history, no
  ground truth in obs, asymmetric critic by 2.7 at the latest; reward tracking term
  exp(−‖e_v‖²/σ_v² − e_ω²/σ_ω²) with σ_v = max(κ‖v_cmd‖, 0.02), σ_ω = max(κ|ω_cmd|, 0.05), κ 0.8 → 0.5
  (standing cannot farm tracking reward; tests: 50 % speed ≫ standing, tripod ≥ 2–3× standing for
  every move command incl. pure turning); PPO log_std_init ln 0.35, ent_coef ≤ 0.001; MJX spike after CPU.
- 2.1 done: `docker/rl` image (no ROS; MuJoCo 3.3.0 = sim image), compose profile `rl`,
  `gurdy_core/rl_obs.py` (shared obs builder), `rl/envs/gurdy_env.py` (skeleton, placeholder reward),
  `rl/vec_env.py` BatchedSubprocVecEnv (10 workers × 4 envs: 6.8k env steps/s, 1.55 GiB vs SB3
  SubprocVecEnv 3.3k / 3.1 GiB), PPO loop 3.6k samples/s with torch threads 2 (4 is slower).
- 2.2 done: env config `rl/configs/flat.yaml`, fixed commands via reset options, foot contacts,
  info diagnostics (never in obs). Tests: check_env, obs layout/history, **obs invariant to world
  position/heading/linear velocity** (no privileged inputs), targets within limits, seeding, command
  sampling/resampling, truncation, falls. Note: MuJoCo sensors (gyro) are 1 physics step (2 ms) old.
- 2.3 done: `rl/envs/reward.py` + weights in `rl/configs/flat.yaml`. Tracking uses the body velocity
  **EMA-filtered (τ 0.2 s)**: tripod strides surge 0.013–0.137 m/s around 0.096; unfiltered, half-speed
  scored ≈ full speed. Action scale now hip_yaw 0.3 / hip_pitch 0.7 / knee 0.6 rad (tripod needed 0.56
  on hip_pitch). Sanity: tripod 3.7–4.0× standing for every move command (pure turns 3.9×), half
  speed 2.5–2.8×, standing wins at zero command. 64 tests in `rl/tests`. `docs/results/phase2_reward.md`.
- Pushing: push to GitHub after every completed step (user, 2026-11-14).
- Next: 2.4 (train.py, configs, TensorBoard, checkpoints/resume, flat training run).
