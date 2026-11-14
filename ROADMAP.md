# Gurdy 2.0 Roadmap

Legend: `[ ]` todo · `[x]` done · **AC** = acceptance criteria (must be demonstrated live with
measured numbers before a phase is ticked). Each phase ends with a short demo (GIF/video + numbers).

RAM rule: Gazebo, RL training and the ops stack never run at the same time.

---

## Phase 0 — Project setup
- [x] CLAUDE.md and ROADMAP.md
- [x] Monorepo folder structure approved (see CLAUDE.md)
- [x] `.gitignore`, LICENSE (MIT)
- [x] `.env.example` (names only), README stub
- [x] Repo moved to WSL filesystem (`~/projects/gurdy2`)

**Phase 0 done.**

---

## Phase 1 — Robot + ROS 2 base
- [x] Docker image: ROS 2 Jazzy + Gazebo Harmonic + ros2_control + MuJoCo; `compose.yaml` with WSLg GUI, port 8080 published
- [x] `robot.yaml` single source of truth (dimensions, masses, limits, damping, gains) + `gait.yaml` + loader/tests in `gurdy_core`
- [x] 18-DOF xacro (6 legs × hip-yaw, hip-pitch, knee), inertias computed from geometry, one-sided knee limits, joint damping/friction, IMU + depth camera frames (Gazebo sensor plugins in next step)
- [x] Gazebo spawn + `ros_gz_bridge` (clock, IMU, camera, depth, ground-truth odom); matching IMU + camera in MuJoCo
- [x] ros2_control via `gz_ros2_control` **effort** interface (joint_state_broadcaster + `gurdy_control/JointPDController` at 500 Hz, same kp/kd as MuJoCo)
- [x] `gurdy_core`: leg FK/IK + tripod gait math (pure Python, unit-tested)
- [x] Tripod gait node following `/cmd_vel` (vx, vy, yaw rate) — tracking results in `docs/results/phase1_tracking.md`
- [x] MuJoCo MJCF generated from the same `robot.yaml`; URDF↔MJCF parity test (masses, inertias, limits, damping)
- [x] Joint step-response parity test: same step command in Gazebo and MuJoCo, compare rise time, overshoot and steady sag (`docs/results/phase1_parity.md`)
- [x] One bringup launch + keyboard teleop (`docker compose up sim`; teleop via `docker compose exec`) — manual teleop session done (user, 2026-11-14)
- [x] Chatter/health metric script (`scripts/parity_experiments.sh stand`, metrics in `gurdy_core.experiments`)

**AC** — ✅ all met 2026-11-14 (results: `docs/results/phase1_parity.md`, `docs/results/phase1_tracking.md`)
- ✅ `docker compose up sim` opens Gazebo GUI through WSLg; RViz shows the robot model and TF.
- ✅ URDF passes `check_urdf`; all 18 joints present; total mass and inertias match `robot.yaml`; knee can never bend backwards (limits + IK test).
- ✅ Robot stands stably for 60 s with no chatter in Gazebo and MuJoCo (worst joint RMS velocity 4.5e-6 rad/s vs limit 1e-2; no armature, passive joint damping).
- ✅ Joint step response: rise time and overshoot match between Gazebo and MuJoCo (rise within 4 ms; tolerances in `gurdy_core.experiments.PARITY_TOL`).
- ✅ Teleop: walks forward, backward, strafes, turns in place (user session, no falls). Measured vx = 0.10 m/s → −8.3 %, yaw = 0.3 rad/s → −11.3 % (limit ±25 %); no fall in the 2-minute automated session.
- ✅ Same tripod gait runs in MuJoCo headless; forward speed +6.5 % vs Gazebo (limit ±30 %).
- ✅ Measured and recorded: Gazebo RTF headless 0.76–0.82, GUI 0.52–0.72; container RAM 245–290 MiB headless, 550–576 MiB GUI.
- ✅ `colcon test` (10) and `pytest` (gurdy_core 33, sim_mujoco 9) pass in the container.

**Phase 1 done.**

---

## Phase 2 — RL locomotion (MuJoCo → Gazebo)
Plan approved 2026-11-14. Decisions: action option **A** (joint offsets around stance; B = residual on
tripod only as fallback); observation = 3-frame history of proprioception only (gravity vector, gyro,
joint pos/vel, previous action, command, gait clock) — no ground truth; asymmetric critic added by
2.7 at the latest (earlier if tracking stalls); SB3 PPO on CPU, `log_std_init` ≈ ln 0.35, entropy coef
≤ 0.001; reward = joint tracking term exp(−‖e_v‖²/σ_v² − e_ω²/σ_ω²) with command-relative sigmas
(κ 0.8 → 0.5) so standing can't farm tracking reward; optional time-boxed MJX spike after CPU works.

- [x] **2.1** `rl` image + compose profile; env skeleton; benchmark 1/4/6/8/10 workers + one PPO update
  — AC: steps/s and RAM per worker count recorded, n_envs chosen, container < 6 GB
  — done: batched VecEnv 10×4 = 40 envs, 6.8k env steps/s, PPO 3.6k samples/s, 1.75 GiB (`docs/results/phase2_bench.md`)
- [x] **2.2** Gymnasium env (flat): shared obs builder in `gurdy_core`, action mapping, termination,
  command sampling — AC: `check_env` passes; tests: obs layout/dims, no privileged inputs, targets
  always within joint limits, deterministic seeding
  — done: 11 env tests + VecEnv equivalence test pass (`rl/tests/`)
- [x] **2.3** Reward + sanity tests — AC: every term unit-tested; walking at 50 % of command clearly beats
  standing; tripod walker ≥ 2–3× standing under every move command (pure turning included);
  standing wins at zero command; tripod reference return recorded
  — done: tripod 3.7–4.0× standing (combined 11.9×), half speed 2.5–2.8×, zero cmd standing wins (`docs/results/phase2_reward.md`)
- [ ] **2.4** `train.py` + configs (flat, no DR), TensorBoard, checkpoints + resume — AC: flat tracking
  error < 20 %, fall rate < 5 % over 100 episodes, resume verified, RAM peak + wall time recorded
  (time box ~60M steps, then asymmetric critic, then option B)
- [ ] **2.5** `eval.py`: same protocol for policy and tripod; tracking, falls, energy, cost of transport,
  videos — AC: `docs/results/phase2_flat.md`
- [ ] **2.6** (optional, time-boxed) MJX + JAX spike on the RTX 4050 — AC: go/no-go from measured
  throughput (≥ 10× end-to-end) and stand/step parity vs CPU MuJoCo
- [ ] **2.7** Domain randomization (friction, masses, motor strength, damping, latency, sensor noise,
  pushes, leg failure) + asymmetric critic — AC: flat AC still met with DR; fall rate < 5 % at DR extremes
- [ ] **2.8** Terrain curriculum (heightfield: rough 0.5–2 cm, slopes 5–12°, steps 1.5–3 cm) — AC:
  success rate per level; RL beats tripod on rough terrain
- [ ] **2.9** ONNX export + `gurdy_rl_policy` ROS node + Gazebo deployment — AC: ONNX == torch (1e-5);
  walks in Gazebo without retraining; no fall in the 2-min protocol; tracking + sim-to-sim gap reported
- [ ] **2.10** Experiment A (rough terrain, RL vs tripod, MuJoCo + Gazebo) and Experiment B (one leg
  disabled) — AC: results tables in `docs/results/`

**AC**
- Training runs without freezing WSL (RAM stays under budget; numbers logged); resume from checkpoint works.
- Flat ground: velocity tracking error < 20 % over the command range; fall rate < 5 % over 100 eval episodes.
- Rough terrain: RL success rate higher than tripod gait (reported with numbers).
- Policy walks in Gazebo via `/cmd_vel` without retraining; tracking error reported for both sims.
- Results table for Experiments A and B in `docs/results/`.

---

## Phase 3 — Mapping + navigation
- [ ] Demo world with obstacles, rooms and coloured objects
- [ ] Depth → laser scan (or 3D) for SLAM; slam_toolbox mapping (RTAB-Map optional)
- [ ] Nav2 configured for a legged base (velocity limits, footprint, controller tuning) driving `/cmd_vel`
- [ ] Works with both tripod gait and RL policy
- [ ] Map save/load; `explore` behaviour (frontier exploration)

**AC**
- Builds a usable map of the demo world from teleop and from autonomous exploration.
- Nav2 reaches ≥ 8/10 goal poses without collision (reported for gait and policy).

---

## Phase 4 — Perception + voice
- [ ] CUDA setup: `gpu` compose profile (or CUDA base image) using the RTX 4050; `nvidia-smi` check in container
- [ ] Object detection by name (YOLO small or small VLM), trained/run on local GPU, CPU/ONNX fallback; 3D position from depth
- [ ] `find_object(name)` / `go_to(object)` behaviours
- [ ] Gnani Prisma STT + Timbre TTS (key from env var only); offline Whisper fallback
- [ ] Replies always start with "Main ek AI assistant hoon"

**AC**
- Detects each demo-world object by name; precision/recall measured on a recorded set.
- "Laal dabba dhundo" style command finds and approaches the object in sim.
- Works with network off (Whisper fallback); STT latency measured for both paths.
- Detector runs on GPU and on the CPU/ONNX fallback; FPS reported for both.

---

## Phase 5 — Gurdy Brain (LLM post-training)
- [ ] Tool schema: walk, turn, go_to, find_object, explore, stop, report
- [ ] Synthetic dataset: Hindi/Hinglish/Kannada/English; noisy-STT variants; multi-step missions; ambiguous and unsafe requests
- [ ] Held-out test split (incl. "baayen"/"haan ye" style confusions)
- [ ] SFT (LoRA, PEFT/TRL) on Qwen3-0.6B locally (RTX 4050, 6 GB); Qwen3-1.7B with QLoRA 4-bit if it fits — Colab/Kaggle notebooks as fallback
- [ ] DPO: prefer clarification over guessing; refuse unsafe actions
- [ ] Eval harness: tool-call exact match, argument accuracy, direction errors, safety refusal rate, clarification rate — vs keyword parser and base model
- [ ] GGUF quantization, Ollama Modelfile; latency measured on local GPU and CPU
- [ ] Hugging Face model card (with eval table)
- [ ] LangGraph mission agent calling ROS 2 tools

**AC**
- Fine-tuned model beats keyword parser and base model on tool-call accuracy and direction errors (numbers in model card).
- Unsafe-request refusal ≥ 95 %, with low over-refusal on safe commands (both reported).
- Runs offline via Ollama on local GPU and on CPU only; latency per command reported for both.
- End-to-end demo: spoken multi-step mission executed in sim.

---

## Phase 6 — Robot Incident Graph (robot-ops)
- [ ] Telemetry/log exporter: ROS 2 → Redpanda (also from rosbag replay)
- [ ] Event normalizer (schema), fingerprinting, embeddings → Qdrant
- [ ] HDBSCAN clustering into incidents → PostgreSQL
- [ ] Redis/Celery workers; FastAPI backend
- [ ] LangGraph investigation agent (logs, config diffs, metrics) → root-cause reports
- [ ] Fault injection: high D-gain chatter, weak motors, slippery floor, disabled leg
- [ ] Recorded telemetry datasets so the stack runs without the simulator

**AC**
- Stack fits in RAM alongside nothing else heavy (measured), runs on recorded telemetry.
- Each injected fault type forms its own incident cluster (purity/ARI reported).
- Investigation agent names the correct root cause for ≥ 3 of 4 fault types.

---

## Phase 7 — Mission control + polish
- [ ] Next.js/TypeScript app: rosbridge, three.js 3D robot view, map, camera, chat, incidents page
- [ ] Docker compose profiles per part
- [ ] GitHub Actions CI (lint + unit tests; ROS build/test job)
- [ ] README with GIFs and result tables; project website

**AC**
- Fresh clone → each phase's demo starts with one documented command.
- CI green on `main`.
