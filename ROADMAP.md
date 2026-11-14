# Gurdy 2.0 Roadmap

Legend: `[ ]` todo · `[x]` done · **AC** = acceptance criteria (must be demonstrated live with
measured numbers before a phase is ticked). Each phase ends with a short demo (GIF/video + numbers).

RAM rule: Gazebo, RL training and the ops stack never run at the same time.

---

## Phase 0 — Project setup
- [x] CLAUDE.md and ROADMAP.md
- [x] Monorepo folder structure approved (see CLAUDE.md)
- [x] `.gitignore`, LICENSE (MIT)
- [ ] `.env.example` (names only), README stub
- [x] Repo moved to WSL filesystem (`~/projects/gurdy2`)

---

## Phase 1 — Robot + ROS 2 base
- [x] Docker image: ROS 2 Jazzy + Gazebo Harmonic + ros2_control + MuJoCo; `compose.yaml` with WSLg GUI, port 8080 published
- [x] `robot.yaml` single source of truth (dimensions, masses, limits, damping, gains) + `gait.yaml` + loader/tests in `gurdy_core`
- [ ] 18-DOF xacro (6 legs × hip-yaw, hip-pitch, knee), inertias computed from geometry, one-sided knee limits, joint damping/friction, IMU, forward depth camera
- [ ] Gazebo spawn + `ros_gz_bridge` (clock, IMU, camera, depth, ground-truth odom)
- [ ] ros2_control via `gz_ros2_control` **effort** interface (joint_state_broadcaster + joint-group PD at 500 Hz, same kp/kd as MuJoCo)
- [ ] `gurdy_core`: leg FK/IK + tripod gait math (pure Python, unit-tested)
- [ ] Tripod gait node following `/cmd_vel` (vx, vy, yaw rate)
- [ ] MuJoCo MJCF generated from the same `robot.yaml`; URDF↔MJCF parity test (masses, inertias incl. folded-in armature, limits)
- [ ] Joint step-response parity test: same step command in Gazebo and MuJoCo, compare rise time and overshoot
- [ ] One bringup launch + keyboard teleop
- [ ] Chatter/health metric script

**AC**
- `docker compose up sim` opens Gazebo GUI through WSLg; RViz shows the robot model and TF.
- URDF passes `check_urdf`; all 18 joints present; total mass and inertias match `robot.yaml`; knee can never bend backwards (limits + IK test).
- Robot stands stably for 60 s with no visible chatter (chatter metric below threshold) — proves the armature-as-link-inertia compensation in Gazebo.
- Joint step response: rise time and overshoot match between Gazebo and MuJoCo (tolerance set and reported in the test).
- Teleop: walks forward, backward, strafes, turns in place. Measured mean velocity within ±25 % of command for vx = 0.10 m/s and yaw = 0.3 rad/s; no fall in a 2-minute session.
- Same tripod gait runs in MuJoCo headless; forward speed within ±30 % of Gazebo.
- Measured and recorded: Gazebo RTF (GUI and headless), container RAM.
- `colcon test` and `pytest gurdy_core` pass in the container.

---

## Phase 2 — RL locomotion (MuJoCo → Gazebo)
- [ ] Gymnasium env (MuJoCo) with velocity-command observation, joint-target actions (residual over a nominal stance)
- [ ] Reward: command tracking (dominant), penalties for energy, falls, joint jerk/action rate, body tilt
- [ ] Reward sanity test: standing still under a move command scores lower than walking
- [ ] PPO (SB3) with parallel CPU envs; config files; checkpoints, resume, TensorBoard logs
- [ ] (Only if needed) CUDA setup for a CNN-encoder terrain policy: `gpu` compose profile / CUDA base image
- [ ] Terrain curriculum: flat → slopes → rough heightfield → steps
- [ ] Domain randomization: friction, link mass, motor strength, latency/noise
- [ ] Evaluation script (fixed seeds, per-terrain metrics) + videos
- [ ] Policy exported (ONNX/TorchScript) and deployed as ROS 2 node in Gazebo (sim-to-sim)
- [ ] Experiment A: RL vs tripod gait on rough terrain
- [ ] Experiment B: RL vs tripod gait with one leg disabled

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
