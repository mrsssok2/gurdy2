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
- [x] **2.4** `train.py` + configs (flat, no DR), TensorBoard, checkpoints + resume — AC: flat tracking
  error < 20 %, fall rate < 5 % over 100 episodes, resume verified, RAM peak + wall time recorded
  (time box ~60M steps, then asymmetric critic, then option B)
  — done (2026-11-15): from-scratch runs stayed below standing within 2M steps (best: C1, action EMA);
  chosen policy **BC-initialised from the tripod gait** + PPO, 60M steps in 3.84 h, 1.64 GiB:
  tracking error 5.7 % (100 eps), 0 % falls, resume verified (`docs/results/phase2_flat.md`)
- [x] **2.5** `eval.py`: same protocol for policy and tripod; tracking, falls, energy, cost of transport,
  videos — AC: `docs/results/phase2_flat.md`
  — done: policy 5.7 % vs tripod 16.1 % window error, both 0 falls; policy CoT 18.3 vs tripod 7.35 (2.9× energy)
- [ ] **2.6** (optional, time-boxed) MJX + JAX spike on the RTX 4050 — AC: go/no-go from measured
  throughput (≥ 10× end-to-end) and stand/step parity vs CPU MuJoCo
- [x] **2.7** Domain randomization (friction, masses, motor strength, damping, latency, sensor noise,
  pushes, leg failure) + asymmetric critic + energy efficiency (policy CoT 18.3 vs tripod 7.35) — AC: flat AC still met with DR; fall rate < 5 % at DR extremes
  — done (2026-11-15): `dr_energy_v1` (BC-init via flat_bc_v1), 60M steps / 5.03 h; **selected 40M**:
  0 % falls in all 9 DR scenarios, tracking 10.0–12.2 % (flat_bc_v1: 99 % under sensor noise), CoT 7.70
  (tripod 7.35, flat 18.3). Known issues: user target tracking < 10 % narrowly missed (10.9 %); after
  ~45M the policy stops starting left strafes from rest (start_test) → 2.7b (`docs/results/phase2_dr.md`)
- [x] **2.7b** Command distribution with axis-aligned + start-from-rest commands; `start_test`
  as a regression gate in eval; short fine-tune from 40M — AC: 0/48 failed starts, tracking < 10 %, CoT ≤ 11
  — done (2026-11-15): `dr_energy_ft` 10M (10M steps, 0.96 h): start gate 0/48, CoT 6.93, 0 % falls in all 9
  scenarios; tracking first reported 10.9 % (eval bug: graded on its own command mix) → **corrected 9.6 ± 0.4 %
  (met nominally)**; ft-6M 9.2 ± 0.6 % is better in all 9 scenarios (selection 6M vs 10M open).
  Found: dead zone — stands still on vx/vy ±0.03, wz ±0.10 (not reward-optimal; sampling) → fix in 2.8
- [ ] **2.8** Terrain curriculum (heightfield: rough 0.5–2 cm, slopes 5–12°, steps 1.5–3 cm) — AC:
  success rate per level; RL beats tripod on rough terrain. Also fix the small-command dead zone:
  ~15 % small commands (15–40 % of limit), zero commands 15 → 10 %, explicit `/cmd_vel` deadband
  (|v| < 0.015 m/s, |ω| < 0.05 rad/s), start gate extended to small commands (from rest + while walking)
  — in progress (2026-11-15): built `rl/envs/terrain.py` (heightfield per env, 10-level per-env curriculum,
  rough 0.2–4 cm / slopes 2–15° / steps 0.5–5 cm + 10 % flat plane), `gurdy_core/rl_command.py` deadband,
  small commands, `rl/eval.py --terrain` grid + 144-start gate; run `terrain_v1` (init ft-6M, 60M steps)
  training. Baseline ft-6M vs tripod (12 eps/cell): tripod drops to 67–75 % success on 3–3.5 cm steps,
  ft-6M tracking degrades on slopes ≥ 8° (27–74 % error); rough ≤ 2.5 cm easy for both
- [ ] **2.9** ONNX export + `gurdy_policy` ROS node + Gazebo deployment — AC: ONNX == torch (1e-5);
  walks in Gazebo without retraining; no fall in the 2-min protocol; tracking + sim-to-sim gap reported
  — prep done (2026-11-15, `dr_energy_ft` 6M, Gazebo NOT run yet): `rl/tools/export_onnx.py` →
  `models/dr_energy_ft_6M.{onnx,json,_parity.npz}`; ONNX vs torch max diff 3e-7; `gurdy_core/policy_runtime.py`
  (obs/deadband/EMA, == env, tested); node `gurdy_policy` (stand pose on cmd/sensor timeout, `mode`
  param policy|tripod); MuJoCo deployment-path protocol: all segments 2–9 %, no fall. Gazebo run done 2026-11-16 (`docs/results/phase2_deploy_gazebo.md`): 0 falls both models, but target
  (within ~5 pts of MuJoCo) NOT met: ft-6M stalls on 4 segments, terrain_v1 60M within ±25 % (gap up to 22 pts); latency
  + stance-load/contact differences diagnosed; decision pending on closing the gap. Was: re-export with terrain_v1
- [ ] **2.10** Experiment A (rough terrain, RL vs tripod, MuJoCo + Gazebo) and Experiment B (one leg
  disabled) — AC: results tables in `docs/results/`
  - Experiment A: MuJoCo part written (`docs/results/exp_A_terrain.md`, from existing evals): RL beats the tripod on steps >= 3 cm and in tracking error everywhere;
    the tripod does NOT fall or fail on slopes <= 15 deg (so "RL success > tripod" holds only on steps). Gazebo part not done.
  - Experiment B: leg failure (limp/locked, random leg + time, actor blind, critic told) built + smoke-tested 2026-11-16; training `leg_failure` started, eval plan
    in `docs/results/exp_B_leg_failure.md` (run after training)

**AC**
- Training runs without freezing WSL (RAM stays under budget; numbers logged); resume from checkpoint works.
- Flat ground: velocity tracking error < 20 % over the command range; fall rate < 5 % over 100 eval episodes.
- Rough terrain: RL success rate higher than tripod gait (reported with numbers).
- Policy walks in Gazebo via `/cmd_vel` without retraining; tracking error reported for both sims.
- Results table for Experiments A and B in `docs/results/`.

---

## Phase 3 — Mapping + navigation
> Gazebo run 2026-11-16 (`docs/results/phase3_gazebo_results.md`): lidar SLAM with GT odom maps well (tripod 2.4 cm / policy 3.1 cm mean error, RTF 0.99); Nav2 with the policy deadlocks at start (MPPI first step 0.04 m/s < new 0.05 deadband); EKF-odometry map distorted despite accurate odometry (open). Checkboxes below not ticked until fixed.
- [~] Demo world with obstacles, rooms and coloured objects — `flat_lab` world written (generator + test); not yet run in Gazebo
- [~] Depth → laser scan (or 3D) for SLAM; slam_toolbox mapping (RTAB-Map optional) — code done (gz lidar + depth→scan, slam_toolbox params); **untested in sim** (`docs/results/phase3_test_plan.md`)
- [~] Nav2 configured for a legged base (velocity limits, footprint, controller tuning) driving `/cmd_vel` — MPPI/Omni params + launch done; **untested in sim**
- [~] Leg odometry + EKF (`odom_source:=ekf`) — code + offline MuJoCo drift test done (1–2 % of path with IMU yaw); **Gazebo GT-vs-EKF comparison not run** (test plan step 6)
- [ ] Works with both tripod gait and RL policy
- [x] Map save/load (`scripts/map_tools.sh`); `explore` behaviour: own frontier explorer (`frontier_explorer_node`), flat_lab from an empty map: 99.1 % (GT odom) / 98.5 % (EKF) of the interior mapped, returns to start — `docs/results/phase3_explore.md`

**AC**
- Builds a usable map of the demo world from teleop and from autonomous exploration. **Met for exploration (2026-11-16, flat_lab, 2 runs).**
- Nav2 reaches ≥ 8/10 goal poses without collision (reported for gait and policy).

---

## Phase 4 — Perception + voice
- [x] CUDA setup: GPU visible in Docker (`nvidia/cuda` image `nvidia-smi` OK, 2026-11-16); `vision` compose profile + `docker/vision` (CUDA 12.1, PyTorch, onnxruntime-gpu) written, **image not built yet**
- [ ] Object detection by name (YOLO small or small VLM), trained/run on local GPU, CPU/ONNX fallback; 3D position from depth
  - [x] prep code (2026-11-16, nothing run in Gazebo/GPU): 7 classes, synthetic dataset generator (Gazebo boundingbox_camera → YOLO labels + depth), `vision/train.py`, `vision/eval_detector.py`, `gurdy_core.vision` (decode/NMS/depth geometry/semantic map), `gurdy_perception` detector node (markers + JSON semantic map), `semantic_lab` world; tests 44 pass. Run commands: `docs/results/phase4_runbook.md`; license notes: `docs/vision_objects.md`
  - [ ] render dataset, train, evaluate mAP, test in semantic_lab
- [ ] `find_object(name)` / `go_to(object)` behaviours
- [ ] Gnani Prisma STT + Timbre TTS (key from env var only); offline Whisper fallback
- [ ] Replies always start with "Main ek AI assistant hoon"

**AC**
- Detects each demo-world object by name; precision/recall measured on a recorded set.
- "Laal dabba dhundo" style command finds and approaches the object in sim.
- Works with network off (Whisper fallback); STT latency measured for both paths.
- Detector runs on GPU and on the CPU/ONNX fallback; FPS reported for both.

---

## Phase 5 — Gurdy Brain (LLM post-training + assistant)

### 5.1 Core: tools, data, training
- [ ] Tool schema: walk, turn, go_to, find_object, explore, patrol, inspect, return_home, stop, pause/resume/cancel, schedule, query_map, query_status, query_memory, query_incidents, describe_view, report
- [ ] Synthetic dataset: Hindi/Hinglish/Kannada/English; noisy-STT variants; multi-step missions; multi-turn dialogues; ambiguous and unsafe requests
- [ ] Held-out test split (incl. "baayen"/"haan ye" style confusions)
- [ ] SFT (LoRA, PEFT/TRL) on Qwen3-0.6B locally (RTX 4050, 6 GB); Qwen3-1.7B with QLoRA 4-bit if it fits — Colab/Kaggle fallback
- [ ] DPO: prefer clarification over guessing; refuse unsafe actions; prefer tool results over invented facts
- [ ] GGUF quantization, Ollama Modelfile; latency on local GPU and CPU
- [ ] Hugging Face model card (with eval table)
- [ ] LangGraph agent calling ROS 2 tools

**AC**
- Fine-tuned model beats keyword parser and base model on tool-call exact match, argument accuracy and direction errors (numbers in model card).
- Runs offline via Ollama on GPU and CPU-only; latency per command reported for both.

### 5.2 Talking
- [ ] English/Hindi/Hinglish text + voice (Gnani STT/TTS, local Whisper + local TTS fallback); replies start with "Main ek AI assistant hoon" where required
- [ ] Multi-turn memory (pronouns/ellipsis: "aur wahan se left" , "wapas jao")
- [ ] Clarification questions on low confidence/ambiguity
- [ ] Confirmation step for risky actions (long trips, rough terrain, leaving mapped area)

**AC**
- Multi-turn test set (≥ 20 dialogues) ≥ 90 % correct reference resolution.
- Clarification asked on ≥ 90 % of ambiguous prompts; unnecessary clarification on clear prompts ≤ 10 %.
- Risky actions never execute without confirmation (0 violations in test set).
- Works with network off (Whisper fallback); STT/TTS latency measured for both paths.

### 5.3 Spatial awareness (tool-grounded)
- [ ] Answers "what's nearby / where is X / left-right-behind of Y / which room / how far / what's blocking" via semantic map, costmap and TF tools
- [ ] Every spatial claim traceable to a tool call (logged); "I don't know / not seen yet" when the tool has no answer

**AC**
- Spatial QA set (≥ 30 questions in a sim world with ground truth): answer accuracy and distance error reported.
- Hallucination check: 0 objects/places stated that are not in the tool output (reported over the whole set).

### 5.4 Proactive alerts
- [ ] Event → speech/text alerts: path blocked, new object seen, stuck, fall, leg failure detected, low battery, map changed
- [ ] Rate limiting/dedup so alerts are not spammed; severity levels

**AC**
- Each alert type triggered in sim by a scripted cause; detection-to-alert latency measured.
- False-alert rate over a 10-minute normal run reported (target 0 repeats of the same alert within the cooldown).

### 5.5 Vision
- [ ] "What do you see?" via a small VLM that fits 6 GB VRAM (e.g. ~2B class, 4-bit); one GPU workload at a time (unload LLM/detector as needed)
- [ ] Change detection vs earlier visits to the same place (objects added/removed/moved)
- [ ] Optional: OCR for signs/labels

**AC**
- VLM VRAM and latency measured; answers on a ≥ 30-image sim test set scored against ground truth.
- Change detection: precision/recall on scripted scene changes reported.

### 5.6 Tasks and missions
- [ ] go to, find, explore, patrol, inspect, return home, multi-step missions
- [ ] Pause / resume / cancel mid-mission; schedules ("5 minute baad patrol karo")

**AC**
- Spoken multi-step mission ("kitchen jao, red box dhundo, wapas aao") executed end-to-end in sim.
- Pause/resume/cancel work in every mission state (state-machine test).
- Scheduled task fires within ±2 s of its time.

### 5.7 Memory
- [ ] Last-seen objects and places persist across sessions (SQLite/JSON)
- [ ] Mission history; Incident Graph queries (Phase 6 hook, stubbed until then)

**AC**
- After a restart, "red box last kahan dekha tha?" answers correctly from memory with timestamp.
- Memory answers carry their age; stale entries flagged.

### 5.8 Self-awareness
- [ ] Status: battery, leg health, mode, speed, current task
- [ ] Explains decisions ("why did you stop?") from logged events/state, not invention

**AC**
- Status values match telemetry exactly (test).
- "Why" questions (≥ 20 cases) cite a real logged cause; unsupported cause = fail.

### 5.9 Safety
- [ ] Voice e-stop ("ruko"/"stop"/"ro-ko" noisy variants) bypasses the LLM (keyword fast-path) and stops the robot
- [ ] No-go zones (set by voice, enforced in costmap)
- [ ] Refuses unsafe commands (stairs/edge, humans too close, outside zone)

**AC**
- E-stop: command-to-zero-velocity latency measured (target < 0.5 s sim time), ≥ 98 % recall on STT-variant set, works with LLM stopped.
- Unsafe-request refusal ≥ 95 %, with low over-refusal on safe commands (both reported).
- Robot never enters a no-go zone in 10 navigation runs.

### 5.10 Reports
- [ ] End-of-mission summary: duration, distance, map snapshot, object photos with positions, alerts raised
- [ ] Spoken short version + saved HTML/Markdown report

**AC**
- Report generated for each demo mission; all listed objects match the semantic map; photos present.

### 5.11 Evaluation
- [ ] ~100-command test set (English + Hindi/Hinglish), incl. noisy STT, multi-step, ambiguous, unsafe
- [ ] Metrics: task success rate, tool-call exact match, latency (p50/p95), hallucination checks, refusal/over-refusal, clarification rate
- [ ] Compare keyword parser vs base vs SFT vs SFT+DPO

**AC**
- One results table (docs/results/phase5_eval.md) with all four systems; SFT+DPO best on success rate and hallucination rate, or the gap is reported honestly.
- End-to-end demo video: spoken mission, alert, spatial question, report.

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

---

## Phase 8 — Air-ground team (planned, not started)
A simulated drone scouts from above, maps, detects objects and sends goals to one or more hexapods;
the LLM brain commands the whole team through tools. Builds on Phases 3–7 (map, Nav2, YOLO, Brain,
Incident Graph, mission control).

- [ ] **8.1** Drone in Gazebo Harmonic: multicopter model (gz `MulticopterMotorModel` + IMU/baro/GPS),
  PX4 SITL (or a PX4-style ROS 2 flight controller if PX4 doesn't fit the RAM budget) via the
  uXRCE-DDS bridge; downward RGB-D camera ≤ 320×240. Measure RAM/RTF of drone alone and drone +
  1 hexapod before going further (RAM rule: 8 GB WSL, one heavy workload — may require headless).
- [ ] **8.2** Offboard control from ROS 2: take-off, hold, waypoint, land; `scan_area(polygon, altitude)`
  lawn-mower coverage pattern
- [ ] **8.3** Aerial mapping: top-down occupancy/elevation map from the drone camera, fused with (or
  seeding) the hexapod's slam_toolbox map; shared `map` frame between robots (ground-truth alignment
  first, then marker/feature-based)
- [ ] **8.4** Aerial detection: Phase 4 YOLO on the drone camera → object positions in the map frame
- [ ] **8.5** Team interfaces: namespaced robots (`/drone`, `/hexapod_1`, …), `go_to` / `inspect` actions,
  team state topic; hexapods keep speaking only `/cmd_vel` + Nav2 below
- [ ] **8.6** LLM brain tools for the team: `scan_area`, `go_to(robot, target)`, `inspect(object)`,
  `report`; the Phase 5 agent plans multi-robot missions ("drone, find the red box; nearest hexapod,
  inspect it")
- [ ] **8.7** Multiple hexapods + 1 drone: task allocation (greedy/auction baseline, then optional
  LLM-assisted), collision-free goal assignment
- [ ] **8.8** Telemetry from all robots into the Incident Graph; mission control shows the team
- [ ] **8.9** Worlds: rubble field and farm (rows, crops, obstacles), procedurally generated

**AC**
- Measured RAM/RTF for drone + 1 and + N hexapods (within the 8 GB WSL budget, reported).
- Drone covers a 20 × 20 m area and finds each target object (precision/recall reported); the assigned
  hexapod reaches each found object (success rate over ≥ 10 missions).
- A spoken/typed multi-step team mission is executed end-to-end in sim.
- 2+ hexapods + 1 drone: allocation completes all tasks without collisions in ≥ 8/10 runs.
- Injected faults on drone and hexapods appear as separate incidents with correct root causes.

---

## Phase 9 — Research experiment: hybrid walk + fly robot (planned, not started)
Can one robot both walk and hop/fly? Key constraint: **thrust-to-weight ≥ 2** (hover at ≤ 50 % thrust).

- [ ] **9.1** Design study (in `robot.yaml` style, single source of truth): (a) lighter legs with fewer DOF
  (e.g. 2-DOF legs) + separate rotors, or (b) legs that double as rotor arms (rotor at the hip/femur,
  legs fold for flight). Mass budget incl. motors, props, battery; thrust-to-weight, flight time and
  hover power estimated from component data, then **measured in sim**
- [ ] **9.2** Thrust model in MuJoCo (rotor force/torque on sites, motor lag, yaw torque) and in Gazebo
  (`MulticopterMotorModel`); parity test between the two like Phase 1 (hover thrust, step response)
- [ ] **9.3** RL skills: take-off from legs, stable hover, landing on legs (touchdown velocity limit),
  flight "hops" over obstacles that walking cannot cross; reuse the Phase 2 pipeline (DR, asymmetric critic)
- [ ] **9.4** Mode-switch policy: when to fly vs walk (energy/time cost model: walking CoT vs flight
  power), obstacle-triggered hops; compare against a rule-based switch
- [ ] **9.5** Experiment write-up: energy per metre walking vs flying, success rate over obstacle courses,
  failure modes

**AC**
- Design meets thrust-to-weight ≥ 2 in sim (measured), with mass and power budget documented.
- Take-off → hop over an obstacle taller than leg clearance → landing on legs: success rate over ≥ 50
  sim trials, landing touchdown speed reported.
- Mode-switch policy beats walk-only and fly-only on an obstacle course in time or energy (numbers).
