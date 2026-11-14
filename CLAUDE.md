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
- 16 GB RAM total; `.wslconfig` caps WSL at **8 GB RAM + 8 GB swap**. CPU only — **no NVIDIA GPU**.
- Gazebo GUI via **WSLg** (X11/Wayland sockets mounted into the container). Rendering is software
  or iGPU — keep cameras small (≤ 320x240) and frame rates modest.
- Free GPUs only on Google Colab / Kaggle (used for LLM training in Phase 5, via notebooks).

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
- ros2_control with `joint_state_broadcaster` + a joint-group position controller.
- MuJoCo + Gymnasium + Stable-Baselines3 PPO (CPU, `SubprocVecEnv`).
- SLAM: slam_toolbox (2D, light) first; RTAB-Map optional. Nav2.
- Perception: YOLO (small) or small VLM; Voice: Gnani Prisma STT / Timbre TTS, Whisper fallback.
  Spoken/text replies start with **"Main ek AI assistant hoon"**.
- LLM: Qwen3-0.6B/1.7B, LoRA via PEFT/TRL (SFT → DPO), GGUF → Ollama, LangGraph agent.
- Ops: Redpanda, Qdrant, HDBSCAN, PostgreSQL, Redis/Celery, FastAPI, LangGraph investigator.
- Web: Next.js/TypeScript, rosbridge, three.js. CI: GitHub Actions.

## Key design decisions
- **Single source of truth for robot geometry:** `robot.yaml` (dimensions, masses, limits, gains)
  is read by the xacro (URDF), the MJCF generator, and the IK library. Parity is checked by a test.
- **Shared pure-Python core** (`gurdy_core`): kinematics/IK and gait math, no ROS imports, used by
  the ROS gait node and the MuJoCo env alike. Unit-tested without ROS.
- Gait/policy nodes publish joint targets; everything above them speaks only `/cmd_vel`
  (`geometry_msgs/Twist`: linear.x, linear.y, angular.z), so Nav2/teleop/LLM are swappable.

## Repo layout
_Proposed — awaiting user approval. Update this section once approved._

## Common commands
_Filled in during Phase 1 (build image, start sim, colcon build, teleop, tests)._

## Current status
- Phase 0: CLAUDE.md + ROADMAP.md written; folder structure proposed; Phase 1 plan awaiting approval.
