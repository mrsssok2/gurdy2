# Phase 5.6 live test — fine-tuned brain controlling the robot in Gazebo (2026-11-18)

Stack (one container `gurdy_agent`, headless, flat_lab): Gazebo + policy_node (terrain_dr2 15M) + Nav2 (localization on `flat_lab_tripod_gt`, GT odom) + scan self-filter +
detector_node (det_v2_s, CPU in the sim image) + semantic map + `frontier_explorer_node serve_action:=true` + `gurdy_safety` monitors (battery SIM / leg health / alerts) +
agent on `RosBackend` (`brain/ros/run_missions.py`) with Ollama `gurdy-sft-400` (unconstrained, compact prompt). No training running. Start it yourself: `scripts/start_agent_chat.sh`.

## 9 typed missions (fresh stack, one run; sim seconds; places file = a copy, so no-go zones do not dirty the repo)
| # | Typed command | Calls the model planned | Result | Time (sim) | What happened |
|---|---|---|---|---|---|
| 1 | go to room B | go_to(place) | **success** | 91 s | arrived 0.1 m from goal; 4 new objects announced on the way |
| 2 | what is near you | whats_nearby | **success** | 0.9 s | green/yellow box with distance + side, 5 lidar sectors; all facts from the tool result |
| 3 | find the red box | find | **success** (from memory, red box seen in #1) | 1.2 s | no motion needed. Live search from an EMPTY map was a separate run, see below |
| 4 | where is the fire extinguisher | where_is | **correct "not seen"** | 0.3 s | flat_lab has no extinguisher (semantic_lab would) |
| 5 | find the red box and then go home | find, return_home | **success** | 97.8 s | arrived 0.2 m from home |
| 6 | go to room B → "stop" 12 s later → why did you stop | go_to / stop / why | **success** (stop) / weak reply (why) | stop at 15 s | robot stopped at (1.30, −0.09) at once; "why" answered with the last 3 events incl. old ones ("find finished: done. return_home finished: reached. go_to finished: cancelled.") |
| 7 | patrol room A and room B | patrol | **success** | 119 s | both rooms visited; reply only "patrol done" (no per-room summary) |
| 8 | go to the box (ambiguous) | go_to(class box) → clarification | **asked** | 2.4 s | listed the 5 known boxes with rooms and asked which one; no motion |
| 9 | go home; mark room B as a no-go zone; go to room B | return_home, no_go_zone, go_to | **refused as intended** | go home 156 s (see issues) | "That is inside or behind a no-go zone." No motion. An earlier variant ("walk forward 3 meters through the doorway") made the model emit an invalid argument (`distance_m`); the validator blocked it, the user saw the raw message `args: unknown argument 'distance_m'` |

**Live find (separate run, empty semantic map, robot at home):** "find the red box" → "Found the red box in room a, 1.5 metres ahead." after **68 s**. Before the fix below the same command
failed after 11 s (`find stopped: timeout`).

Per-turn LLM latency 0.4–1.0 s (first call after idle 2.2 s); 1 LLM call per command; 0 grounding rejections visible; every motion command ended with a `task_finished` / `task_failed` event.

## Bugs found and fixed
1. **`find` gave up after 11 s on a mapped area** (localization mode): the explorer ends at once with "no frontiers left", `find` had no further steps → `timeout`. Fix (`RosBackend.find`): after the explore step, visit the rooms (optional nav steps, `until` stops as soon as the object appears). Verified live (68 s). No unit test: `FakePort` only reveals objects while the explore action runs.
2. `run_text` used the JSON-schema constrained decoder for the fine-tuned model (13/100 in the 5.5 eval): now unconstrained for `gurdy-sft*` models.
3. ROS image had no `langgraph` (agent and rclpy must share a container): added to `docker/ros/Dockerfile` (image rebuilt).
4. `ros2 launch` read the terminal's stdin (would eat typed chat input): start commands use `< /dev/null`.
5. rclpy spin-thread traceback at exit of the chat: `os._exit(0)` after flush.
6. Helper: `run_missions.py` (scripted typed missions), `start_agent_chat.sh`.

## Leg-health calibration (`scripts/leg_health_record.py`, `leg_health_drive.py`, `leg_health_analyze.py`)
Healthy walking in flat_lab Gazebo (forward / backward / strafe / turns / Nav2 trips, 560 s sim, 52k samples, 303 s with ≥ 3 legs moving):
EMA tracking error per leg: mean 0.033–0.040, p99 0.046–0.059, **max 0.090 rad** (rf). The monitor's absolute threshold is 0.15 rad → 1.7× margin; **0 false alarms** with the default (also at 0.12 and 0.10).
The ratio rule (3× the median of the other legs) is not the limiting condition (ratios up to 13 occur in healthy data because the median is tiny); the absolute threshold carries the protection.
**Kept 0.15 rad / 3× / 2 s** (comment in `LegHealthConfig`). Synthetic injection (measured joints of one leg frozen = locked leg, 6 legs × 3 injection times): 18/18 detected, median 84 s of recording
(waiting time while the robot stands counts; detection needs the leg to move away from the frozen pose). **Limits:** flat ground only (rough terrain will have larger errors → recalibrate with terrain data before trusting alerts there);
no real fault injected in Gazebo (frozen-measurement synthetic only).

## Resources (whole session ≈ 55 min, sampled every 5 s: `data/phase5/p56/resources.csv`)
Peak **WSL RAM used 4.37 GiB** of 8 GiB (sim container 1.24 GiB); **VRAM peak 2.2 GiB** (Ollama gurdy-sft-400 ≈ 1.8 GiB resident; detector runs on CPU here); **GPU temperature peak 62 °C**,
GPU utilisation spikes to 100 % during LLM turns. CPU temperature is not readable inside WSL. Gazebo server ≈ 1.4 cores.

## Bigger issues (not fixed)
1. **Idle ROS nodes burn CPU on sim time** (/clock ≈ 400 Hz): `frontier_explorer_node` 94 % of a core while idle, `alert_monitor_node` 54 %, `battery_sim_node` 40 %, detector 41 %. Fix: slower /clock for these nodes or wall-time timers.
2. **`why` replies list old events** ("find finished: done. return_home finished: reached. go_to finished: cancelled."); the answer should be the latest one. Data layer or training-data issue.
3. **Patrol reply** is only "patrol done" (no places visited / new / missing); the template needs the patrol result.
4. **Invalid-argument plans reach the user as raw validator text** (`args: unknown argument 'distance_m'`); should become a clarification or a re-plan.
5. **Return-home retry near the goal**: mission 9 "go home" took 156 s and re-planned "Going home, 0.6 metres" (goal tolerance retry); mission 1 reaches room B in 91 s.
6. Still missing: Nav2 keepout filter for zones, `cmd_vel` gate for e-stop, VLM, persistent memory (5.8). The e-stop path was not run live this session (unit tests only).
7. The room sweep in `find` visits room centres only (no full coverage); fine for flat_lab.
8. `no_go_zone add` writes into the places file (repo copy changes); the live runs use a copy.
9. flat_lab has only boxes; `where is the fire extinguisher` / class variety needs `semantic_lab` + a places file for it.
10. Detector runs on CPU in the sim image (GPU detector only in the vision image).


## Live voice + Gazebo test (2026-11-18, typed + voice, `gurdy-sft-400`)
Passed: `go_to` (arrived within 0.1 m), honest object report, go home, stop in 0.3 s.

### Open issue: "the other box" (logged, NOT fixed)
"Walk towards the other box" went to the red box that had just been discussed. Expected: "the other X" excludes the **last-mentioned X**;
if several remain, ask a question ("Which one: the blue box or the yellow box?"); if exactly one remains, go to it.
Where it probably belongs: reference resolution (dialogue state: last-mentioned object per class) before planning, plus training / test cases
(`testset_v1_1` has no "the other ..." case). Not covered by the final eval (5.11) headline rate; add a test case when it is fixed.
