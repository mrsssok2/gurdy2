# Phase 5.6 polish (2026-11-18)

Live runs: headless flat_lab, one container, `scripts/p56_polish_run.sh` (data: `data/phase5/p56b/o` = idle CPU, e-stop test, 9 missions, detector sightings; `p56b_m6/o` = mission 6 re-run).

| # | Item | Result (measured) |
|---|---|---|
| 1 | Detector false positive | **Cause confirmed:** det_v2_s labels the red box as `fire_extinguisher` (6 of ~165 sightings at the box position, mean score 0.6) and `toolbox` (11). In my 444-sighting log those never reached confirmation under the old rules, so the announced extinguisher from your run is **not reproduced**. Added a *dominance* rule (a track is hidden while another track within 0.35 m has ≥ 4× its hits) and raised `min_hits` 5→8, `min_mean_score` 0.5→0.6. Replay on both semantic_lab logs: 11/11 objects kept, false tracks 7→5 and 6→5, low-hit false tracks 1→0 (`vision/replay_semantic_map.py`). Unit test added. Not verified live in a run that actually produced the false track. |
| 2 | cmd_vel e-stop gate | `estop_gate_node` (`sim.launch.py estop_gate:=true`: /cmd_vel → /cmd_vel_safe). **Live PASS** while /cmd_vel kept at 0.12 m/s: walking 0.129 m/s; e-stop: /cmd_vel_safe exactly 0 (75 msgs), robot speed 0.000; release: 0.127 m/s again. The policy node also stops on /estop by itself, so this shows the gate output is zero, not that the gate alone stops the robot. Typed `!estop` / `!release` in the chat. |
| 3 | why / find | "why did you stop" → "You told me to stop." (live, mission 6; only the latest cause, user stop + blocked + finished variants unit-tested). `find` answered from memory now says where and how far: "Found the red box in room a, 3.8 metres behind-left." (was "Looking for the red box."). |
| 4 | Chat UI | Background thread prints async messages, erases and redraws the prompt with the typed text (`readline` buffer); also the thread now ticks the backend between commands; room names capitalised in display ("Room A"). Unit-tested; **not tested in a real terminal by me** (no tty here), please try it. |
| 5 | Idle CPU (30 s, robot standing, % of one core) | frontier explorer 94 → 25, alert monitor 54 → 19, battery 40 → 4. New `sim_clock_node` (/clock → /clock_slow 10 Hz) costs 40 %. Sum 188 → 88. "Before" = the 5.6 live test, not re-measured in the same session. |

## 9 live missions (fine-tuned `gurdy-sft-400`, sim time)
| # | Command | Result | Time |
|---|---|---|---|
| 1 | go to room B | arrived 0.1 m | 65 s |
| 2 | what is near you | correct, with distances | 1 s |
| 3 | find the red box | "Found the red box in room a, 3.8 metres behind-left." | 1.5 s |
| 4 | where is the fire extinguisher | correct "not seen" | 0.5 s |
| 5 | find the red box, then go home | found + arrived 0.1 m | 136 s |
| 6 | go to room B, stop, why | stopped at once; "You told me to stop." (first run: Nav2 "no path" at start, so re-run alone: OK) | 15 s |
| 7 | patrol room A and room B | done (reply still only "patrol done") | 103 s |
| 8 | go to the box | asked which of 6 | 2 s |
| 9 | go home, no-go room B, go to room B | refused as intended | 124 s |

Open: patrol reply text, raw validator text, the intermittent Nav2 "cannot find a path" at the first goal (1 of 2 runs), sim_clock_node's own CPU.
