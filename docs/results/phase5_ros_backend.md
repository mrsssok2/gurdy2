# Phase 5.6 prep — RosBackend (code only; nothing run in Gazebo)

`brain/ros/`: `backend.py` (`RosBackend`, same `RobotBackend` interface and result shapes as `MockBackend`), `port.py` (`RosPort`: what the backend
needs from ROS, plain Python types), `rclpy_port.py` (real adapter: rclpy node in a background thread; imports checked in the sim image, never run
against a live graph), `fake_port.py` (test double: fake topics / Nav2 / explore action servers), `relmotion.py` (closed-loop walk/turn), `run_text.py`
(type commands against the robot). Places/zones file: `maps/flat_lab.places.yaml`. Tests: `brain/tests/test_ros_backend.py` (35, no ROS).

Design: the backend is stepped by `tick()` (called inside `poll_events()`, so the agent loop drives it) and only reads time from the port (/clock).
Motion tools return `started` at once; the Nav2 goal is sent in the same call; the final result arrives as a `task_finished` / `task_failed` event.

## Tool -> ROS mapping

| Tool | ROS source | Status |
|---|---|---|
| `go_to`, `return_home` | place/object/point resolved here -> `compute_path_to_pose` check -> `navigate_to_pose`; object = nearest of 8 ring points at the standoff, goal yaw faces it | wired |
| `walk`, `turn` | `RelativeMotion` on TF pose, publishes `cmd_vel_smoothed` (min-speed lift + collision monitor stay in the path); stops on no progress for 8 s (`blocked`) | wired (the "NEW executor" of design 1.3, as library code; no separate node) |
| `find` | semantic map first; else explore until the object appears, cancel explore, optionally approach | **needs explore action** |
| `explore` | explore action | **GAP: no action yet** (`frontier_explorer_node` runs to completion at launch) -> `UNAVAILABLE` |
| `patrol` | sequence of `navigate_to_pose` goals over `maps/*.places.yaml`; a blocked place is recorded `reached: false` and the patrol goes on; new/missing/moved = semantic map diff start vs end | wired (a never-dropped track cannot become `missing`) |
| `inspect` | go to the object (standoff 0.6), face it, `confirmed` = track re-detected in the last 5 s | wired; `photo_path` = None |
| `stop` | cancel goal / explore, zero `cmd_vel_smoothed` | wired; works with no LLM |
| `mission_control` | pause = cancel goal, resume = re-send it (Nav2 has no pause); cancel | wired |
| `schedule` | backend timer on /clock | wired (lost on restart) |
| `where_is`, `relation`, `recall(object)`, `whats_nearby` (objects) | `semantic_map/objects` JSON | wired |
| `whats_nearby` (obstacles) | 8 sectors from `/scan` (the self-filtered one) | wired; sectors with a return only |
| `where_am_i` | TF `map -> base_link` + places file | wired; `localisation` is always `"ok"` (no covariance source yet) |
| `path_check` | `compute_path_to_pose`, `/map` cell of the goal (unknown = `leaves_mapped_area`), zones checked against the plan points | wired |
| `status` | speed (odom), `/policy/status`, IMU tilt, `/estop`, active alerts | wired; **battery and legs unknown** (below) |
| `why` | event log: backend task events + alert topic | wired |
| `recall(mission/place)`, `report` | in-memory mission log / places file | wired; lost on restart (SQLite memory is 5.8) |
| `describe_view` | detections seen in the last 1.5 s (detector output); `unverified: true` | partial: no VLM, no photo |
| `no_go_zone`, `save_place` | places file (`no_go_zones`, `places`), enforced by the backend on goals and Nav2 plans | partial: **no Nav2 keepout filter**, a plan that merely cuts a zone is refused, walk/turn ignore zones |
| `ask_user` | handled by the tool layer | n/a |
| events | `/gurdy/alerts` String JSON `{stamp,type,severity,text,data}` (interim); `fall` / `estop` alerts end the active task | topic **has no producer yet** (monitors = 5.7) |
| e-stop | `RosBackend.estop(bool)` (not a tool) publishes latched `/estop`, cancels the task | **gate node does not exist** (5.7): until then it only cancels + zeroes |

## Gaps (no ROS counterpart yet)
1. Explore action (`gurdy_interfaces/Explore`, wrapping `frontier_explorer_node`: start/cancel/feedback) — blocks `explore` and live `find`.
2. `battery_sim_node` -> `battery_pct` is `None`; the agent now treats unknown as "no battery rule fires" and says "battery unknown" (templates, state block, safety patched).
3. `leg_health_node` -> `legs` empty, `legs_known: false`; the failed-leg safety rule cannot fire.
4. `/gurdy/alerts` / `gurdy_interfaces/Event` producers (path_blocked monitor, stuck, fall, object_new/missing differ). The backend emits its own `path_blocked` on Nav2 abort.
5. `cmd_vel_gate_node` + `/estop` subscriber, Nav2 keepout filter for zones.
6. Localisation quality, VLM / photos, persistent memory (SQLite per `map_id`), `mapped_pct` needs `interior_bounds` for the map (None otherwise).
Assumptions to check live: Nav2 `compute_path_to_pose` server is up with `navigate_to_pose`; `/imu` orientation is the body orientation; `/scan` bearing 0 = robot forward;
GoalStatus codes 4/5/6 = succeeded/canceled/aborted; semantic JSON `last_seen` is /clock time (design assumes it, `detector_node` stamps from sim time).

## Found on the way
`MockBackend.return_home` calls `go_to`, so a failed leg REFUSES it (the rule exempts `return_home`); RosBackend avoids it (`_tool`). The mock bug is not fixed here.
Untested: the backend ran only against `FakePort`; the real adapter's action callbacks / TF / QoS are unrun.

## Gazebo test commands (later, when the GPU is free; one heavy workload at a time)
```bash
cd ~/projects/gurdy2 && docker ps        # must be empty
# 0. unit tests anywhere:  docker compose run --rm brain python3 -m pytest -q brain/tests/test_ros_backend.py
# 1. flat_lab + Nav2 localization on the saved GT map + detector (policy controller), headless. Arguments checked against sim.launch.py / phase4_semantic_run.sh.
docker compose run -d --rm --name gurdy_agent sim bash -c 'cd ~/gurdy2 && source ros2_ws/install/setup.bash && source scripts/lib_nav_start.sh && mkdir -p /tmp/p56 && \
  start_nav_stack /tmp/p56 gui:=false world:=flat_lab perception:=true target_frame:=map nav:=localization map:=flat_lab_tripod_gt navigation:=true \
  controller:=policy odom_source:=ground_truth && sleep infinity'
#    wait ~2 min:  docker exec gurdy_agent cat /tmp/p56/nav_ready.txt   (ends with "ready")
#    flat_lab objects are plain boxes / a cylinder, so the detector reports class `box` (colours red / blue / yellow; the cylinder becomes a box too).
# 2. smoke test of the adapter, no LLM (second terminal):
docker exec -it gurdy_agent /entrypoint.sh bash -c 'cd ~/gurdy2 && python3 - <<PY
import time
from brain.ros.rclpy_port import RclpyPort
from brain.ros import RosBackend
p = RclpyPort(map_id="flat_lab"); time.sleep(3)
b = RosBackend(p, "maps/flat_lab.places.yaml")
print("pose", p.pose(), "objects", len(p.semantic_objects() or []), "nav", p.nav_ready())
print(b.where_am_i()); print(b.status()); print(b.path_check(place="room b"))
print(b.walk("forward", 0.3))
for _ in range(100):
    time.sleep(0.3); b.tick()
    if b.task is None: break
print(b.events[-1])
print(b.go_to(place="room b"))
for _ in range(600):
    time.sleep(0.5); b.tick()
    if b.task is None: break
print(b.events[-1])
PY'
# 3. typed commands (keyword planner, or --ollama MODEL with OLLAMA_URL set):
docker exec -it gurdy_agent /entrypoint.sh bash -c 'cd ~/gurdy2 && python3 -m brain.ros.run_text --places maps/flat_lab.places.yaml --map-id flat_lab'
#    try: "walk forward half a metre", "turn left 90 degrees", "where is the red box", "go to room b", "come back", "stop" (mid-walk), "what is nearby"
# 4. stop:  docker rm -f gurdy_agent
```
Pass criteria (first run): pose + semantic objects arrive; `walk 0.3` moves 0.27-0.33 m; `turn left 90` ends within 5 deg; `go_to room b` returns `reached` with
final error < 0.25 m; `stop` mid-goal zeroes `cmd_vel` within 1 s (check `ros2 topic echo /cmd_vel`); `where_is` matches the detector's map. Record RTF and RAM.
`explore` / live `find` stay `UNAVAILABLE` until gap 1 is closed. Place names in `maps/flat_lab.places.yaml` match flat_lab only.
