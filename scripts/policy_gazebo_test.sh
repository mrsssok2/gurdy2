#!/bin/bash
# Step 2.9 deployment test: RL policy in Gazebo (headless, IMU on) vs the SAME policy in MuJoCo
# through the deployment path (scripts/policy_track_mujoco.py). ~2 min protocol each.
# Usage (inside the sim container, nothing else running): scripts/policy_gazebo_test.sh OUT_DIR [model.onnx]
set -eo pipefail
OUT=${1:?out dir}; MODEL=${2:-$HOME/gurdy2/models/dr_energy_ft_6M.onnx}; mkdir -p "$OUT"; cd ~/gurdy2
set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:${PYTHONPATH:-}
ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=true controller:=policy model:="$MODEL" \
  > "$OUT/gazebo_launch_policy.log" 2>&1 &
LAUNCH=$!
for _ in $(seq 120); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" = 2 ] && break; sleep 1
done
sleep 2
python3 scripts/gz_tracking.py > "$OUT/gazebo_track.json"          # the shared 123 s /cmd_vel protocol
grep -E "policy tick|status" "$OUT/gazebo_launch_policy.log" | tail -5 || true
kill -TERM $LAUNCH 2>/dev/null || true
for _ in $(seq 20); do kill -0 $LAUNCH 2>/dev/null || break; sleep 0.5; done
pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|ros2 launch" 2>/dev/null || true
python3 scripts/policy_track_mujoco.py --model "$MODEL" > "$OUT/mujoco_track.json"
python3 - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]
gz, mj = (json.load(open(f"{o}/{s}_track.json")) for s in ("gazebo", "mujoco"))
print(f"RTF gazebo {gz['rtf']:.2f} | fell: gazebo {gz['fell']}, mujoco {mj['fell']} | min body z "
      f"{gz['min_body_z']:.3f} / {mj['min_body_z']:.3f} m | max tilt {gz['max_tilt_rad']:.3f} / {mj['max_tilt_rad']:.3f} rad\n")
print("| segment | command (vx, vy, wz) | gazebo measured | err % | mujoco measured | err % | mj vs gz (main axis) |")
print("|---|---|---|---|---|---|---|")
for a, b in zip(gz["segments"], mj["segments"]):
    k = max(range(3), key=lambda i: abs(a["cmd"][i]))
    rel = 100 * (b["measured"][k] - a["measured"][k]) / a["measured"][k]
    f = lambda v: "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"
    print(f"| {a['segment']} | {f(a['cmd'])} | {f(a['measured'])} | {a['main_axis_error_pct']:+.1f} | "
          f"{f(b['measured'])} | {b['main_axis_error_pct']:+.1f} | {rel:+.1f} |")
print(f"\nno fall + all within ±25 %: gazebo {gz['all_pass']} | mujoco {mj['all_pass']}")
PY
