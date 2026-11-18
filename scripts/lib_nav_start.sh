#!/bin/bash
# Shared start-up helper for the headless Nav2 runs (source it; needs ROS + workspace sourced, cwd = repo root).
#
#   start_nav_stack OUT_DIR  <sim.launch.py arguments...>      # sets LAUNCH (pid of the launch), returns 0 when Nav2 is ready
#   stop_nav_stack                                              # kills the launch and every leftover sim/nav process
#
# "Ready" = scripts/wait_nav_ready.py: navigate_to_pose, compute_path_to_pose and follow_path action servers + map->base_link TF.
# Known start-up flake (seen 2026-11-16): Nav2's lifecycle_manager sometimes never finishes bring-up because a DDS service
# response from controller_server/change_state is lost ("failed to send response ... timeout"); the manager then waits forever and
# re-sending RESET/STARTUP hangs too. Waiting longer does not help, so after NAV_READY_TIMEOUT s the whole stack is restarted
# (up to NAV_START_ATTEMPTS times, failed launch logs kept as launch_failed_N.log).
stop_nav_stack() {
  kill -TERM "$LAUNCH" 2>/dev/null || true
  sleep 4
  pkill -KILL -f "gz sim|robot_state_publisher|parameter_bridge|policy_node|gait_node|nav2|slam_toolbox|amcl|ros2 launch|measure_rtf|component_container|frontier_explorer|explore_monitor|lifecycle_manager|controller_server|planner_server|bt_navigator" 2>/dev/null || true
  sleep 2
}

start_nav_stack() {
  local out=$1; shift
  local attempts=${NAV_START_ATTEMPTS:-3} timeout=${NAV_READY_TIMEOUT:-90}
  : > "$out/nav_ready.txt"
  for attempt in $(seq "$attempts"); do
    ros2 launch gurdy_bringup sim.launch.py "$@" > "$out/launch.log" 2>&1 &
    LAUNCH=$!
    sleep 3; kill -0 "$LAUNCH" 2>/dev/null || { echo "ros2 launch exited at once:"; tail -5 "$out/launch.log"; return 1; }   # bad launch argument etc.: fail fast
    for _ in $(seq 180); do
      kill -0 "$LAUNCH" 2>/dev/null || break   # launch died (e.g. controller spawner timeout): go to the restart path
      [ "$(timeout 10 ros2 control list_controllers 2>/dev/null | grep -c " active")" = 2 ] && break; sleep 1
    done
    if kill -0 "$LAUNCH" 2>/dev/null && python3 scripts/wait_nav_ready.py --timeout "$timeout" >> "$out/nav_ready.txt" 2>&1; then
      echo "attempt $attempt/$attempts: ready" >> "$out/nav_ready.txt"
      cat "$out/nav_ready.txt"
      return 0
    fi
    echo "attempt $attempt/$attempts: NOT ready after ${timeout}s -> restarting the stack" | tee -a "$out/nav_ready.txt"
    cp "$out/launch.log" "$out/launch_failed_$attempt.log"
    stop_nav_stack
  done
  return 1
}
