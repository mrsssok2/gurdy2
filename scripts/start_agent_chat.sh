#!/bin/bash
# ONE command: Gazebo (headless) + Nav2 + policy + detector + explore + monitors + the fine-tuned brain, with a typed chat in this terminal.
#   scripts/start_agent_chat.sh            # flat_lab, model gurdy-sft-400
#   GUI=true scripts/start_agent_chat.sh   # with the Gazebo window (WSLg)
#   MODEL=qwen3:1.7b scripts/start_agent_chat.sh
#   MEMORY=false scripts/start_agent_chat.sh   # no persistent memory (default: on, file ~/.gurdy/memory.db, rows per map_id)
# Needs: nothing else running (docker ps empty). Starts this project's Ollama on :11435 if it is not up. Takes ~2.5 min until the prompt appears.
# Type e.g.: go to room B | find the red box | what is near you | stop | why did you stop | patrol room A and room B | quit
set -eo pipefail
cd "$(dirname "$0")/.."
export MODEL=${MODEL:-gurdy-sft-400} GUI=${GUI:-false} MEMORY=${MEMORY:-true}; PORT=11435
mkdir -p "$HOME/.gurdy"
if [ -n "$(docker ps -q)" ]; then echo "Other containers are running (docker ps). Stop them first (one heavy workload at a time)."; exit 1; fi
if ! curl -sf "localhost:$PORT/api/tags" >/dev/null; then
  echo "starting Ollama on :$PORT ..."
  OLLAMA_HOST=0.0.0.0:$PORT OLLAMA_MODELS=$HOME/ollama_models LD_LIBRARY_PATH=/usr/lib/wsl/lib nohup "$HOME/ollama/bin/ollama" serve > /tmp/ollama_gurdy.log 2>&1 &
  for _ in $(seq 30); do curl -sf "localhost:$PORT/api/tags" >/dev/null && break; sleep 1; done
fi
curl -s "localhost:$PORT/api/tags" | grep -q "\"$MODEL" || { echo "model $MODEL not in Ollama (see brain/tools/Modelfile.sft)"; exit 1; }
curl -s "localhost:$PORT/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":\"30m\"}" >/dev/null   # load the model before the sim competes for RAM
export OLLAMA_URL=http://$(hostname -I | cut -d' ' -f1):$PORT
docker compose run --rm $([ -t 0 ] || echo -T) --name gurdy_agent -e OLLAMA_URL -e MODEL -e GUI -e MEMORY -e DETECTOR_LOG -v "$HOME/.gurdy:/home/ubuntu/.gurdy" sim bash -c '
  cd ~/gurdy2 && source ros2_ws/install/setup.bash && export PYTHONPATH=$HOME/gurdy2:$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:$PYTHONPATH
  source scripts/lib_nav_start.sh; mkdir -p /tmp/chat; cp maps/flat_lab.places.yaml /tmp/chat/places.yaml   # zones the chat adds stay out of git
  trap "stop_nav_stack" EXIT
  echo "starting the simulation (about 2 minutes) ..."
  start_nav_stack /tmp/chat gui:=$GUI world:=flat_lab perception:=true ${DETECTOR_LOG:+detector_log:=$DETECTOR_LOG} estop_gate:=true target_frame:=map nav:=localization map:=flat_lab_tripod_gt navigation:=true controller:=policy odom_source:=ground_truth > /tmp/chat/start.log 2>&1 < /dev/null \
    || { echo "Nav2 never became ready (see /tmp/chat/launch.log)"; exit 1; }
  ros2 launch gurdy_safety monitors.launch.py odom_topic:=/odom > /tmp/chat/monitors.log 2>&1 < /dev/null &   # first: it starts the /clock_slow relay
  ros2 run gurdy_navigation frontier_explorer_node --ros-args -p serve_action:=true -p clock_topic:=/clock_slow -p log_path:=/tmp/chat/explore_log.json -p done_path:=/tmp/chat/explore.done > /tmp/chat/explorer.log 2>&1 < /dev/null &
  sleep 12
  MEMORY_ARGS=""; [ "$MEMORY" = true ] && MEMORY_ARGS="--memory-db /home/ubuntu/.gurdy/memory.db"
  python3 -m brain.ros.run_text --ollama "$MODEL" --places /tmp/chat/places.yaml --map-id flat_lab --odom-topic /odom ${MEMORY_ARGS} 2>&1 | grep --line-buffered -v "^terminate"
'
