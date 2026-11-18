#!/bin/bash
# Phase 5.11 final Gazebo eval: the scripted mission suite (brain/eval/final_suite.py) through the real agent, typed input, model gurdy-sft-400.
# ONE fresh Gazebo + Nav2 stack per mission (clean world, battery, faults, zones, memory), one stack at a time, per-mission time limit,
# automatic restart + retry (max ATTEMPTS) when the stack fails to start, Gazebo hangs (no /clock progress 90 s) or the container dies.
#   nohup scripts/p5_final_eval.sh > /tmp/p5_final_eval.out 2>&1 &            # whole suite (about 3-4 h)
#   ONLY=nav_room_b,stop_mid_walk scripts/p5_final_eval.sh                    # a subset
#   RUN_DIR=brain/runs/final_eval/<ts> scripts/p5_final_eval.sh               # resume: missions with a valid result are skipped
# Results: brain/runs/final_eval/<timestamp>/{meta.json,status.txt,run.log,missions/<id>.json,logs/,report.md,summary.json}; at the end docs/results/phase5_final_eval.md.
set -uo pipefail
cd "$(dirname "$0")/.."
export MODEL=${MODEL:-gurdy-sft-400}; PORT=11435; ATTEMPTS=${ATTEMPTS:-3}
TS=$(date -u +%Y%m%d_%H%M%S); RUN=${RUN_DIR:-brain/runs/final_eval/$TS}
mkdir -p "$RUN/missions" "$RUN/logs" "$RUN/attempts"
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$RUN/run.log"; }
IDS=$(python3 -c "
import os
from brain.eval.final_suite import SUITE
only=[x for x in os.environ.get('ONLY','').split(',') if x]
print(' '.join(m['id'] for m in SUITE if not only or m['id'] in only))")
[ -n "$IDS" ] || { echo "no missions selected"; exit 1; }
N=$(echo $IDS | wc -w)

# ---- preflight
if [ -n "$(docker ps -q)" ]; then log "other containers are running (docker ps): stop them first (one heavy workload at a time)"; docker ps --format '{{.Names}}' | tee -a "$RUN/run.log"; exit 1; fi
for d in "$HOME" /mnt/c; do
  free=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc 0-9)
  [ -n "$free" ] && [ "$free" -lt 5 ] && { log "less than 5 GB free on $d ($free GB): refusing to start (a full disk crashed WSL before)"; exit 1; }
done
if ! curl -sf "localhost:$PORT/api/tags" >/dev/null; then
  log "starting Ollama on :$PORT"
  OLLAMA_HOST=0.0.0.0:$PORT OLLAMA_MODELS=$HOME/ollama_models LD_LIBRARY_PATH=/usr/lib/wsl/lib nohup "$HOME/ollama/bin/ollama" serve > /tmp/ollama_gurdy.log 2>&1 &
  for _ in $(seq 30); do curl -sf "localhost:$PORT/api/tags" >/dev/null && break; sleep 1; done
fi
curl -s "localhost:$PORT/api/tags" | grep -q "\"$MODEL" || { log "model $MODEL not in Ollama"; exit 1; }
export OLLAMA_URL=http://$(hostname -I | cut -d' ' -f1):$PORT
cat > "$RUN/meta.json" <<JSON
{"started": "$(date -u +%FT%TZ)", "commit": "$(git rev-parse --short HEAD)", "dirty": $(git status --porcelain | grep -q . && echo true || echo false), "model": "$MODEL", "missions": $N, "attempts_max": $ATTEMPTS, "host": "$(hostname)"}
JSON
trap 'docker rm -f gurdy_final_eval >/dev/null 2>&1; log "interrupted / exiting"' EXIT
log "final eval: $N missions, model $MODEL, run dir $RUN"

result_status() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status','?'))" "$1" 2>/dev/null || echo "unreadable"; }
limit_of() { python3 -c "from brain.eval.final_suite import BY_ID;print(BY_ID['$1']['limit_s'])"; }

run_one() {   # $1 = mission id; returns 0 when a valid result file exists
  local id=$1 limit attempt rc st out="$RUN/missions/$1.json"
  limit=$(limit_of "$id")
  if [ -f "$out" ] && [[ "$(result_status "$out")" =~ ^(ok|timeout)$ ]]; then log "$id: valid result exists, skipped"; return 0; fi
  echo '{"attempts": 0, "log": []}' > "$RUN/attempts/$id.json"
  for attempt in $(seq "$ATTEMPTS"); do
    rm -f "$out"
    docker rm -f gurdy_final_eval >/dev/null 2>&1
    curl -s "localhost:$PORT/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":\"30m\"}" >/dev/null   # model resident before the sim takes RAM
    log "$id: attempt $attempt/$ATTEMPTS (limit ${limit}s sim)"
    timeout -k 30 $((1200 + limit * 2)) docker compose run --rm -T --name gurdy_final_eval -e OLLAMA_URL -e MODEL -e ID="$id" -e OUT="$out" sim bash -c '
      cd ~/gurdy2 && source ros2_ws/install/setup.bash && export PYTHONPATH=$HOME/gurdy2:$HOME/gurdy2/gurdy_core:$HOME/gurdy2/sim_mujoco:$PYTHONPATH
      source scripts/lib_nav_start.sh; mkdir -p /tmp/chat; cp maps/flat_lab.places.yaml /tmp/chat/places.yaml
      trap "stop_nav_stack" EXIT
      start_nav_stack /tmp/chat gui:=false world:=flat_lab perception:=true estop_gate:=true target_frame:=map nav:=localization map:=flat_lab_tripod_gt navigation:=true controller:=policy odom_source:=ground_truth > /tmp/chat/start.log 2>&1 < /dev/null \
        || { echo "Nav2 never became ready"; tail -20 /tmp/chat/launch.log; exit 5; }
      ros2 launch gurdy_safety monitors.launch.py odom_topic:=/odom > /tmp/chat/monitors.log 2>&1 < /dev/null &
      ros2 run gurdy_navigation frontier_explorer_node --ros-args -p serve_action:=true -p clock_topic:=/clock_slow -p log_path:=/tmp/chat/explore_log.json -p done_path:=/tmp/chat/explore.done > /tmp/chat/explorer.log 2>&1 < /dev/null &
      sleep 12
      python3 -m brain.ros.final_eval_runner --mission "$ID" --out "$OUT" --ollama "$MODEL" --places /tmp/chat/places.yaml --map-id flat_lab --odom-topic /odom --memory-db /tmp/chat/memory.db 2>&1 | grep --line-buffered -v "^terminate"
      exit ${PIPESTATUS[0]}
    ' > "$RUN/logs/$id.attempt$attempt.log" 2>&1
    rc=$?
    docker rm -f gurdy_final_eval >/dev/null 2>&1
    st=none; [ -f "$out" ] && st=$(result_status "$out")
    python3 - "$RUN/attempts/$id.json" "$attempt" "$rc" "$st" <<'PY'
import json, sys
p, a, rc, st = sys.argv[1:]
d = json.load(open(p)); d["attempts"] = int(a); d["log"].append({"attempt": int(a), "rc": int(rc), "status": st}); json.dump(d, open(p, "w"))
PY
    log "$id: attempt $attempt finished rc=$rc result=$st"
    if [[ "$st" =~ ^(ok|timeout)$ ]]; then return 0; fi
    sleep 5
  done
  python3 - "$out" "$id" "$RUN/logs/$id.attempt$ATTEMPTS.log" <<'PY'
import json, sys
out, mid, lg = sys.argv[1:]
try:
    tail = open(lg, errors="replace").read()[-600:]
except OSError:
    tail = ""
json.dump({"mission": mid, "status": "infra_failed", "error": "no valid run in all attempts; last log: " + tail, "poses": [], "messages": [], "turns": []}, open(out, "w"))
PY
  return 1
}

i=0
for id in $IDS; do
  i=$((i + 1))
  echo "$i/$N running $id ($(date -u +%H:%M:%S))" > "$RUN/status.txt"
  run_one "$id" && log "$id: done" || log "$id: INFRASTRUCTURE FAILURE (no valid result after $ATTEMPTS attempts)"
  python3 -m brain.eval.final_report "$RUN" --note "PARTIAL: $i of $N missions finished" 2>&1 | tail -1 | tee -a "$RUN/run.log"
done
python3 -m brain.eval.final_report "$RUN" --doc docs/results/phase5_final_eval.md 2>&1 | tee -a "$RUN/run.log"
echo "DONE $N/$N ($(date -u +%H:%M:%S))" > "$RUN/status.txt"
log "finished; report: docs/results/phase5_final_eval.md and $RUN/report.md"
