# Source me (cwd = repo root). This project's Ollama listens on :11435 (a Windows Ollama owns :11434).
#   ollama_prepare MODEL            start Ollama if needed (0.0.0.0, so containers can reach it), check the EXACT model exists, load it,
#                                   find an address the CONTAINERS can reach (tests each candidate from a throwaway container) and export OLLAMA_URL.
#                                   Returns 1 with a diagnosis otherwise.
#   ollama_verify_in CONTAINER MODEL   run one real prompt through the agent's own LLM client INSIDE the running container (same env, same network);
#                                      returns 1 with the error otherwise. Call it before telling the user the stack is ready.
OLLAMA_PORT=${OLLAMA_PORT:-11435}

_ollama_up() { curl -sf "localhost:$OLLAMA_PORT/api/tags" >/dev/null; }

_ollama_start() {
  echo "starting Ollama on :$OLLAMA_PORT (listening on all addresses) ..."
  OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT OLLAMA_MODELS=$HOME/ollama_models LD_LIBRARY_PATH=/usr/lib/wsl/lib nohup "$HOME/ollama/bin/ollama" serve > /tmp/ollama_gurdy.log 2>&1 &
  for _ in $(seq 40); do _ollama_up && return 0; sleep 1; done
  echo "Ollama did not start (see /tmp/ollama_gurdy.log)"; return 1
}

# Can a container fetch /api/tags from this address? (throwaway container on the default network, 4 s limit)
_reach_from_container() {
  local url=$1 img=gurdy2/brain:cpu
  docker image inspect "$img" >/dev/null 2>&1 || img=gurdy2/ros:jazzy
  docker run --rm --add-host host.docker.internal:host-gateway --entrypoint python3 "$img" -c \
    "import urllib.request,sys; urllib.request.urlopen('$url/api/tags', timeout=4); print('ok')" 2>/dev/null | grep -q ok
}

_pick_url() {
  local c candidates=() a
  for a in $(hostname -I); do candidates+=("$a"); done
  candidates+=(host.docker.internal 172.17.0.1)
  for c in "${candidates[@]}"; do
    if _reach_from_container "http://$c:$OLLAMA_PORT"; then OLLAMA_URL="http://$c:$OLLAMA_PORT"; export OLLAMA_URL; return 0; fi
    OLLAMA_TRIED="${OLLAMA_TRIED:-} $c"
  done
  return 1
}

ollama_prepare() {
  local model=$1
  _ollama_up || _ollama_start || return 1
  # the EXACT model name (an Ollama tag list can hold gurdy-sft-v2, gurdy-sft-400, ...: a prefix match would pick the wrong one)
  curl -s "localhost:$OLLAMA_PORT/api/tags" | MODEL="$model" python3 -c '
import json, os, sys
m = os.environ["MODEL"]; names = [x["name"] for x in json.load(sys.stdin).get("models", [])]
sys.exit(0 if (m in names or m + ":latest" in names) else 1)' \
    || { echo "model '$model' is not in this Ollama. Installed: $(curl -s "localhost:$OLLAMA_PORT/api/tags" | python3 -c 'import sys,json;print(", ".join(x["name"] for x in json.load(sys.stdin)["models"]))')"
         echo "(MODEL=keyword runs without an LLM)"; return 1; }
  curl -s "localhost:$OLLAMA_PORT/api/generate" -d "{\"model\":\"$model\",\"keep_alive\":\"30m\"}" >/dev/null      # load it before the sim competes for RAM
  OLLAMA_TRIED=""
  if ! _pick_url; then
    # Ollama answers on localhost but no container can reach it: it was started listening on loopback only. It is this project's own port: restart it on all addresses.
    echo "Ollama answers on localhost but not from containers (tried:$OLLAMA_TRIED); restarting it on 0.0.0.0:$OLLAMA_PORT ..."
    local pid; pid=$(ss -ltnp 2>/dev/null | awk -v p=":$OLLAMA_PORT" '$4 ~ p"$" {print $6}' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null; sleep 2
    _ollama_start || return 1
    curl -s "localhost:$OLLAMA_PORT/api/generate" -d "{\"model\":\"$model\",\"keep_alive\":\"30m\"}" >/dev/null
    OLLAMA_TRIED=""
    _pick_url || { echo "No container can reach Ollama. Tried:$OLLAMA_TRIED"; echo "hostname -I: $(hostname -I)"; ss -ltn | grep ":$OLLAMA_PORT" ; return 1; }
  fi
  echo "LLM: model $model, Ollama reachable from containers at $OLLAMA_URL"
}

ollama_verify_in() {
  local ctr=$1 model=$2 out
  echo "checking the language model from inside $ctr (one real prompt) ..."
  out=$(docker exec -w /home/ubuntu/gurdy2 -e PYTHONPATH=/home/ubuntu/gurdy2:/home/ubuntu/gurdy2/gurdy_core "$ctr" timeout 170 python3 -c "
import os, sys
from brain.agent.ollama_llm import OllamaLLM
from brain.agent.llm import LLMUnavailable
m = OllamaLLM(os.environ['MODEL'], constrain=False, timeout=160)
try:
    t = m._chat('selftest', 'Reply with the single word OK.', None)
except LLMUnavailable as e:
    print('LLM_FAIL', e); sys.exit(1)
print('LLM_OK', repr(t.strip()[:60]), 'via', m.host)
" 2>&1) || { echo "LANGUAGE MODEL CHECK FAILED inside $ctr:"; echo "$out" | tail -5; return 1; }
  echo "$out" | tail -1 | grep -q LLM_OK || { echo "LANGUAGE MODEL CHECK FAILED inside $ctr:"; echo "$out" | tail -5; return 1; }
  echo "$out" | tail -1
}
