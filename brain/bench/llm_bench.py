#!/usr/bin/env python3
"""Milestone 5.1: LLM serving numbers through the Ollama HTTP API (stdlib only).
    OLLAMA_HOST=127.0.0.1:11434 python3 brain/bench/llm_bench.py MODEL [MODEL ...]
Prompt = a system prompt carrying all 24 tool specs (JSON) + a short state block + one command, so TTFT includes the real prompt length.
Per model: cold load time, VRAM delta (nvidia-smi), then 5 warm runs (prompt cache off via a different last sentence): TTFT, prompt tokens/s, decode tokens/s."""
import json, subprocess, sys, time, urllib.request
sys.path.insert(0, ".")
from brain.tools.specs import TOOLS

HOST = "http://127.0.0.1:11434"

def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.split()[0])

def post(path, body, stream=False):
    req = urllib.request.Request(HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

tools = [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in TOOLS.values()]
SYSTEM = ("You are Gurdy, a hexapod robot. Answer ONLY with a JSON list of tool calls.\nTools:\n" + json.dumps(tools) +
          "\nState: pose (1.2, 0.4), battery 81 %, known places: kitchen, room_b.")
CMDS = ["go to the red box and then come back here", "explore the room and tell me every object you find", "turn left ninety degrees then walk forward two metres",
        "stop what you are doing and wait", "find the fire extinguisher and stay next to it"]

def run(model, cmd, n_predict=120, nonce=""):
    t0 = time.perf_counter(); first = None; last = {}
    r = post("/api/generate", {"model": model, "system": nonce + SYSTEM, "prompt": cmd, "stream": True, "think": False, "keep_alive": "10m",
                               "options": {"num_ctx": 4096, "num_predict": n_predict, "temperature": 0}})
    for line in r:
        d = json.loads(line)
        if first is None and d.get("response"): first = time.perf_counter() - t0
        if d.get("done"): last = d
    return first, last

for model in sys.argv[1:]:
    post("/api/generate", {"model": model, "keep_alive": 0}).read(); time.sleep(2)
    base = vram(); t0 = time.perf_counter()
    post("/api/generate", {"model": model, "prompt": "hi", "stream": False, "think": False, "keep_alive": "10m", "options": {"num_ctx": 4096, "num_predict": 1}}).read()
    load = time.perf_counter() - t0; v = vram() - base
    rows = []
    for i, c in enumerate(CMDS):
        first, d = run(model, c, nonce=f"[{time.time()}-{i}] ")   # unique prefix -> no prompt-cache hit (worst case)
        rows.append(dict(ttft=first, prompt_tok=d["prompt_eval_count"], pp_tps=d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9),
                         out_tok=d["eval_count"], tg_tps=d["eval_count"] / (d["eval_duration"] / 1e9)))
    warm = [run(model, c)[0] for c in CMDS]   # same system prompt as the previous run -> prefix cache hit (normal turn)
    med = lambda k: round(sorted(r[k] for r in rows)[len(rows) // 2], 3)
    print(json.dumps(dict(model=model, cold_load_s=round(load, 2), vram_mib_delta=v, ttft_cold_prefix_s=med("ttft"), ttft_cached_prefix_s=round(sorted(warm)[len(warm)//2], 3), prompt_tok=rows[0]["prompt_tok"],
                          prompt_tok_per_s=med("pp_tps"), decode_tok_per_s=med("tg_tps"), out_tok=[r["out_tok"] for r in rows], ttft_cold_all=[round(r["ttft"],2) for r in rows], ttft_cached_all=[round(w,3) for w in warm])), flush=True)
