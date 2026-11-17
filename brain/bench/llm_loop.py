#!/usr/bin/env python3
"""LLM load for the set-A test: one planning call every ~5 s for DURATION s (unique prompt prefix = no cache hit, worst case)."""
import sys, time, json
sys.argv_saved = sys.argv; sys.path.insert(0, ".")
import importlib.util
src = open("brain/bench/llm_bench.py").read().split("for model in sys.argv[1:]:")[0]
exec(src)
model, dur = sys.argv_saved[1], float(sys.argv_saved[2]); end = time.time() + dur; i = 0; ttfts = []; tps = []
post("/api/generate", {"model": model, "prompt": "hi", "stream": False, "think": False, "keep_alive": "30m", "options": {"num_ctx": 4096, "num_predict": 1}}).read()
while time.time() < end:
    t0 = time.time(); f, d = run(model, CMDS[i % len(CMDS)], nonce="" if (len(sys.argv_saved) > 3 and sys.argv_saved[3] == "shared") else f"[{t0}] "); ttfts.append(f); tps.append(d["eval_count"] / (d["eval_duration"] / 1e9)); i += 1
    time.sleep(max(0, 5 - (time.time() - t0)))
print(json.dumps(dict(model=model, calls=i, ttft_median=round(sorted(ttfts)[len(ttfts)//2], 3), ttft_max=round(max(ttfts), 3), decode_tps_median=round(sorted(tps)[len(tps)//2], 1))))
