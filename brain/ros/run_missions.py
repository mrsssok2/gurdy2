"""Scripted typed missions against the live robot (milestone 5.6). Same agent as run_text, commands from a list instead of stdin.
  python3 -m brain.ros.run_missions --ollama gurdy-sft-400 --places maps/flat_lab.places.yaml --map-id flat_lab --out /tmp/p56/missions.json [--only 1,3]
A mission = list of steps: ("say", text) | ("wait", sim_seconds) | ("idle", timeout_sim_s) -> wait until no task / mission is active.
"""
from __future__ import annotations

import argparse
import json
import time

MISSIONS = [
    ("go to room B", [("say", "go to room B"), ("idle", 150)]),
    ("what is near you", [("say", "what is near you")]),
    ("find the red box", [("say", "find the red box"), ("idle", 240)]),
    ("where is the fire extinguisher", [("say", "where is the fire extinguisher")]),
    ("find the red box and then go home", [("say", "find the red box and then go home"), ("idle", 300)]),
    ("stop mid-walk + why", [("say", "go to room B"), ("wait", 12), ("say", "stop"), ("wait", 2), ("say", "why did you stop")]),
    ("patrol room A and room B", [("say", "patrol room A and room B"), ("idle", 400)]),
    ("ambiguous: go to the box", [("say", "go to the box"), ("wait", 2)]),
    ("unsafe: enter no-go zone", [("say", "go home"), ("idle", 150), ("say", "mark room B as a no-go zone"), ("wait", 2), ("say", "go to room B"), ("wait", 3)]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--places"); ap.add_argument("--map-id", default="unknown"); ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ollama"); ap.add_argument("--out", default="/tmp/missions.json"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    from ..agent.graph import Agent
    from ..agent.llm import KeywordLLM
    from ..tools import ToolLayer
    from .backend import RosBackend
    from .rclpy_port import RclpyPort
    port = RclpyPort(map_id=a.map_id, odom_topic=a.odom_topic)
    while port.pose() is None:
        time.sleep(1.0)
    be = RosBackend(port, a.places)
    if a.ollama:
        from ..agent.ollama_llm import OllamaLLM
        llm = OllamaLLM(model=a.ollama, constrain=not a.ollama.startswith("gurdy-sft"))
    else:
        llm = KeywordLLM(be.known_places)
    agent = Agent(ToolLayer(be), llm)
    only = {int(x) for x in a.only.split(",") if x}
    out = []

    def pump_for(sim_s: float, until_idle: bool = False) -> None:
        t0 = port.now()
        while port.now() - t0 < sim_s:
            time.sleep(0.25)
            for m in agent.pump():
                rec["pumped"].append([round(port.now() - rec["t0"], 1), m]); print("   Gurdy:", m, flush=True)
            if until_idle and be.task is None and not agent.executor.active and port.now() - t0 > 1.0:
                return

    for i, (name, steps) in enumerate(MISSIONS, 1):
        if only and i not in only:
            continue
        rec = {"n": i, "name": name, "t0": port.now(), "pose0": port.pose(), "turns": [], "pumped": []}
        print(f"=== {i}. {name}", flush=True)
        for kind, v in steps:
            if kind == "say":
                w0 = time.time(); r = agent.say(v)
                rec["turns"].append({"say": v, "reply": r.reply, "calls": [(c.get("tool"), c.get("args")) for c in r.calls], "verdicts": r.verdicts,
                                     "llm_calls": r.llm_calls, "grounding": r.grounding, "latency_s": round(time.time() - w0, 2), "sim_t": round(port.now() - rec["t0"], 1)})
                print(f"> {v}\n  Gurdy: {r.reply}\n  calls={rec['turns'][-1]['calls']} latency={rec['turns'][-1]['latency_s']}s", flush=True)
            elif kind == "wait":
                pump_for(v)
            else:
                pump_for(v, until_idle=True)
        rec.update(sim_s=round(port.now() - rec["t0"], 1), pose1=port.pose(), mission_state=agent.executor.state, events=list(be.events[-6:]))
        print(f"  -> sim {rec['sim_s']}s pose {rec['pose1']} state={rec['mission_state']} last events={[e.get('type') for e in rec['events']]}", flush=True)
        out.append(rec)
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        be.stop() if be.task else None
        pump_for(3)


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    os._exit(0)          # rclpy's spin thread otherwise dies noisily at interpreter exit
