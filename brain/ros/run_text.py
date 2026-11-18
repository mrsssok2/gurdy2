"""Type commands against the real robot in Gazebo (RosBackend). Run inside the sim container after the sim is up
(see docs/results/phase5_ros_backend.md). Planner: keyword baseline by default, `--ollama MODEL` for the LLM.
  python3 -m brain.ros.run_text --places maps/flat_lab.places.yaml --map-id flat_lab
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", default=None)
    ap.add_argument("--map-id", default="unknown")
    ap.add_argument("--odom-topic", default="/odometry/filtered")
    ap.add_argument("--memory-db", default=None, help="SQLite file: objects / missions / places survive restarts (per --map-id)")
    ap.add_argument("--ollama", default=None, help="model name; needs OLLAMA_URL")
    a = ap.parse_args()
    from ..agent.graph import Agent
    from ..agent.llm import KeywordLLM
    from ..tools import ToolLayer
    from .backend import RosBackend
    from .rclpy_port import RclpyPort
    port = RclpyPort(map_id=a.map_id, odom_topic=a.odom_topic)
    while port.pose() is None:
        print("waiting for TF map->base_link ...")
        time.sleep(1.0)
    be = RosBackend(port, a.places)
    if a.memory_db:
        from ..memory import MemoryBackend, MemoryStore
        be = MemoryBackend(be, MemoryStore(a.memory_db, a.map_id))
        print(f"persistent memory: {a.memory_db} (map {a.map_id})")
    if a.ollama:
        from ..agent.ollama_llm import OllamaLLM
        llm = OllamaLLM(model=a.ollama, constrain=not a.ollama.startswith("gurdy-sft"))   # fine-tuned model: unconstrained (phase5_eval_sft.md)
    else:
        llm = KeywordLLM(be.known_places)
    agent = Agent(ToolLayer(be), llm)
    from .chat_ui import Chat
    chat = Chat(agent, be.known_places, prompt="> ")
    print("Gurdy (ROS). Type a command; 'status' line = '?'; '!estop' / '!release' = emergency stop latch; 'quit' to exit.")
    chat.start()                       # progress, alerts and new objects arrive on their own, without breaking what you type
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text == "quit":
            break
        if text == "?":
            print(f"  t={port.now():.0f}s pose={port.pose()} mission={agent.executor.state}")
        elif text in ("!estop", "!release"):
            with chat.lock:
                be.estop(text == "!estop")
            print("Gurdy: " + ("Emergency stop engaged (latched)." if text == "!estop" else "Emergency stop released."))
        elif text:
            print("Gurdy:", chat.ask(text))
    chat.stop()


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    os._exit(0)          # rclpy's spin thread otherwise dies noisily at interpreter exit
