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
    if a.ollama:
        from ..agent.ollama_llm import OllamaLLM
        llm = OllamaLLM(model=a.ollama)
    else:
        llm = KeywordLLM(be.known_places)
    agent = Agent(ToolLayer(be), llm)
    print("Gurdy (ROS). Empty line = show progress; 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text == "quit":
            break
        if text:
            print("Gurdy:", agent.say(text).reply)
        for _ in range(20):                 # let the task engine run while we wait (0.25 s steps, 5 s)
            time.sleep(0.25)
            for m in agent.pump():
                print("Gurdy:", m)
        print(f"  t={port.now():.0f}s pose={port.pose()} mission={agent.executor.state}")


if __name__ == "__main__":
    main()
