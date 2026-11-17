"""Terminal demo: type commands, the robot moves in the mock world (virtual clock). Planner = keyword baseline
until a model server exists (5.3). Run: python3 -m brain.agent.cli   (needs langgraph on PYTHONPATH)"""
from __future__ import annotations

from ..mock import MockBackend
from ..tools import ToolLayer
from .graph import Agent
from .llm import KeywordLLM


def main() -> None:
    b = MockBackend()
    agent = Agent(ToolLayer(b), KeywordLLM(b.known_places))
    print("Gurdy (mock world, keyword planner). Type a command; empty line = advance 5 s; 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text == "quit":
            break
        if text:
            print("Gurdy:", agent.say(text).reply)
        else:
            b.advance(5)
        for m in agent.pump():
            print("Gurdy:", m)
        if not text:
            print(f"  t={b.t:.0f}s pose=({b.x:.2f},{b.y:.2f}) mission={agent.executor.state}")


if __name__ == "__main__":
    main()
