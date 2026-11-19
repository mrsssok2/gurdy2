"""Builders for the public demo: the mock world with the keyword planner (no LLM, no GPU, no robot). `make_agent` is also the reset factory."""
from __future__ import annotations


def make_agent():
    from brain.agent.graph import Agent
    from brain.agent.llm import KeywordLLM
    from brain.mock import MockBackend, gazebo_flat_lab
    from brain.tools import ToolLayer
    b = MockBackend(gazebo_flat_lab())            # the real Gazebo flat_lab layout, generated from its files
    return Agent(ToolLayer(b), KeywordLLM(b.known_places, smart=True))


def build_host():
    from brain.voice.host import AgentHost
    return AgentHost(make_agent(), sim_speed=3.0)          # mock world: virtual clock runs 3x wall time
