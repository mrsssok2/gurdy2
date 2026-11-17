"""LLM interface of the agent + a scripted fake (tests) + a keyword 'LLM' (terminal demo, baseline).

plan(...)    -> {"calls": [{"tool", "args"}...], "continue": bool}   or   {"reply": "text"}
respond(...) -> text, phrased from tool results only (checked by the grounding check afterwards)
The real implementation (Ollama, JSON-schema-constrained output) comes in 5.3; it only has to match this protocol.
"""
from __future__ import annotations

from typing import Any, Protocol


class LLM(Protocol):
    def plan(self, system: str, state: str, tools: list[dict], utterance: str, tool_results: list[dict]) -> dict: ...
    def respond(self, system: str, utterance: str, tool_results: list[dict], feedback: str | None = None) -> str: ...


class ScriptedLLM:
    """Returns pre-written outputs in order and records every prompt it was given. A callable output is
    invoked with the keyword arguments of the call (to react to tool results)."""

    def __init__(self, plans: list | None = None, responses: list | None = None):
        self.plans, self.responses = list(plans or []), list(responses or [])
        self.plan_calls: list[dict] = []
        self.respond_calls: list[dict] = []

    def plan(self, system, state, tools, utterance, tool_results):
        self.plan_calls.append({"system": system, "state": state, "n_tools": len(tools), "utterance": utterance,
                                "tool_results": tool_results})
        out = self.plans.pop(0)
        return out(**self.plan_calls[-1]) if callable(out) else out

    def respond(self, system, utterance, tool_results, feedback=None):
        self.respond_calls.append({"utterance": utterance, "tool_results": tool_results, "feedback": feedback})
        out = self.responses.pop(0)
        return out(**self.respond_calls[-1]) if callable(out) else out


class KeywordLLM:
    """The Gurdy-1.0-style baseline as an agent planner: regex parse, template reply. No learning."""

    def __init__(self, places_fn):
        self.places_fn = places_fn

    def plan(self, system, state, tools, utterance, tool_results):
        from ..eval.keyword_parser import parse
        return parse(utterance, self.places_fn())

    def respond(self, system, utterance, tool_results, feedback=None):
        from .templates import summarize_results
        return summarize_results(tool_results)
