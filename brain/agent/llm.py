"""LLM interface of the agent + a scripted fake (tests) + a keyword 'LLM' (terminal demo, baseline).

plan(...)    -> {"calls": [{"tool", "args"}...], "continue": bool}   or   {"reply": "text"}
respond(...) -> text, phrased from tool results only (checked by the grounding check afterwards)
replan(...)  -> like plan(), asked ONCE after a mission step failed (`failure` = executor.failure_context());
                {} means "no recovery, just report". Its calls pass the safety checker like any other plan.
The real implementation (Ollama, JSON-schema-constrained output) comes in 5.3; it only has to match this protocol.
"""
from __future__ import annotations

from typing import Any, Protocol


class LLMUnavailable(RuntimeError):
    """The language model cannot be reached or does not have the model (Ollama down, wrong address, model missing). `user_text` is what the
    robot says; the technical detail (`str(e)`) goes to the agent log only."""

    def __init__(self, detail: str, user_text: str = "My language model is not running right now."):
        super().__init__(detail)
        self.user_text = user_text


class LLM(Protocol):
    def plan(self, system: str, state: str, tools: list[dict], utterance: str, tool_results: list[dict]) -> dict: ...
    def respond(self, system: str, utterance: str, tool_results: list[dict], feedback: str | None = None) -> str: ...
    def replan(self, system: str, state: str, tools: list[dict], failure: dict, tool_results: list[dict],
               feedback: str | None = None) -> dict: ...


class ScriptedLLM:
    """Returns pre-written outputs in order and records every prompt it was given. A callable output is
    invoked with the keyword arguments of the call (to react to tool results)."""

    def __init__(self, plans: list | None = None, responses: list | None = None, replans: list | None = None):
        self.plans, self.responses, self.replans = list(plans or []), list(responses or []), list(replans or [])
        self.plan_calls: list[dict] = []
        self.respond_calls: list[dict] = []
        self.replan_calls: list[dict] = []

    def plan(self, system, state, tools, utterance, tool_results):
        self.plan_calls.append({"system": system, "state": state, "n_tools": len(tools), "utterance": utterance,
                                "tool_results": tool_results})
        out = self.plans.pop(0)
        return out(**self.plan_calls[-1]) if callable(out) else out

    def respond(self, system, utterance, tool_results, feedback=None):
        self.respond_calls.append({"utterance": utterance, "tool_results": tool_results, "feedback": feedback})
        out = self.responses.pop(0)
        return out(**self.respond_calls[-1]) if callable(out) else out


    def replan(self, system, state, tools, failure, tool_results, feedback=None):
        self.replan_calls.append({"state": state, "failure": failure, "tool_results": tool_results, "feedback": feedback})
        out = self.replans.pop(0)
        return out(**self.replan_calls[-1]) if callable(out) else out


class KeywordLLM:
    """The Gurdy-1.0-style baseline as an agent planner: regex parse, template reply. No learning."""

    def __init__(self, places_fn, smart: bool = False):
        """`smart=True` first normalises common phrasings (`phrasing.normalize`: "enter room b", "take me to ..."); the default stays the
        plain Gurdy-1.0-style baseline that the evaluation compares against."""
        self.places_fn = places_fn
        self.smart = smart

    def plan(self, system, state, tools, utterance, tool_results):
        from ..eval.keyword_parser import parse
        places = self.places_fn()
        if self.smart:
            from .phrasing import normalize
            r = parse(normalize(utterance, places), places)
            if "calls" in r:
                return r
        return parse(utterance, places)

    def respond(self, system, utterance, tool_results, feedback=None):
        from .templates import summarize_results
        return summarize_results(tool_results)

    def replan(self, system, state, tools, failure, tool_results, feedback=None):
        return {}          # the baseline cannot recover from a failed step; the agent just reports it
