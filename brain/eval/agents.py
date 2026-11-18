"""Agent factories for the evaluation harness. Each takes (backend, case) and returns an Agent over that backend.

keyword  the Gurdy-1.0-style regex parser as planner, inside the same agent graph (so the same safety layer,
         grounding check and executor apply). It is the baseline the LLM has to beat.
oracle   plans exactly the first allowed trace of each turn. Not a competitor: an upper bound that proves the test set
         is solvable and the scoring consistent (a case the oracle fails is a broken case).
"""
from __future__ import annotations

from ..agent import Agent, KeywordLLM
from ..agent.templates import summarize_results
from ..mock import MockBackend
from ..tools import ToolLayer


class OracleLLM:
    def __init__(self, case: dict):
        self.script: list[tuple[str, list[dict]]] = []
        for t in case["turns"]:
            traces = t.get("expect", {}).get("calls")
            if traces:
                self.script.append((t["say"], traces[0]))

    def plan(self, system, state, tools, utterance, tool_results):
        for i, (say, trace) in enumerate(self.script):
            if say == utterance:
                self.script.pop(i)
                calls = [{"tool": c["tool"], "args": dict(c.get("args", {}))} for c in trace]
                for c in calls:
                    if c["tool"] == "ask_user" and "question" not in c["args"]:
                        c["args"]["question"] = "Could you tell me more precisely what you mean?"
                return {"calls": calls, "continue": False} if calls else {"reply": "I cannot do that."}
        return {"reply": "Sorry, I did not understand that."}

    def respond(self, system, utterance, tool_results, feedback=None):
        return summarize_results(tool_results)


def keyword(b: MockBackend, case: dict) -> Agent:
    return Agent(ToolLayer(b), KeywordLLM(b.known_places))


def oracle(b: MockBackend, case: dict) -> Agent:
    return Agent(ToolLayer(b), OracleLLM(case))


_LLMS: dict = {}


def _ollama(model: str, style: str = "compact", constrain: bool = True):
    def make(b: MockBackend, case: dict) -> Agent:
        from ..agent.ollama_llm import OllamaLLM
        llm = _LLMS.setdefault((model, style, constrain), OllamaLLM(model, prompt_style=style, constrain=constrain))     # one client (stats accumulate over the run)
        return Agent(ToolLayer(b), llm)
    return make


AGENTS = {"keyword": keyword, "oracle": oracle,
          "qwen3-1.7b": _ollama("qwen3:1.7b"), "gurdy-sft-400-grammar": _ollama("gurdy-sft-400"), "gurdy-sft-400": _ollama("gurdy-sft-400", constrain=False), "qwen3-0.6b": _ollama("qwen3:0.6b"),
          # old JSON-schema prompt, for reference runs (phase5_eval_base.md)
          "qwen3-1.7b-json": _ollama("qwen3:1.7b", "json"), "qwen3-0.6b-json": _ollama("qwen3:0.6b", "json")}
