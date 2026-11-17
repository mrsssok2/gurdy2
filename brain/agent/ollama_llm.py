"""Ollama client implementing the `LLM` protocol (design 3.3). Standard library only.

Prompt-cache rule (measured trap in 5.1: every distinct prompt prefix costs ~0.3 GiB of host RAM in llama.cpp):
the SYSTEM message is built ONCE and is byte-identical for plan, respond and replan. It holds the role, the output
rules and the tool block. Everything that changes (state, tool results, the user's words, the task) goes into the
USER message after it. So the cached prefix (~2.5k tokens) is evaluated once per model load; `stats` records
`prompt_eval_count` (tokens actually evaluated) per call to show that it stays flat.

plan / replan: JSON-schema-constrained output (`format`): either {"calls": [...], "continue": bool} or {"reply": str}.
respond: free text. Thinking is off (`think: false`), temperature 0.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..tools import TOOLS

SYSTEM_RULES = """You are Gurdy, a six-legged robot. You turn the user's request into tool calls, or you ask, or you refuse.
You know NO facts about the world (places, objects, colours, distances, battery). Use query tools for facts and never state a name, number or direction that a tool result did not contain.
Output rules for a planning task: answer with one JSON object, either
  {"calls": [{"tool": <name>, "args": {...}}, ...], "continue": false}   -- tool calls in order. "continue": false for everything that is complete as planned, INCLUDING plain questions (the answer is phrased after the query runs). "continue": true only if a later call depends on a query result
  {"reply": "<short text>"}                                              -- only for refusing, small talk, or when no tool fits
  {"done": true}                                                         -- only when TOOL RESULTS SO FAR already answer the request; then nothing more is called
Use ask_user (one question) when the request is ambiguous or a required value is missing. Do not guess. Dangerous requests (leaving the mapped area, driving into people, disabling safety, stopping the emergency stop) are refused with a reply. "stop" always means the stop tool.
Output rules for a reply task: answer in one or two short sentences, in plain English, using only the tool results.
Examples:
  user: go to the kitchen -> {"calls": [{"tool": "go_to", "args": {"place": "kitchen"}}], "continue": false}
  user: where is the red box -> {"calls": [{"tool": "where_is", "args": {"object": {"class": "box", "color": "red"}}}], "continue": false}
  user: turn left ninety degrees -> {"calls": [{"tool": "turn", "args": {"direction": "left", "angle_deg": 90}}], "continue": false}
"""
_HEADER_JSON = "Tools (JSON schema of each):\n"
_HEADER_COMPACT = ("Tools, one per line: name(params) -- description. In params `?` = optional, `a|b` = one of, `str>=1` = non-empty string, "
                   "`num[lo,hi]` / `int[lo,hi]` = bounds, `list<T>[min,max]`, `{...}` = object, `obj` = free-form object; "
                   "`exactly_one_of(a,b)` = exactly one of those keys. CLASS is the object class enum defined on the first line.\n")

PROMPT_STYLES = ("compact", "json")   # compact = default everywhere (decision 2026-11-17); json = the old prompt, kept for reference runs


def build_system(style: str = "compact") -> str:
    """The one stable system message. `compact`: lossless compact tool block + CLASS alias (brain/train/compact_tools.py, -59 % tool tokens).
    `json`: the original JSON schema block (base-model baseline of 5.3, phase5_eval_base.md)."""
    if style == "json":
        tools = [s.as_llm_tool() for s in TOOLS.values()]
        return SYSTEM_RULES + _HEADER_JSON + "\n".join(json.dumps(t, separators=(",", ":")) for t in tools)
    if style == "compact":
        from ..train.compact_tools import build_compact_block
        return SYSTEM_RULES + _HEADER_COMPACT + build_compact_block(alias=True)
    raise ValueError(f"prompt style {style!r}, expected one of {PROMPT_STYLES}")


def plan_schema(with_results: bool = False) -> dict:
    """With tool results already in the prompt the model may also answer {"done": true}: the results are enough, phrase them."""
    call = {"type": "object", "properties": {"tool": {"type": "string", "enum": sorted(TOOLS)}, "args": {"type": "object"}},
            "required": ["tool", "args"]}
    done = [{"type": "object", "properties": {"done": {"type": "boolean"}}, "required": ["done"]}] if with_results else []
    return {"anyOf": done + [
        {"type": "object", "properties": {"calls": {"type": "array", "items": call, "minItems": 1}, "continue": {"type": "boolean"}},
         "required": ["calls", "continue"]},
        {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}]}


def _results(entries: list[dict]) -> str:
    rows = [{"tool": e["tool"], "args": e.get("args", {}), "result": e["result"]} for e in entries]
    s = json.dumps(rows, separators=(",", ":"))
    return s if len(s) < 6000 else s[:6000] + "...(truncated)"


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None, num_ctx: int = 4096, num_predict: int = 400,
                 keep_alive: str = "30m", timeout: float = 120.0, prompt_style: str = "compact"):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11435"
        self.options = {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}
        self.keep_alive, self.timeout = keep_alive, timeout
        self.prompt_style = prompt_style
        self.system = build_system(prompt_style)
        self.schema, self.schema_results = plan_schema(), plan_schema(True)
        self.stats: list[dict] = []
        self.parse_failures = 0

    # ------------------------------------------------------------------ transport
    def _chat(self, kind: str, user: str, fmt: dict | None) -> str:
        body: dict[str, Any] = {"model": self.model, "stream": False, "think": False, "keep_alive": self.keep_alive,
                                "options": self.options,
                                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}]}
        if fmt:
            body["format"] = fmt
        t0 = time.perf_counter()
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {self.host}: HTTP {e.code} {e.read()[:200]!r} (model {self.model})") from e
        self.stats.append({"kind": kind, "wall_s": round(time.perf_counter() - t0, 3), "prompt_eval": d.get("prompt_eval_count"),
                           "out_tokens": d.get("eval_count"), "load_s": round(d.get("load_duration", 0) / 1e9, 3)})
        return d["message"]["content"]

    def _parse(self, text: str) -> dict:
        try:
            out = json.loads(text)
            if isinstance(out, dict) and ("calls" in out or "reply" in out or "done" in out):
                return out
        except json.JSONDecodeError:
            pass
        self.parse_failures += 1
        return {"reply": "Sorry, I did not understand that."}

    # ------------------------------------------------------------------ LLM protocol
    def plan(self, system, state, tools, utterance, tool_results):
        u = f"TASK: plan\n{state}\n"
        if tool_results:
            u += f"TOOL RESULTS SO FAR: {_results(tool_results)}\n"
        u += f'USER SAID: "{utterance}"\nJSON:'
        return self._parse(self._chat("plan", u, self.schema_results if tool_results else self.schema))

    def respond(self, system, utterance, tool_results, feedback=None):
        u = f'TASK: reply\nUSER SAID: "{utterance}"\nTOOL RESULTS: {_results(tool_results)}\n'
        if feedback:
            u += f"Your previous answer was rejected: {feedback}\n"
        u += "Answer:"
        text = self._chat("respond", u, None).strip()
        if text.startswith("{"):   # free-text request answered as {"reply": "..."}: use the text
            try:
                d = json.loads(text)
                if isinstance(d, dict) and isinstance(d.get("reply"), str):
                    text = d["reply"].strip()
            except json.JSONDecodeError:
                pass
        return text

    def replan(self, system, state, tools, failure, tool_results, feedback=None):
        u = (f"TASK: plan (a mission step failed; recover or give up)\n{state}\nFAILURE: {json.dumps(failure, separators=(',', ':'))}\n"
             f"TOOL RESULTS SO FAR: {_results(tool_results)}\n")
        if feedback:
            u += f"Your previous answer was rejected: {feedback}\n"
        u += "JSON:"
        out = self._parse(self._chat("replan", u, self.schema_results))
        return out if out.get("calls") else {}
