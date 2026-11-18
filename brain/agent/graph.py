"""The agent as an explicit LangGraph state machine (design 1.4).

route -> plan (LLM) -> safety_check -> execute -> observe -> respond (LLM) -> grounding_check -> end

* `route` handles, without the LLM: unsafe requests, the e-stop fast path, answers to a pending confirmation.
* `plan` may be called up to MAX_LLM_CALLS times per turn (only when the model sets "continue": it wants to see
  query results first). Motion is acknowledged by a template, so a plain command costs exactly one LLM call.
* `respond` only runs when there are query results to phrase; its text passes the grounding check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..tools import TOOLS, ToolLayer
from . import templates
from .executor import MissionExecutor
from .grounding import GroundedReply, collect_facts, grounded_reply, violations
from .llm import LLM
from .safety import ConfirmationGate, SafetyChecker, Verdict, is_unsafe_request

MAX_LLM_CALLS = 4
SYSTEM_PROMPT = ("You are Gurdy, a hexapod robot. Turn the user's request into tool calls, or ask / refuse. "
                 "You know no facts about the world: use tools. Never state a name, number or direction "
                 "that a tool did not return.")
_STRONG_STOP = re.compile(r"\b(stop|halt|freeze|emergency)\b", re.I)
_WEAK_STOP = re.compile(r"^\W*(stap|stob|stup|top|stopp?|holt|hault|hallt)\W*$", re.I)     # STT confusions, only as a whole utterance
# a question about stopping ("why did you stop") or a request to switch the e-stop off is not a stop command
_NOT_A_COMMAND = re.compile(r"^\W*(why|what|when|where|how|which|who)\b|\b(did|have|had|was|were)\s+(you|it)\b", re.I)


def is_estop_phrase(text: str) -> bool:
    """E-stop fast path (design 4.3 level 2): runs before the LLM, on every transcript."""
    t = text.strip()
    if _WEAK_STOP.match(t):
        return True
    return (bool(_STRONG_STOP.search(t)) and len(t.split()) <= 6 and not _NOT_A_COMMAND.search(t)
            and not is_unsafe_request(t) and not re.search(r"\b(don'?t|do not|never|until)\b", t, re.I))


class TurnState(TypedDict, total=False):
    utterance: str
    stt_confidence: float | None
    log_start: int
    llm_calls: int
    calls: list[dict]
    cont: bool
    approved: bool
    reply_parts: list[str]
    reply: str
    next: str
    has_query: bool
    grounding: dict
    verdicts: list[dict]
    events: list[dict]


@dataclass
class TurnResult:
    reply: str
    calls: list[dict]
    verdicts: list[dict]
    llm_calls: int
    grounding: dict
    events: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, layer: ToolLayer, llm: LLM, safety: SafetyChecker | None = None,
                 executor: MissionExecutor | None = None):
        self.layer, self.llm = layer, llm
        self.safety = safety or SafetyChecker(layer)
        self.executor = executor or MissionExecutor(layer, self.safety)
        self.gate = ConfirmationGate(layer, self.safety.cfg["confirmation_timeout_s"])
        self.replan_used = False        # at most one re-plan per user request (a re-plan's own failure is only reported)
        self.replans: list[dict] = []   # log of re-plan attempts (evaluation / debugging)
        self.graph = self._build()

    # ------------------------------------------------------------------ public API
    def say(self, utterance: str, stt_confidence: float | None = None) -> TurnResult:
        st: TurnState = {"utterance": utterance, "stt_confidence": stt_confidence, "log_start": len(self.layer.log),
                         "llm_calls": 0, "calls": [], "reply_parts": [], "verdicts": [], "events": [], "grounding": {},
                         "approved": False, "has_query": False, "cont": False}
        out = self.graph.invoke(st)
        return TurnResult(out["reply"], out["calls"], out["verdicts"], out["llm_calls"], out["grounding"], out["events"])

    def pump(self) -> list[str]:
        """Advance the executor with backend events; returns what the robot says about them."""
        evs = self.executor.pump()
        if self.executor.state == "awaiting_confirmation" and self.executor.waiting and not self.gate.pending:
            w = self.executor.waiting
            self.gate.hold("mission_step", {"calls": self.executor.remaining_calls(), "resume": True}, w.reason)
        return self._announce(evs)

    def _announce(self, evs: list[dict]) -> list[str]:
        out: list[str] = []
        for e in evs:
            if e["type"] == "mission_failed":
                out += self._on_mission_failed(e)
            elif (m := templates.describe_event(e)):
                out.append(m)
        return out

    # ---- re-plan after a failed mission step (one LLM call, safety-checked like any plan)
    def _on_mission_failed(self, ev: dict) -> list[str]:
        failure_text = templates.describe_event(ev) or "The mission failed."
        replan = getattr(self.llm, "replan", None)
        if replan is None or self.replan_used:
            return [failure_text]
        self.replan_used = True
        ctx = self.executor.failure_context()
        entries = self.layer.log[self.executor.log_start:]
        facts = collect_facts(entries, "", [self.layer.state_block(), failure_text, ctx.get("error") or ""])
        record = {"failure": ctx, "calls": None, "reply": None, "llm_calls": 0, "outcome": None}
        self.replans.append(record)
        out: dict = {}
        for attempt in (1, 2):                      # the second attempt only if the first reply was ungrounded
            fb = None if attempt == 1 else record.get("feedback")
            out = replan(SYSTEM_PROMPT, self.layer.state_block(), self.layer.schemas(), ctx, entries, fb) or {}
            record["llm_calls"] += 1
            if out.get("calls") or not out.get("reply"):
                break
            bad = violations(out["reply"], facts, self.layer.backend.known_places())
            if not bad:
                break
            record["feedback"] = "Your answer contained unsupported claims: " + ", ".join(
                f"{x['kind']} '{x['text']}'" for x in bad) + ". Use only values from the tool results."
            out = {"reply": ""}
        if out.get("calls"):
            return [failure_text] + self._start_replan(out["calls"], record)
        if out.get("reply"):
            record["reply"], record["outcome"] = out["reply"], "reply"
            return [out["reply"]]
        record["outcome"] = "none"
        return [failure_text]

    def _start_replan(self, calls: list[dict], record: dict) -> list[str]:
        record["calls"] = calls
        for c in calls:
            v = self.safety.check(c["tool"], c.get("args", {}))
            if v.outcome == "refuse" or v.outcome == "clarify":
                record["outcome"] = f"blocked:{v.rule}"
                return [v.reason]
            if v.outcome == "confirm":
                self.gate.hold(c["tool"], {"calls": calls}, v.reason)
                record["outcome"] = f"confirm:{v.rule}"
                return [v.reason]
        record["outcome"] = "started"
        evs = self.executor.submit(calls)
        acks = [templates.ack(st.tool, st.args, st.result) for st in self.executor.steps
                if st.status in ("running", "done") and st.result is not None]
        msgs = ["I will try this instead: " + " ".join(acks)] if acks else []
        return msgs + self._announce([e for e in evs if e["type"] == "mission_failed"])

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(TurnState)
        for name in ("route", "plan", "safety_check", "execute", "respond", "finish"):
            g.add_node(name, getattr(self, "n_" + name))
        g.set_entry_point("route")
        g.add_conditional_edges("route", lambda s: s["next"], {"plan": "plan", "execute": "execute", "finish": "finish"})
        g.add_conditional_edges("plan", lambda s: s["next"], {"safety_check": "safety_check", "respond": "respond", "finish": "finish"})
        g.add_conditional_edges("safety_check", lambda s: s["next"], {"execute": "execute", "finish": "finish"})
        g.add_conditional_edges("execute", lambda s: s["next"], {"plan": "plan", "respond": "respond", "finish": "finish"})
        g.add_edge("respond", "finish")
        g.add_edge("finish", END)
        return g.compile()

    def _entries(self, s: TurnState) -> list[dict]:
        return self.layer.log[s["log_start"]:]

    # ---- route
    def n_route(self, s: TurnState) -> dict:
        text = s["utterance"]
        if is_estop_phrase(text):
            ev = self.executor.cancel()
            self.gate.clear()
            res = self.layer.call("stop")
            return {"reply_parts": ["Stopped."], "next": "finish", "events": ev, "has_query": False,
                    "calls": [{"tool": "stop", "args": {}}], "verdicts": [{"outcome": "allow", "rule": "estop_fast_path"}]} \
                if res.ok else {"reply_parts": ["I could not stop."], "next": "finish"}
        if is_unsafe_request(text):
            return {"reply_parts": ["I will not do that."], "next": "finish",
                    "verdicts": [{"outcome": "refuse", "rule": "unsafe_request"}]}
        if self.gate.pending is not None:
            status, p = self.gate.answer(text)
            if status == "expired":
                return {"reply_parts": ["That confirmation has expired, so I did nothing."], "next": "finish"}
            if status == "cancelled":
                return {"reply_parts": ["Okay, cancelled."], "next": "finish"}
            if status == "confirmed":
                if p.tool == "mission_step":           # executor was waiting on its next step
                    ev = self.executor.approve_and_continue()
                    return {"reply_parts": ["Okay."], "next": "finish", "events": ev}
                return {"calls": p.args["calls"], "approved": True, "next": "execute",
                        "verdicts": [{"outcome": "confirmed", "rule": p.reason}]}
        return {"next": "plan"}

    # ---- plan
    def n_plan(self, s: TurnState) -> dict:
        n = s["llm_calls"]
        out = self.llm.plan(SYSTEM_PROMPT, self.layer.state_block(), self.layer.schemas(), s["utterance"],
                            self._entries(s))
        n += 1
        if out.get("done") and s["has_query"]:      # the model says the results so far answer the request
            return {"llm_calls": n, "next": "respond"}
        if "calls" in out and out["calls"]:
            return {"calls": s["calls"] + list(out["calls"]), "cont": bool(out.get("continue")), "llm_calls": n,
                    "next": "safety_check"}
        text = out.get("reply", "")
        # a free-text answer from the planner (refusal / small talk) is not backed by tools: grounding applies
        gr = self._ground(lambda fb: text, s)
        return {"reply_parts": [gr.text], "llm_calls": n, "next": "finish", "grounding": _g(gr)}

    # ---- safety
    def n_safety_check(self, s: TurnState) -> dict:
        verdicts = []
        pending_calls = [c for c in s["calls"] if not c.get("_done")]
        for c in pending_calls:
            v = self.safety.check(c["tool"], c.get("args", {}), s["stt_confidence"])
            verdicts.append({"tool": c["tool"], "outcome": v.outcome, "rule": v.rule, "reason": v.reason})
            if v.outcome == "refuse":
                return {"reply_parts": [v.reason], "verdicts": s["verdicts"] + verdicts, "next": "finish"}
            if v.outcome == "clarify":
                q = v.reason + (f" ({', '.join(v.options)})" if v.options else "")
                self.layer.call("ask_user", {"question": v.reason, **({"options": v.options[:6]} if v.options else {})})
                return {"reply_parts": [q], "verdicts": s["verdicts"] + verdicts, "next": "finish"}
            if v.outcome == "confirm":
                self.gate.hold(c["tool"], {"calls": pending_calls}, v.reason)
                return {"reply_parts": [v.reason], "verdicts": s["verdicts"] + verdicts, "next": "finish"}
        return {"verdicts": s["verdicts"] + verdicts, "next": "execute"}

    # ---- execute (+ observe)
    def n_execute(self, s: TurnState) -> dict:
        motion = []
        parts: list[str] = []
        has_query = False
        asked = False
        for c in s["calls"]:
            kind = TOOLS[c["tool"]].kind if c["tool"] in TOOLS else "query"
            if c["tool"] == "stop":
                ev = self.executor.cancel()
                res = self.layer.call("stop")
                parts.append("Stopped.")
                continue
            if c["tool"] == "ask_user":
                res = self.layer.call("ask_user", c.get("args", {}))
                parts.append(res.data.get("question", "") if res.ok else res.error["message"])
                asked = True
                continue
            if kind == "motion" and c["tool"] != "mission_control":
                motion.append(c)
                continue
            res = self.layer.call(c["tool"], c.get("args", {}))
            if c["tool"] == "mission_control" and res.ok:
                if c["args"]["action"] == "cancel":
                    self.executor.cancel()
                elif c["args"]["action"] == "pause":
                    self.executor.state = "paused" if self.executor.active else self.executor.state
                else:
                    self.executor.state = "running" if self.executor.state == "paused" else self.executor.state
            if kind == "query":
                has_query = True                    # phrased by the LLM (grounded)
            else:                                   # config / mission_control: template from the result
                parts.append(templates.summarize(c["tool"], c.get("args", {}), res.to_dict()))
        events: list[dict] = []
        if motion:
            if self.executor.active:
                parts.append(f"I am busy with {self.executor.steps[0].tool}; say stop first.")
            else:
                self.replan_used = False
                events = self.executor.submit(motion, approved=s["approved"])
                for step in self.executor.steps:
                    if step.status in ("running", "done") and step.result is not None and step.tool not in ("stop",):
                        # a find that is answered from memory finishes inside submit(): say the RESULT (where, how far), not "Looking for"
                        done = next((e for e in events if e["type"] in ("task_finished", "task_failed") and step.tool == "find"
                                     and e["data"].get("task_id") == getattr(step, "task_id", None) and e["data"].get("tool") == "find"), None)
                        parts.append(templates.describe_event(done) if done else templates.ack(step.tool, step.args, step.result))
                    elif step.status == "failed":
                        parts.append(step.result["error"]["message"] if step.result and step.result.get("error") else "I could not start that.")
                if self.executor.state == "awaiting_confirmation" and self.executor.waiting:
                    self.gate.hold("mission_step", {"calls": self.executor.remaining_calls(), "resume": True},
                                   self.executor.waiting.reason)
                    parts.append(self.executor.waiting.reason)
        upd = {"reply_parts": s["reply_parts"] + parts, "events": s["events"] + events, "has_query": has_query,
               "calls": [{**c, "_done": True} for c in s["calls"]]}
        if s["cont"] and s["llm_calls"] < MAX_LLM_CALLS and has_query and not asked:
            return {**upd, "cont": False, "next": "plan"}
        return {**upd, "next": "respond" if has_query and not asked else "finish"}

    # ---- respond + grounding
    def n_respond(self, s: TurnState) -> dict:
        entries = [e for e in self._entries(s) if e["tool"] not in ("stop",)
                   and not (TOOLS[e["tool"]].kind == "motion" and e["tool"] != "mission_control")]

        def gen(fb):
            return self.llm.respond(SYSTEM_PROMPT, s["utterance"], entries, fb)
        gr = self._ground(gen, s, entries)
        return {"reply_parts": [p for p in s["reply_parts"] if p] + [gr.text], "grounding": _g(gr),
                "llm_calls": s["llm_calls"] + gr.attempts}

    def _ground(self, gen, s: TurnState, entries: list[dict] | None = None) -> GroundedReply:
        entries = self._entries(s) if entries is None else entries
        return grounded_reply(gen, entries, s["utterance"], self.layer.backend.known_places(),
                              extra_facts=[self.layer.state_block()])

    def n_finish(self, s: TurnState) -> dict:
        reply = " ".join(p for p in s.get("reply_parts", []) if p) or "Okay."
        return {"reply": reply, "calls": [{k: v for k, v in c.items() if k != "_done"} for c in s.get("calls", [])]}


def _g(gr: GroundedReply) -> dict:
    return {"attempts": gr.attempts, "fallback": gr.fallback, "rejected": gr.rejected, "violations": gr.violations}
