"""Mission executor: a deterministic state machine that runs a plan (ordered tool calls) step by step.

The LLM never blocks on motion: it emits a plan, the executor starts one step at a time and moves on when the
backend reports `task_finished` for it. Every step is safety-checked again right before it starts (battery, legs
and pose may have changed since the plan was made). On a failed step the rest of the plan is cancelled and a
`mission_failed` event is produced so the agent can report (or re-plan).

Step states: pending -> running -> done | failed | cancelled.
Mission states: idle | running | paused | awaiting_confirmation | done | failed | cancelled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..tools import TOOLS, ToolLayer
from .safety import SafetyChecker, Verdict

OK_REASONS = ("reached", "done")


@dataclass
class Step:
    tool: str
    args: dict
    status: str = "pending"
    task_id: int | None = None
    result: dict | None = None
    reason: str = ""

    def as_call(self) -> dict:
        return {"tool": self.tool, "args": self.args}


class MissionExecutor:
    def __init__(self, layer: ToolLayer, safety: SafetyChecker):
        self.layer, self.safety, self.backend = layer, safety, layer.backend
        self.steps: list[Step] = []
        self.state = "idle"
        self.approved = False                  # the user confirmed this plan: confirm-level verdicts pass
        self.waiting: Verdict | None = None    # set while awaiting_confirmation
        self.history: list[dict] = []          # finished missions (state, steps)

    # ------------------------------------------------------------------ public
    @property
    def active(self) -> bool:
        return self.state in ("running", "paused", "awaiting_confirmation")

    def remaining_calls(self) -> list[dict]:
        return [s.as_call() for s in self.steps if s.status in ("pending", "running")]

    def submit(self, calls: list[dict], approved: bool = False) -> list[dict]:
        """Start a plan. Returns synthesized events (mission_started, ...)."""
        if self.active:
            raise RuntimeError("a mission is already active; cancel it first")
        self.steps = [Step(c["tool"], c.get("args", {})) for c in calls]
        self.approved, self.waiting, self.state = approved, None, "running"
        events = [self._ev("mission_started", "info", f"Mission with {len(self.steps)} step(s) started.")]
        return events + self._advance()

    def cancel(self) -> list[dict]:
        if not self.active:
            return []
        if any(s.status == "running" for s in self.steps):
            self.backend.stop()
        for s in self.steps:
            if s.status in ("pending", "running"):
                s.status = "cancelled"
        return [self._finish("cancelled", "Mission cancelled.")] + self._drain()

    def pause(self) -> None:
        if self.state == "running":
            self.layer.call("mission_control", {"action": "pause"})
            self.state = "paused"

    def resume(self) -> list[dict]:
        if self.state != "paused":
            return []
        self.layer.call("mission_control", {"action": "resume"})
        self.state = "running"
        return []

    def approve_and_continue(self) -> list[dict]:
        """The user said yes to the step the executor was waiting on."""
        if self.state != "awaiting_confirmation":
            return []
        self.approved, self.waiting, self.state = True, None, "running"
        return self._advance()

    def pump(self) -> list[dict]:
        """Consume backend events; advance the plan. Returns every event (backend + mission_*) in order."""
        return self._drain()

    def run(self, advance: Callable[[float], None], max_s: float = 3600.0, dt: float = 1.0) -> list[dict]:
        """Drive the plan with a clock (`advance(dt)`: MockBackend.advance). Stops when idle/waiting/failed."""
        out, t = [], 0.0
        while self.state == "running" and t < max_s:
            advance(dt)
            t += dt
            out += self.pump()
        return out

    # ------------------------------------------------------------------ internals
    def _ev(self, typ: str, severity: str, text: str, **data: Any) -> dict:
        return {"stamp": round(self.backend.now(), 2), "type": typ, "severity": severity, "source": "executor",
                "text": text, "data": data}

    def _drain(self) -> list[dict]:
        out: list[dict] = []
        for ev in self.backend.poll_events():
            out.append(ev)
            if ev["type"] in ("task_finished", "task_failed"):
                self.layer.note_task_finished(ev)
                cur = next((s for s in self.steps if s.status == "running" and s.task_id == ev["data"].get("task_id")), None)
                if cur is None:
                    continue
                cur.result, cur.reason = ev["data"], ev["data"].get("stopped_reason", "done")
                if cur.reason in OK_REASONS:
                    cur.status = "done"
                    out += self._advance()
                else:
                    cur.status = "failed"
                    out += self._fail(f"{cur.tool} stopped: {cur.reason}", cur)
        return out

    def _advance(self) -> list[dict]:
        """Start the next pending step (query-like steps complete instantly, so loop)."""
        out: list[dict] = []
        while self.state == "running":
            nxt = next((s for s in self.steps if s.status == "pending"), None)
            if nxt is None:
                if not any(s.status == "running" for s in self.steps):
                    out.append(self._finish("done", "Mission finished."))
                return out
            if any(s.status == "running" for s in self.steps):
                return out
            v = self.safety.check(nxt.tool, nxt.args)
            if v.outcome == "confirm" and not self.approved:
                self.state, self.waiting = "awaiting_confirmation", v
                out.append(self._ev("confirmation_needed", "info", v.reason, tool=nxt.tool, rule=v.rule))
                return out
            if v.outcome in ("refuse", "clarify"):
                nxt.status, nxt.reason = "failed", v.rule
                out += self._fail(v.reason, nxt)
                return out
            res = self.layer.call(nxt.tool, nxt.args)
            if not res.ok:
                nxt.status, nxt.reason, nxt.result = "failed", res.code or "error", res.to_dict()
                out += self._fail(res.error["message"], nxt)
                return out
            nxt.result = res.data
            if res.data.get("state") == "started" and "task_id" in res.data:
                nxt.status, nxt.task_id = "running", res.data["task_id"]
                return out + self._drain()       # a no-op task may already have finished
            nxt.status = "done"
        return out

    def _fail(self, why: str, step: Step) -> list[dict]:
        for s in self.steps:
            if s.status == "pending":
                s.status = "cancelled"
        return [self._finish("failed", why, step=step.tool, reason=step.reason)]

    def _finish(self, state: str, text: str, **data: Any) -> dict:
        self.state = state
        self.history.append({"state": state, "steps": [(s.tool, s.status) for s in self.steps]})
        typ = "mission_finished" if state == "done" else f"mission_{state}"
        return self._ev(typ, "info" if state in ("done", "cancelled") else "warning", text, **data)
