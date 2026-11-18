"""Proactive alerts (design 4.2): spoken-style sentences built from the event's own fields, no LLM, so an alert is fast
and cannot state anything the event does not carry. Rate limiting: one alert per dedup key per cooldown (stamps are the
event's own sim/wall stamps); `info` alerts are only spoken when the robot is idle; `critical` always passes the idle rule."""
from __future__ import annotations

from typing import Any

COOLDOWN_S = {"critical": 30.0, "warning": 30.0, "info": 300.0}
ALERT_TYPES = ("path_blocked", "stuck", "battery_low", "battery_critical", "leg_failure", "object_new", "object_missing", "fall")


def _name(d: dict) -> str | None:
    if d.get("class"):
        return f"{d.get('color') or ''} {d['class'].replace('_', ' ')}".strip()
    return None


def _text(ev: dict) -> str | None:
    t, d = ev["type"], ev.get("data", {})
    if t == "path_blocked":
        by = f" by the {d['blocked_by']}" if d.get("blocked_by") else ""
        return f"My path is blocked{by}."
    if t == "stuck":
        s = f" for {d['seconds']:g} seconds" if d.get("seconds") is not None else ""
        return f"I am stuck. I have not been making progress{s}."
    if t in ("battery_low", "battery_critical"):
        pct = f", {d['battery_pct']:.0f} percent" if d.get("battery_pct") is not None else ""
        sim = " (simulated)" if d.get("simulated", True) and d.get("battery_pct") is not None else ""
        return f"Battery is {'critically ' if t == 'battery_critical' else ''}low{pct}{sim}."
    if t == "leg_failure":
        return f"The {d['leg'].replace('_', ' ')} leg has failed." if d.get("leg") else "A leg has failed."
    if t == "fall":
        return "I have fallen over."
    if t in ("object_new", "object_missing"):
        n = _name(d)
        if n:
            return f"I see a new {n}." if t == "object_new" else f"The {n} is gone."
        return ev.get("text")          # the event's own text (already built from its fields by the producer)
    return None


class AlertManager:
    def __init__(self, cooldown_s: dict[str, float] | None = None):
        self.cooldown = {**COOLDOWN_S, **(cooldown_s or {})}
        self.last: dict[str, float] = {}
        self.suppressed = 0
        self.spoken: list[dict] = []        # log: stamp, type, text (evaluation / the `why` trail)

    @staticmethod
    def key(ev: dict) -> str:
        d = ev.get("data", {})
        return ev.get("dedup_key") or ":".join(str(x) for x in (ev["type"], d.get("leg") or d.get("object_id") or d.get("task_id") or ""))

    def handle(self, ev: dict, idle: bool = True) -> str | None:
        """The sentence to speak for this event, or None (not an alert / rate-limited / info while busy)."""
        if ev["type"] not in ALERT_TYPES:
            return None
        text = _text(ev)
        if not text:
            return None
        sev = ev.get("severity", "info")
        if sev == "info" and not idle:
            self.suppressed += 1
            return None
        k, stamp = self.key(ev), float(ev.get("stamp", 0.0))
        if k in self.last and stamp - self.last[k] < self.cooldown.get(sev, 30.0):
            self.suppressed += 1
            return None
        self.last[k] = stamp
        self.spoken.append({"stamp": stamp, "type": ev["type"], "text": text})
        return text
