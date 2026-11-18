"""Which logged events answer a `why` question - shared by MockBackend and RosBackend so tests and the real robot answer from the same facts.

stop       the LATEST cause only, phrased as a cause ("I stopped because the go to task ended: cancelled."); older events are history, not the answer.
last_task  the events of the most recent task only (its start and its end), not a mix of everything that ran before.
Found 2026-11-18: with the mock returning the last 3 / 5 events of any task, the model answered "why did you stop" with an older "I have arrived" line."""
from __future__ import annotations


def select_why(about: str, all_events: list[dict], typed: list[dict]) -> list[dict]:
    """`typed`: the events of the types relevant for `about` (already filtered, oldest first); `all_events`: everything logged."""
    if about == "stop":
        evs = [e for e in typed if e["type"] != "task_started"][-1:]
        if not evs:
            return []
        e = evs[0]
        tool = str(e.get("data", {}).get("tool", "task")).replace("_", " ")
        if e["type"] == "task_finished":
            return [dict(e, text=f"I stopped because the {tool} task finished.")]
        if e["type"] == "task_failed":
            reason = e.get("data", {}).get("stopped_reason", "failed")
            if reason == "blocked":
                pb = [x for x in all_events if x["type"] == "path_blocked" and 0 <= e["stamp"] - x["stamp"] <= 5.0]
                if pb:
                    return pb[-1:]          # the blockage is the cause, the failed task is its consequence
            return [dict(e, text=f"I stopped because the {tool} task ended: {reason}.")]
        return evs
    if about == "last_task":
        started = [e for e in all_events if e["type"] == "task_started"]
        tid = started[-1].get("data", {}).get("task_id") if started else None
        if tid is not None:
            return [e for e in typed if e.get("data", {}).get("task_id") == tid][-5:]
    return typed[-5:]

