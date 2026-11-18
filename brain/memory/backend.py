"""MemoryBackend: wraps any RobotBackend (mock or ROS) and makes `recall`, `save_place` and mission history
persistent. Everything else is delegated untouched, so the agent, safety layer and executor do not change.

- `recall("object")`: live tracks are first synced into SQLite (class, colour, position, place, when last seen),
  then the answer comes from the DB, so objects seen in an earlier run are still answered, with their age.
- `recall("mission")`: history built from task_started / task_finished / task_failed events (`since_s` = "today").
- `save_place` persists; saved places are restored into the inner backend at start (if it keeps a `places` dict).
Records from memory carry `from_memory` and `age_s`; older than `stale_after_s` they are flagged `stale`."""
from __future__ import annotations

from typing import Any

from brain.tools.types import Result, ok

from .store import MemoryStore

MOTION = {"walk", "turn", "go_to", "find", "explore", "patrol", "inspect", "return_home"}


class MemoryBackend:
    def __init__(self, inner: Any, store: MemoryStore, stale_after_s: float = 1800.0):
        self._inner, self.store, self.stale_after_s = inner, store, stale_after_s
        self._pending: dict[str, dict] = {}          # tool -> args of the motion call that was just issued
        self._rows: dict[Any, int] = {}              # inner task id -> mission row
        store.close_open_missions()
        if hasattr(inner, "places"):
            for p in store.places():
                inner.places.setdefault(p["name"], (p["x"], p["y"]))

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name in MOTION and callable(attr):
            def call(*a, **kw):
                self._pending[name] = dict(kw)
                return attr(*a, **kw)
            return call
        return attr

    # ------------------------------------------------------------------ events -> mission history
    def poll_events(self) -> list[dict]:
        evs = self._inner.poll_events()
        for e in evs:
            d, t = e.get("data", {}), e["type"]
            if t == "task_started":
                tool = d.get("tool", "task")
                self._rows[d.get("task_id")] = self.store.mission_started(tool, _target(self._pending.pop(tool, {})))
            elif t in ("task_finished", "task_failed") and d.get("task_id") in self._rows:
                self.store.mission_ended(self._rows.pop(d["task_id"]), d.get("stopped_reason", "done" if t == "task_finished" else "failed"),
                                         d.get("distance_m"), {k: v for k, v in d.items() if k in ("tool", "found", "stopped_reason")})
            elif t in ("object_new", "object_missing"):
                self.sync_objects()
        return evs

    def sync_objects(self) -> None:
        res = self._inner.recall("object")
        if not res.ok:
            return
        now = self.store.wall()
        for r in res.data["records"]:
            self.store.upsert_object(r["class"], r.get("color"), r["x"], r["y"], r.get("place"),
                                     now - r.get("age_s", 0.0), r.get("state", "present"))

    # ------------------------------------------------------------------ tools
    def recall(self, what: str, query: dict | None = None) -> Result:
        query = query or {}
        if what == "object":
            self.sync_objects()
            recs = [{**o, "stale": o["age_s"] > self.stale_after_s, "from_memory": True, "map_id": self.store.map_id}
                    for o in self.store.objects(query.get("class"), query.get("color"), query.get("since_s"))]
            return ok(records=recs)
        if what == "mission":
            return ok(records=self.store.missions(query.get("since_s")))
        if what == "place":
            return ok(records=[{**p, "map_id": self.store.map_id} for p in self.store.places()] or
                      self._inner.recall("place", query).data["records"])
        return self._inner.recall(what, query)

    def save_place(self, name: str) -> Result:
        res = self._inner.save_place(name)
        if res.ok:
            p = res.data["place"]
            self.store.save_place(p["name"], p["x"], p["y"])
        return res


def _target(args: dict) -> str | None:
    if args.get("place"):
        return args["place"]
    o = args.get("object")
    if o:
        return f"{o.get('color') or ''} {o['class']}".strip()
    return args.get("direction")
