"""SQLite incident store. Same thread-safety pattern as brain/memory/store.py: one connection (`check_same_thread=False`) behind an
RLock around every public method; results are plain dataclasses / dicts, never cursors. Foreign keys are enforced."""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from .model import (CAUSE_SOURCES, EVIDENCE_KINDS, INCIDENT_STATUSES, ROLES, Cause, Event, Evidence, Incident, Run,
                    severity_rank)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, robot_yaml_hash TEXT, controller TEXT, fault_label TEXT,
  config TEXT, created REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), stamp REAL NOT NULL,
  type TEXT NOT NULL, severity TEXT NOT NULL, source TEXT, text TEXT, dedup_key TEXT, fingerprint TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_run_stamp ON events (run_id, stamp);
CREATE TABLE IF NOT EXISTS incidents (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), start REAL NOT NULL,
  end REAL, severity TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', title TEXT);
CREATE TABLE IF NOT EXISTS incident_events (incident_id INTEGER NOT NULL REFERENCES incidents(id),
  event_id INTEGER NOT NULL REFERENCES events(id), role TEXT NOT NULL, PRIMARY KEY (incident_id, event_id));
CREATE TABLE IF NOT EXISTS causes (id INTEGER PRIMARY KEY, incident_id INTEGER NOT NULL REFERENCES incidents(id),
  label TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY, cause_id INTEGER NOT NULL REFERENCES causes(id),
  kind TEXT NOT NULL, ref TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS similar (incident_id INTEGER NOT NULL REFERENCES incidents(id),
  other_id INTEGER NOT NULL REFERENCES incidents(id), score REAL NOT NULL, PRIMARY KEY (incident_id, other_id));
"""


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)
    return wrapper


def _check(value: str, allowed: tuple[str, ...], what: str) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}; expected one of {allowed}")


class IncidentStore:
    def __init__(self, path: str | Path, wall: Callable[[], float] = time.time):
        self.path, self.wall = str(path), wall
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    @_locked
    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # ------------------------------------------------------------------ runs
    @_locked
    def add_run(self, run: Run) -> None:
        """Insert a run; re-adding an existing run_id raises sqlite3.IntegrityError (a replay must not silently overwrite a run)."""
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
                        (run.run_id, run.robot_yaml_hash, run.controller, run.fault_label,
                         json.dumps(run.config, sort_keys=True), run.created or self.wall()))
        self.db.commit()

    @_locked
    def get_run(self, run_id: str) -> Run | None:
        r = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return None if r is None else Run(r["run_id"], r["robot_yaml_hash"], r["controller"], r["fault_label"],
                                          json.loads(r["config"] or "{}"), r["created"])

    @_locked
    def runs(self) -> list[Run]:
        return [self.get_run(r["run_id"]) for r in self.db.execute("SELECT run_id FROM runs ORDER BY created, run_id")]

    # ------------------------------------------------------------------ events
    @_locked
    def add_event(self, ev: Event) -> int:
        cur = self.db.execute("INSERT INTO events (run_id, stamp, type, severity, source, text, dedup_key, fingerprint, data)"
                              " VALUES (?,?,?,?,?,?,?,?,?)",
                              (ev.run_id, ev.stamp, ev.type, ev.severity, ev.source, ev.text, ev.dedup_key, ev.fingerprint,
                               json.dumps(ev.data, sort_keys=True, default=str)))
        self.db.commit()
        ev.id = cur.lastrowid
        return ev.id

    @staticmethod
    def _event(r: sqlite3.Row) -> Event:
        return Event(r["run_id"], r["stamp"], r["type"], r["severity"], r["source"], r["text"], r["dedup_key"],
                     json.loads(r["data"] or "{}"), r["fingerprint"], r["id"])

    @_locked
    def get_event(self, event_id: int) -> Event | None:
        r = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return None if r is None else self._event(r)

    @_locked
    def events(self, run_id: str, since: float | None = None, until: float | None = None, type: str | None = None) -> list[Event]:
        q, args = "SELECT * FROM events WHERE run_id=?", [run_id]
        if since is not None:
            q, args = q + " AND stamp>=?", args + [since]
        if until is not None:
            q, args = q + " AND stamp<=?", args + [until]
        if type is not None:
            q, args = q + " AND type=?", args + [type]
        return [self._event(r) for r in self.db.execute(q + " ORDER BY stamp, id", args)]

    @_locked
    def fingerprint_counts(self, run_id: str) -> dict[str, int]:
        return {r["fingerprint"]: r["n"] for r in self.db.execute(
            "SELECT fingerprint, COUNT(*) n FROM events WHERE run_id=? GROUP BY fingerprint", (run_id,))}

    # ------------------------------------------------------------------ incidents
    @_locked
    def open_incident(self, run_id: str, start: float, severity: str, title: str = "") -> int:
        severity_rank(severity)
        cur = self.db.execute("INSERT INTO incidents (run_id, start, severity, status, title) VALUES (?,?,?,'open',?)",
                              (run_id, start, severity, title))
        self.db.commit()
        return cur.lastrowid

    @_locked
    def add_event_to_incident(self, incident_id: int, event_id: int, role: str = "member") -> None:
        """Link an event. Severity becomes the max over members (context events do not raise it); the incident's end moves to the
        latest member stamp. A closed incident cannot take new members (reopen is not a thing: open a new incident)."""
        _check(role, ROLES, "role")
        inc = self._row("incidents", incident_id)
        if inc["status"] != "open":
            raise ValueError(f"incident {incident_id} is {inc['status']}; cannot add events")
        ev = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if ev is None:
            raise KeyError(f"event {event_id} not found")
        if ev["run_id"] != inc["run_id"]:
            raise ValueError("event and incident belong to different runs")
        self.db.execute("INSERT INTO incident_events VALUES (?,?,?)", (incident_id, event_id, role))
        sev = inc["severity"]
        if role != "context" and severity_rank(ev["severity"]) > severity_rank(sev):
            sev = ev["severity"]
        end = max(ev["stamp"], inc["end"] if inc["end"] is not None else ev["stamp"])
        self.db.execute("UPDATE incidents SET severity=?, end=? WHERE id=?", (sev, end, incident_id))
        self.db.commit()

    @_locked
    def close_incident(self, incident_id: int, end: float | None = None) -> None:
        inc = self._row("incidents", incident_id)
        last = inc["end"] if inc["end"] is not None else inc["start"]
        self.db.execute("UPDATE incidents SET status='closed', end=? WHERE id=?",
                        (max(last, end) if end is not None else last, incident_id))
        self.db.commit()

    @_locked
    def set_title(self, incident_id: int, title: str) -> None:
        self._row("incidents", incident_id)
        self.db.execute("UPDATE incidents SET title=? WHERE id=?", (title, incident_id))
        self.db.commit()

    def _row(self, table: str, row_id: int) -> sqlite3.Row:
        r = self.db.execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()    # table is internal, never user input
        if r is None:
            raise KeyError(f"{table[:-1]} {row_id} not found")
        return r

    def _incident(self, r: sqlite3.Row) -> Incident:
        ids = [x["event_id"] for x in self.db.execute(
            "SELECT ie.event_id FROM incident_events ie JOIN events e ON e.id=ie.event_id WHERE ie.incident_id=? ORDER BY e.stamp, e.id",
            (r["id"],))]
        return Incident(r["run_id"], r["start"], r["severity"], r["title"] or "", r["end"], r["status"], r["id"], ids)

    @_locked
    def get_incident(self, incident_id: int) -> Incident | None:
        r = self.db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        return None if r is None else self._incident(r)

    @_locked
    def incidents(self, run_id: str | None = None, status: str | None = None, min_severity: str | None = None,
                  since: float | None = None) -> list[Incident]:
        """Filtered list ordered by start. `since` keeps incidents that ended (or are still open) at/after that run time."""
        if status is not None:
            _check(status, INCIDENT_STATUSES, "status")
        floor = severity_rank(min_severity) if min_severity else 0
        out = []
        for r in self.db.execute("SELECT * FROM incidents ORDER BY start, id"):
            if run_id is not None and r["run_id"] != run_id:
                continue
            if status is not None and r["status"] != status:
                continue
            if severity_rank(r["severity"]) < floor:
                continue
            if since is not None and r["end"] is not None and r["end"] < since:
                continue
            out.append(self._incident(r))
        return out

    # ------------------------------------------------------------------ causes, evidence, similar incidents
    @_locked
    def add_cause(self, incident_id: int, label: str, confidence: float, source: str) -> int:
        """Causes only ADD (design 4: each step adds evidence, never overwrites); several causes per incident are ranked by confidence."""
        _check(source, CAUSE_SOURCES, "cause source")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence} outside [0, 1]")
        self._row("incidents", incident_id)
        cur = self.db.execute("INSERT INTO causes (incident_id, label, confidence, source) VALUES (?,?,?,?)",
                              (incident_id, label, confidence, source))
        self.db.commit()
        return cur.lastrowid

    @_locked
    def add_evidence(self, cause_id: int, kind: str, ref: str, summary: str) -> int:
        _check(kind, EVIDENCE_KINDS, "evidence kind")
        self._row("causes", cause_id)
        cur = self.db.execute("INSERT INTO evidence (cause_id, kind, ref, summary) VALUES (?,?,?,?)", (cause_id, kind, ref, summary))
        self.db.commit()
        return cur.lastrowid

    @_locked
    def causes(self, incident_id: int) -> list[Cause]:
        """Highest confidence first (ties: oldest first), each with its evidence."""
        out = []
        for c in self.db.execute("SELECT * FROM causes WHERE incident_id=? ORDER BY confidence DESC, id", (incident_id,)):
            ev = [Evidence(e["kind"], e["ref"], e["summary"], e["id"])
                  for e in self.db.execute("SELECT * FROM evidence WHERE cause_id=? ORDER BY id", (c["id"],))]
            out.append(Cause(c["incident_id"], c["label"], c["confidence"], c["source"], ev, c["id"]))
        return out

    @_locked
    def set_similar(self, incident_id: int, other_id: int, score: float) -> None:
        if incident_id == other_id:
            raise ValueError("an incident cannot be similar to itself")
        self._row("incidents", incident_id)
        self._row("incidents", other_id)
        self.db.execute("INSERT INTO similar VALUES (?,?,?) ON CONFLICT(incident_id, other_id) DO UPDATE SET score=excluded.score",
                        (incident_id, other_id, score))
        self.db.commit()

    @_locked
    def similar(self, incident_id: int, k: int = 5) -> list[tuple[int, float]]:
        return [(r["other_id"], r["score"]) for r in self.db.execute(
            "SELECT other_id, score FROM similar WHERE incident_id=? ORDER BY score DESC, other_id LIMIT ?", (incident_id, k))]

    # ------------------------------------------------------------------ summary used by the API / brain tool later
    @_locked
    def detail(self, incident_id: int) -> dict | None:
        inc = self.get_incident(incident_id)
        if inc is None:
            return None
        counts: dict[str, int] = {}
        for eid in inc.event_ids:
            t = self.get_event(eid).type
            counts[t] = counts.get(t, 0) + 1
        causes = self.causes(incident_id)
        return {"id": inc.id, "run_id": inc.run_id, "title": inc.title, "start": inc.start, "end": inc.end, "status": inc.status,
                "severity": inc.severity, "n_events": len(inc.event_ids), "events_by_type": counts,
                "causes": [{"label": c.label, "confidence": c.confidence, "source": c.source,
                            "evidence": [{"kind": e.kind, "ref": e.ref, "summary": e.summary} for e in c.evidence]} for c in causes],
                "similar": [{"incident_id": i, "score": s} for i, s in self.similar(incident_id)]}
