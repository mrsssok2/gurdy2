"""SQLite store: objects (last seen), missions (history), places. Times are WALL-clock epoch seconds (a restart does
not reset the age of a memory); `wall` is injectable so tests control time. Every row carries `map_id`:
coordinates from one SLAM run mean nothing in another, so a query never crosses maps."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, map_id TEXT NOT NULL, class TEXT NOT NULL, color TEXT,
  x REAL, y REAL, place TEXT, first_seen REAL, last_seen REAL, hits INTEGER DEFAULT 0, state TEXT DEFAULT 'present');
CREATE TABLE IF NOT EXISTS missions (id INTEGER PRIMARY KEY, map_id TEXT NOT NULL, session TEXT, tool TEXT, target TEXT,
  start REAL, end REAL, state TEXT, distance_m REAL, data TEXT);
CREATE TABLE IF NOT EXISTS places (map_id TEXT NOT NULL, name TEXT NOT NULL, x REAL, y REAL, saved REAL,
  PRIMARY KEY (map_id, name));
"""
SAME_OBJECT_M = 0.6          # same class + colour within this distance = the same object (re-sighting)


class MemoryStore:
    def __init__(self, path: str | Path, map_id: str, wall: Callable[[], float] = time.time):
        self.path, self.map_id, self.wall = str(path), map_id, wall
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.session = f"s{int(self.wall() * 1000)}"

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # ------------------------------------------------------------------ objects
    def upsert_object(self, cls: str, color: str | None, x: float, y: float, place: str | None,
                      last_seen: float, state: str = "present") -> int:
        rows = self.db.execute("SELECT * FROM objects WHERE map_id=? AND class=? AND COALESCE(color,'')=?",
                               (self.map_id, cls, color or "")).fetchall()
        near = [r for r in rows if math.hypot(r["x"] - x, r["y"] - y) <= SAME_OBJECT_M]
        if near:
            r = min(near, key=lambda r: math.hypot(r["x"] - x, r["y"] - y))
            newer = last_seen >= r["last_seen"]
            self.db.execute("UPDATE objects SET x=?, y=?, place=?, last_seen=?, hits=hits+1, state=? WHERE id=?",
                            (x if newer else r["x"], y if newer else r["y"], place if newer else r["place"],
                             max(r["last_seen"], last_seen), state if newer else r["state"], r["id"]))
            self.db.commit()
            return r["id"]
        cur = self.db.execute("INSERT INTO objects (map_id, class, color, x, y, place, first_seen, last_seen, hits, state)"
                              " VALUES (?,?,?,?,?,?,?,?,1,?)", (self.map_id, cls, color or "", x, y, place, last_seen, last_seen, state))
        self.db.commit()
        return cur.lastrowid

    def objects(self, cls: str | None = None, color: str | None = None, since_s: float | None = None) -> list[dict]:
        now = self.wall()
        out = []
        for r in self.db.execute("SELECT * FROM objects WHERE map_id=? ORDER BY last_seen DESC", (self.map_id,)):
            if cls and r["class"] != cls:
                continue
            if color and (r["color"] or "").lower() != color.lower():
                continue
            age = max(0.0, now - r["last_seen"])
            if since_s is not None and age > since_s:
                continue
            out.append({"id": r["id"], "class": r["class"], "color": r["color"] or "", "x": r["x"], "y": r["y"],
                        "place": r["place"], "age_s": round(age, 1), "state": r["state"], "hits": r["hits"]})
        return out

    # ------------------------------------------------------------------ missions
    def mission_started(self, tool: str, target: str | None) -> int:
        cur = self.db.execute("INSERT INTO missions (map_id, session, tool, target, start, state) VALUES (?,?,?,?,?,'running')",
                              (self.map_id, self.session, tool, target, self.wall()))
        self.db.commit()
        return cur.lastrowid

    def mission_ended(self, row_id: int, state: str, distance_m: float | None, data: dict[str, Any]) -> None:
        self.db.execute("UPDATE missions SET end=?, state=?, distance_m=?, data=? WHERE id=?",
                        (self.wall(), state, distance_m, json.dumps(data, default=str), row_id))
        self.db.commit()

    def close_open_missions(self) -> int:
        """Missions still 'running' from an earlier session ended with that process: mark them interrupted."""
        cur = self.db.execute("UPDATE missions SET state='interrupted', end=start WHERE map_id=? AND state='running' AND session<>?",
                              (self.map_id, self.session))
        self.db.commit()
        return cur.rowcount

    def missions(self, since_s: float | None = None) -> list[dict]:
        now = self.wall()
        out = []
        for r in self.db.execute("SELECT * FROM missions WHERE map_id=? ORDER BY start", (self.map_id,)):
            age = max(0.0, now - (r["end"] or r["start"]))
            if since_s is not None and age > since_s:
                continue
            out.append({"id": r["id"], "tool": r["tool"], "target": r["target"], "state": r["state"],
                        "age_s": round(age, 1), "distance_m": r["distance_m"]})
        return out

    # ------------------------------------------------------------------ places
    def save_place(self, name: str, x: float, y: float) -> None:
        self.db.execute("INSERT INTO places (map_id, name, x, y, saved) VALUES (?,?,?,?,?) ON CONFLICT(map_id, name) "
                        "DO UPDATE SET x=excluded.x, y=excluded.y, saved=excluded.saved", (self.map_id, name, x, y, self.wall()))
        self.db.commit()

    def places(self) -> list[dict]:
        return [{"name": r["name"], "x": r["x"], "y": r["y"]}
                for r in self.db.execute("SELECT * FROM places WHERE map_id=? ORDER BY saved", (self.map_id,))]
