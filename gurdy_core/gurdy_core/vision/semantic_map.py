"""Semantic map: de-duplicated object tracks from repeated detections (map frame, static objects).

Association: nearest existing track of the SAME class within a class-specific gate. Position = inverse-variance
weighted mean (sigma grows with range, so a far sighting never overrides a near one). A track becomes
*confirmed* after `min_hits` sightings (suppresses one-frame false positives). Unconfirmed tracks that are not
seen again within `ttl` seconds are dropped. Confirmed tracks of one class that drift within the gate of each
other are merged. Time is whatever the caller passes (sim time in the ROS node).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .classes import SIZE_PRIOR


def obs_sigma(range_m: float) -> float:
    """1-sigma position noise of a depth-camera sighting (m): ~3 cm close, ~8 cm at 2 m, ~35 cm at 4 m."""
    return 0.03 + 0.02 * range_m ** 2


@dataclass
class Track:
    id: int
    cls: str
    pos: np.ndarray
    weight: float
    hits: int = 1
    best_score: float = 0.0
    score_sum: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    colors: dict = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return self.score_sum / max(self.hits, 1)

    @property
    def color(self) -> str:
        return max(self.colors, key=self.colors.get) if self.colors else ""

    @property
    def sigma(self) -> float:
        return float(1.0 / np.sqrt(self.weight))

    def to_dict(self) -> dict:
        return {"id": self.id, "class": self.cls, "x": float(self.pos[0]), "y": float(self.pos[1]),
                "z": float(self.pos[2]), "color": self.color, "hits": self.hits, "score": round(self.mean_score, 3),
                "sigma_m": round(self.sigma, 3), "first_seen": self.first_seen, "last_seen": self.last_seen}


class SemanticMap:
    def __init__(self, min_hits: int = 3, ttl: float = 10.0, gate_scale: float = 1.0, min_gate: float = 0.35,
                 gates: dict | None = None):
        self.min_hits, self.ttl, self.min_gate = min_hits, ttl, min_gate
        self.gates = gates or {c: max(gate_scale * max(s[0], s[1]), min_gate) for c, s in SIZE_PRIOR.items()}
        self.tracks: list[Track] = []
        self._next = 1

    def gate(self, cls: str) -> float:
        return self.gates.get(cls, self.min_gate)

    def update(self, cls: str, pos, score: float, stamp: float, range_m: float = 1.0, color: str = "") -> Track:
        pos = np.asarray(pos, float)
        w = 1.0 / obs_sigma(range_m) ** 2
        best, bd = None, self.gate(cls)
        for t in self.tracks:
            if t.cls == cls:
                d = float(np.linalg.norm(t.pos[:2] - pos[:2]))
                if d <= bd:
                    best, bd = t, d
        if best is None:
            best = Track(self._next, cls, pos.copy(), w, 1, score, score, stamp, stamp)
            self._next += 1
            self.tracks.append(best)
        else:
            best.pos = (best.pos * best.weight + pos * w) / (best.weight + w)
            best.weight += w
            best.hits += 1
            best.best_score = max(best.best_score, score)
            best.score_sum += score
            best.last_seen = stamp
        if color:
            best.colors[color] = best.colors.get(color, 0) + 1
        self._expire(stamp)
        self._merge()
        return best

    def _expire(self, now: float) -> None:
        self.tracks = [t for t in self.tracks if t.hits >= self.min_hits or now - t.last_seen <= self.ttl]

    def _merge(self) -> None:
        i = 0
        while i < len(self.tracks):
            a, merged = self.tracks[i], False
            for j in range(i + 1, len(self.tracks)):
                b = self.tracks[j]
                if a.cls == b.cls and np.linalg.norm(a.pos[:2] - b.pos[:2]) <= self.gate(a.cls) * 0.6:
                    a.pos = (a.pos * a.weight + b.pos * b.weight) / (a.weight + b.weight)
                    a.weight += b.weight
                    a.hits += b.hits
                    for k, v in b.colors.items():
                        a.colors[k] = a.colors.get(k, 0) + v
                    a.score_sum += b.score_sum
                    a.best_score = max(a.best_score, b.best_score)
                    a.first_seen, a.last_seen = min(a.first_seen, b.first_seen), max(a.last_seen, b.last_seen)
                    del self.tracks[j]
                    merged = True
                    break
            if not merged:
                i += 1

    def confirmed(self) -> list:
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def find(self, cls: str, near=None, color: str = "") -> list:
        """Confirmed tracks of a class, best first (nearest to `near` xy if given, else highest mean score)."""
        c = [t for t in self.confirmed() if t.cls == cls and (not color or t.color == color)]
        if near is not None:
            return sorted(c, key=lambda t: float(np.linalg.norm(t.pos[:2] - np.asarray(near)[:2])))
        return sorted(c, key=lambda t: -t.mean_score)

    def to_json(self) -> str:
        return json.dumps({"objects": [t.to_dict() for t in self.confirmed()]}, indent=1)
