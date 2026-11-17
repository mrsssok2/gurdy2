"""Grounding check ("never invented", design 1.5): every name, number and direction in a reply must be supported
by a tool result of the same turn, the user's own words, or the state block (itself built from tools).

Checked: object class words / aliases, colours, place names, numbers with units (m, cm, %, deg, s, min), ids (#4),
bare numbers, direction words. Numbers match within rounding (distance +-0.1 m, percent / degrees / seconds +-1).
`violations()` is also the hallucination metric of the evaluation (5.3).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from gurdy_core.vision.classes import ALIASES
except ImportError:                       # repo checkout without gurdy_core installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gurdy_core"))
    from gurdy_core.vision.classes import ALIASES

from .templates import summarize_results

COLOURS = {"red", "green", "blue", "yellow", "orange", "white", "black", "grey", "gray", "brown", "purple", "pink"}
DIRECTION_GROUPS = {"ahead": "ahead", "front": "ahead", "forward": "ahead", "forwards": "ahead",
                    "behind": "behind", "backward": "behind", "backwards": "behind", "left": "left", "right": "right"}
ALIAS_WORDS = {w for al in ALIASES.values() for w in al}
_NUMWORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_UNIT = r"(metres?|meters?|m|centimetres?|centimeters?|cm|%|percent|°|degrees?|deg|seconds?|secs?|s|minutes?|mins?)"
_NUM_UNIT = re.compile(r"(?<![\w.#])(\d+(?:\.\d+)?)\s*" + _UNIT + r"(?![A-Za-z])", re.I)
_ID = re.compile(r"#(\d+)")
_BARE = re.compile(r"(?<![\w.#])(\d+(?:\.\d+)?)(?![\w.%°])")


def _unit_kind(u: str) -> str:
    u = u.lower()
    if u in ("cm", "centimetre", "centimetres", "centimeter", "centimeters"):
        return "cm"
    if u.startswith("m") and u not in ("min", "mins", "minute", "minutes"):
        return "m"
    if u in ("%", "percent"):
        return "pct"
    if u in ("°", "deg", "degree", "degrees"):
        return "deg"
    if u.startswith("min"):
        return "min"
    return "s"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("_", " ").replace("-", " "))


def _sing(t: str) -> str:
    if t.endswith("xes") or t.endswith("ches"):
        return t[:-2]
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


@dataclass
class Facts:
    tokens: set[str] = field(default_factory=set)
    text: str = ""
    numbers: list[float] = field(default_factory=list)
    ids: set[int] = field(default_factory=set)

    def add_text(self, s: str) -> None:
        self.tokens.update(_sing(t) for t in _tokens(s))
        self.text += " " + " ".join(_tokens(s))
        for m in re.finditer(r"\d+(?:\.\d+)?", s):
            self.numbers.append(float(m.group()))

    def add(self, v: Any) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            self.numbers.append(float(v))
        elif isinstance(v, str):
            self.add_text(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                self.add_text(str(k))
                self.add(x)
                if k == "id" and isinstance(x, int):
                    self.ids.add(x)
                if k == "object_id" and isinstance(x, int):
                    self.ids.add(x)
        elif isinstance(v, (list, tuple)):
            self.numbers.append(float(len(v)))
            for x in v:
                self.add(x)


def collect_facts(entries: Iterable[dict], user_text: str = "", extra: Iterable[str] = ()) -> Facts:
    """entries = ToolLayer.log slice of this turn (args and results both count: the user's own request is a fact)."""
    f = Facts()
    for e in entries:
        f.add(e["args"])
        f.add(e["result"]["data"])
        if e["result"]["error"]:
            f.add(e["result"]["error"])
    f.add_text(user_text)
    for s in extra:
        f.add(s)
    return f


def _num_ok(v: float, kind: str, facts: Facts) -> bool:
    for n in facts.numbers:
        if kind == "m" and (abs(n - v) <= 0.1 or abs(n / 100 - v) <= 0.1 or abs(n * 1000 - v) <= 0.1):
            return True
        if kind == "cm" and (abs(n * 100 - v) <= 10 or abs(n - v) <= 1):
            return True
        if kind in ("pct", "deg", "s") and abs(n - v) <= 1:
            return True
        if kind == "s" and abs(n * 60 - v) <= 1:
            return True
        if kind == "min" and abs(n / 60 - v) <= 0.5:
            return True
    return False


def violations(reply: str, facts: Facts, places: Iterable[str] = ()) -> list[dict]:
    """Unsupported claims in `reply`: [{"kind": number|id|class|colour|place|direction, "text": ...}]."""
    out: list[dict] = []
    covered: list[tuple[int, int]] = []
    for m in _NUM_UNIT.finditer(reply):
        covered.append(m.span())
        if not _num_ok(float(m.group(1)), _unit_kind(m.group(2)), facts):
            out.append({"kind": "number", "text": m.group(0)})
    for m in _ID.finditer(reply):
        covered.append(m.span())
        if int(m.group(1)) not in facts.ids:
            out.append({"kind": "id", "text": m.group(0)})
    for m in _BARE.finditer(reply):
        if any(a <= m.start() < b for a, b in covered):
            continue
        if not any(abs(n - float(m.group(1))) < 0.05 for n in facts.numbers):
            out.append({"kind": "number", "text": m.group(0)})
    for w in re.findall(r"[a-z]+", reply.lower()):
        if w in _NUMWORDS and w not in ("zero", "one") and not any(abs(n - _NUMWORDS[w]) < 0.05 for n in facts.numbers):
            out.append({"kind": "number", "text": w})
    toks = [_sing(t) for t in _tokens(reply)]
    seen: set[str] = set()
    for t in toks:
        if t in seen:
            continue
        if t in {_sing(a) for a in ALIAS_WORDS} and t not in facts.tokens:
            out.append({"kind": "class", "text": t})
        elif t in COLOURS and t not in facts.tokens:
            out.append({"kind": "colour", "text": t})
        elif t in DIRECTION_GROUPS:
            grp = DIRECTION_GROUPS[t]
            if not any(DIRECTION_GROUPS.get(x) == grp for x in facts.tokens):
                out.append({"kind": "direction", "text": t})
        else:
            continue
        seen.add(t)
    rtext = " ".join(_tokens(reply))
    for p in places:
        pt = " ".join(_tokens(p))
        if pt and re.search(rf"\b{re.escape(pt)}\b", rtext) and not re.search(rf"\b{re.escape(pt)}\b", facts.text):
            out.append({"kind": "place", "text": p})
    return out


@dataclass
class GroundedReply:
    text: str
    violations: list[dict]            # of the text that was finally returned (empty unless fallback also failed)
    attempts: int                     # LLM generations used (1 or 2)
    fallback: bool
    rejected: list[str] = field(default_factory=list)


def grounded_reply(generate: Callable[[str | None], str], entries: list[dict], user_text: str, places: Iterable[str],
                   extra_facts: Iterable[str] = ()) -> GroundedReply:
    """Generate -> check -> regenerate once with the violations as feedback -> template fallback."""
    facts = collect_facts(entries, user_text, extra_facts)
    places = list(places)
    rejected: list[str] = []
    feedback = None
    for attempt in (1, 2):
        text = generate(feedback)
        v = violations(text, facts, places)
        if not v:
            return GroundedReply(text, [], attempt, False, rejected)
        rejected.append(text)
        feedback = "Your answer contained unsupported claims: " + ", ".join(f"{x['kind']} '{x['text']}'" for x in v) \
                   + ". Use only values from the tool results."
    fb = summarize_results(entries)
    return GroundedReply(fb, violations(fb, facts, places), 2, True, rejected)
