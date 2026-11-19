"""Deterministic phrasing normaliser for the keyword planner used in demos (public demo, text / voice mock).

It rewrites common ways of saying a command into the canonical wording `brain.eval.keyword_parser` understands
("enter room b" -> "go to room b"). Pure regex, no learning, no state, so the same input always gives the same plan.
The evaluation baseline (`KeywordLLM(smart=False)`, the Gurdy-1.0-style parser) does NOT use it, so its numbers stay comparable."""
from __future__ import annotations

import re

_POLITE = re.compile(r"\b(?:hey|hi|ok|okay)?\s*gurdy\b|\b(?:can|could|would|will) you\b|\bplease\b|\bfor me\b|\bi (?:want|need|would like|'d like) you to\b|\bi'd like you to\b|\bkindly\b|\bjust\b|\bgo ahead and\b|\blet's\b|\blets\b")
_MOVE_TO = re.compile(r"\b(?:enter|visit|go into|go in to|go inside|walk into|walk in to|head into|move into|move to|take me to|bring me to|drive to|proceed to|get to|reach|go over to|head over to|run to|travel to|come over to|come into|step into|approach)\b")
_GO_BACK_TO = re.compile(r"\b(?:go|head|get|come|walk|move|return) back (?:to|into)\b")
_HOME_WORDS = re.compile(r"\b(?:the )?(?:start|base|origin|beginning|charging (?:spot|station)|where (?:you|we) started)\b")
_BARE_BACK = re.compile(r"^\s*(?:go|head|get|come|return|walk|move) back(?: here| there| again| please)?\s*$|^\s*(?:return|come back|go back|head back)(?: to (?:me|base|start))?\s*$|^\s*come (?:here|home)\s*$|^\s*(?:take yourself|go) home\s*$")
_FIND = re.compile(r"\b(?:go find|go and find|go look for|hunt for|hunt down|seek|try to find|help me find|find me|get me|where can i find|can you spot|spot)\b")
_SEARCH_OBJ = re.compile(r"\b(?:search|scan|look) (?:the |this )?(?:room|area|place)?\s*for\b")
_AREA_SCAN = re.compile(r"\b(?:look around|have a look around|scan (?:the |this )?(?:room|area|place)|map (?:the |this )?(?:room|area|place)|check out (?:the |this )?(?:room|area)|survey)\b")
_AROUND = re.compile(r"\bwhat(?:'s| is) (?:around|near|close)(?: to)?(?: me| you)?\b|\bwhat(?:'s| is) nearby\b|\bwhat(?:'s| is) here\b")
_TURN = re.compile(r"\b(?:rotate|spin|pivot|swivel)\b")


def normalize(text: str, places: list[str]) -> str:
    t = " " + re.sub(r"[’]", "'", text.lower().strip()) + " "
    t = _POLITE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" .!?,")
    if not t:
        return text
    t = _AREA_SCAN.sub("explore", t)
    t = _AROUND.sub("what is nearby", t)
    t = _TURN.sub("turn", t)
    t = _SEARCH_OBJ.sub("find", t)
    t = re.sub(r"\bsearch (?!for\b)", "find ", t) if _has_object_word(t) else t
    t = _FIND.sub("find", t)
    # "go back to the start / base" -> home; "go back to room a" -> go to room a (not a backward walk)
    t = _GO_BACK_TO.sub("go to", t)
    t = re.sub(r"\bgo to " + _HOME_WORDS.pattern, "go home", t)
    t = _MOVE_TO.sub("go to", t)
    # whole-clause rewrites after "and"/"then" splitting happen in the parser, so rewrite per clause here too
    parts = re.split(r"(\band then\b|\bthen\b|\bafter that\b|\band\b|,|;)", t)
    out = []
    for p in parts:
        q = p.strip()
        if _BARE_BACK.match(q):
            q = "return home"
        elif _is_bare_place(q, places):
            q = "go to " + q
        out.append(q)
    return " ".join(x for x in (s.strip() for s in out) if x)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _is_bare_place(clause: str, places: list[str]) -> bool:
    c = re.sub(r"^(?:the|to|into) ", "", _norm(clause))
    return bool(c) and any(c == _norm(p) for p in places)


def _has_object_word(t: str) -> bool:
    try:
        from gurdy_core.vision.classes import ALIASES
    except ImportError:
        return False
    words = set(re.findall(r"[a-z]+", t))
    return any(w in al for al in ALIASES.values() for w in words)
