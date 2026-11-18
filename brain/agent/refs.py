"""Reference resolution before planning: "the other box" / "another barrel" (logged open issue, phase5_live_test.md).

"Walk towards the other box" went to the red box that had just been discussed. Rule: "the other X" is an X that is NOT the one just talked about
(`layer.focus["object"]`). With exactly one such object the utterance is rewritten with an unambiguous description ("the blue box") before the planner
sees it; with several the robot asks which one; with none it says what it knows. No LLM, no guessing: it only uses objects the robot has actually seen
(`recall`). Works for every object class and every word for it (crate, drum, cone, ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data_gen.lexicon import OBJECT_WORDS

COLORS = ("red", "blue", "green", "yellow", "orange", "white", "black", "brown", "grey", "gray")
_WORD_TO_CLASS: dict[str, str] = {}
for _cls, _phrases in OBJECT_WORDS.items():                      # single-word names first: "box" is a box, not the last word of "tool box"
    for _p in _phrases:
        if " " not in _p:
            _WORD_TO_CLASS.setdefault(_p, _cls)
for _cls, _phrases in OBJECT_WORDS.items():
    for _p in _phrases:
        _WORD_TO_CLASS.setdefault(_p.split()[-1], _cls)           # "extinguisher", "cone", ...
_WORD_TO_CLASS.update({"boxes": "box", "barrels": "barrel", "cones": "traffic_cone", "chairs": "chair", "toolboxes": "toolbox"})
_OTHER = re.compile(r"\b(?:the other|another|other)\s+(?:(?P<color>" + "|".join(COLORS) + r")\s+)?(?P<word>[a-z]+)\b", re.I)


@dataclass
class Resolution:
    text: str | None = None                     # rewritten utterance (planner input)
    reply: str | None = None                    # nothing to plan: say this
    question: str | None = None                 # ask which one
    options: dict[str, str] = field(default_factory=dict)   # answer word (colour) -> rewritten utterance
    rule: str = ""


def _label(r: dict, cls_word: str) -> str:
    return f"{r['color']} {cls_word}".strip() if r.get("color") else cls_word


def resolve_other(text: str, focus: dict | None, records_fn) -> Resolution | None:
    """None = the utterance has no "other X" reference. `records_fn(cls) -> [ {id, class, color, place?} ... ]` of seen objects of that class."""
    m = _OTHER.search(text)
    if not m:
        return None
    word = m.group("word").lower()
    cls = _WORD_TO_CLASS.get(word) or _WORD_TO_CLASS.get(word.rstrip("s"))
    if cls is None:
        return None                              # "the other side", "another way": not an object reference
    want_color = (m.group("color") or "").lower().replace("gray", "grey") or None
    recs = [r for r in records_fn(cls) if r.get("state", "present") == "present"]
    fo = focus.get("object") if focus else None
    last = fo if fo and fo.get("class") == cls else None
    cand = [r for r in recs if not (last and r.get("id") == last.get("id")) and (want_color is None or (r.get("color") or "") == want_color)]
    noun = word if word != "boxes" else "box"
    if not cand:
        if not recs:
            return Resolution(reply=f"I haven't seen any {noun} yet.", rule="other_none_seen")
        only = _label(recs[0], noun)
        return Resolution(reply=f"I only know one {noun}: the {only}." if len(recs) == 1 else f"I don't know another {noun} like that.", rule="other_none_left")
    span = m.span()

    def rewrite(r: dict) -> str:
        return text[:span[0]] + f"the {_label(r, noun)}" + text[span[1]:]

    def unique(r: dict) -> bool:
        key = (r.get("color") or "")
        return key != "" and sum(1 for x in recs if (x.get("color") or "") == key) == 1     # the description names exactly one known object
    if len(cand) == 1 and unique(cand[0]):
        return Resolution(text=rewrite(cand[0]), rule="other_resolved")
    named = [r for r in cand if unique(r)]
    if len(named) == len(cand) and len(cand) > 1:
        names = [_label(r, noun) for r in cand]
        q = "Which one: " + ", ".join(f"the {n}" for n in names[:-1]) + f" or the {names[-1]}?"
        return Resolution(question=q, options={r["color"]: rewrite(r) for r in cand}, rule="other_which")
    places = sorted({r.get("place") for r in cand if r.get("place")})
    where = f" ({', '.join(places)})" if places else ""
    return Resolution(question=f"I know more than one other {noun}{where}. Which one do you mean? Say its colour, for example 'the red {noun}'.", rule="other_which_unclear")
