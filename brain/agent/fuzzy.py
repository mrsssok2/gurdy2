"""Speech-recognition repair: "go to room V" -> "Did you mean room B?" (no LLM, pure strings).

A place or object word that is NOT known but is phonetically close to a known one is never executed and never
silently corrected: the agent asks, and only a clear yes runs the corrected request (Agent.say, `pending_fix`).

* places:  multi-token names ("room B") match an n-gram of the text token by token; a single-letter token matches
           a letter of the same sound group (b/d/p/t/v/e/c/g/z, m/n, f/s, ...), longer tokens by edit similarity.
* objects: the word after find / go to / ... that is not a known class word but is close to one ("barrow" -> barrel,
           "bocks" -> box), and a colour word before a known class ("bread box" -> red box is NOT attempted: too loose).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..data_gen.lexicon import COLORS_BY_CLASS, OBJECT_WORDS

# letters Whisper / people confuse when spoken ("bee dee pee vee ee tee cee gee zee", "em en", "eff ess", "eye", ...)
_LETTER_GROUPS = ["bdpvtecgz", "mn", "fsx", "ai", "hk", "ou", "jy", "lr"]
_GROUP = {ch: i for i, g in enumerate(_LETTER_GROUPS) for ch in g}
_TOK = re.compile(r"[a-z0-9']+")
_PREPS = {"to", "towards", "toward", "into", "at", "near", "from", "the"}
_SAFE_SKIP = re.compile(r"\b(save|remember|call|name|label|mark|bookmark)\b", re.I)     # the user may be NAMING a new place
_OBJ_CONTEXT = re.compile(r"\b(?:find|locate|look for|search for|go to|walk to|head to|bring me|take me to|approach|inspect|visit)\s+"
                          r"(?:the |a |an )?((?:[a-z]+ ){0,1}[a-z]+)\b", re.I)


@dataclass
class Suggestion:
    heard: str
    options: list[str]                  # known names, best first
    corrected: dict[str, str] = field(default_factory=dict)   # option -> full corrected utterance
    kind: str = "place"

    @property
    def question(self) -> str:
        opts = self.options
        if len(opts) == 1:
            return f"Did you mean {opts[0]}? I heard \"{self.heard}\"."
        return f"Did you mean {', '.join(opts[:-1])} or {opts[-1]}? I heard \"{self.heard}\"."


def _tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def _tok_sim(heard: str, known: str) -> float:
    """1.0 equal, 0.9 same-sound single letters, else edit ratio (only for words of 4+ letters), else 0."""
    if heard == known:
        return 1.0
    if len(heard) == 1 and len(known) == 1:
        return 0.9 if heard in _GROUP and _GROUP.get(heard) == _GROUP.get(known) else 0.0
    if len(heard) < 4 or len(known) < 4:
        return 0.0
    r = SequenceMatcher(None, heard, known).ratio()
    return r if r >= 0.75 else 0.0


def _ngram_score(gram: list[str], name: list[str]) -> float:
    sims = [_tok_sim(g, n) for g, n in zip(gram, name)]
    return 0.0 if 0.0 in sims else min(sims)


def suggest_place(text: str, known_places: list[str]) -> Suggestion | None:
    """The first n-gram of `text` that is close to, but not exactly, a known place (and is not itself a known place)."""
    if not known_places or _SAFE_SKIP.search(text):
        return None
    toks = _tokens(text)
    names = {p: _tokens(p) for p in known_places}
    exact = {tuple(n) for n in names.values()}
    best: dict[int, tuple[float, str, str]] = {}                 # start index -> (score, place, heard phrase)
    cands: dict[int, dict[str, float]] = {}
    for p, n in names.items():
        k = len(n)
        for i in range(len(toks) - k + 1):
            gram = toks[i:i + k]
            if tuple(gram) in exact:
                continue
            if k > 1 and gram[0] != n[0] and _tok_sim(gram[0], n[0]) < 1.0:
                continue                                          # multi-word names: the head word ("room") must be right
            if k == 1 and (i == 0 or toks[i - 1] not in _PREPS):
                continue                                          # a lone word is only a place after "to / into / at ..." ("come back" is not "home")
            sc = _ngram_score(gram, n)
            if sc >= 0.75:
                cands.setdefault(i * 10 + k, {})[p] = sc
    if not cands:
        return None
    key = min(cands)                                              # leftmost
    i, k = divmod(key, 10)
    opts = sorted(cands[key], key=lambda p: -cands[key][p])
    top = cands[key][opts[0]]
    opts = [p for p in opts if cands[key][p] >= top - 0.05]
    heard = " ".join(toks[i:i + k])
    sug = Suggestion(heard, opts)
    pat = re.compile(r"\W+".join(re.escape(t) for t in toks[i:i + k]), re.I)     # replace in the original text (keeps case / punctuation)
    for p in opts:
        sug.corrected[p] = pat.sub(p, text, count=1)
    return sug


def _known_object_tokens() -> set[str]:
    return {w for ws in OBJECT_WORDS.values() for ph in ws for w in ph.split()}


_COMMON = {"the", "a", "an", "it", "there", "here", "me", "my", "some", "any", "one", "that", "this", "red", "blue", "green", "yellow",
           "white", "orange", "black", "brown", "grey", "gray", "home", "room", "doorway", "person", "people", "table", "door", "wall",
           "kitchen", "office", "garage", "bathroom", "bedroom"}


def suggest_object(text: str, known_places: list[str] | None = None) -> Suggestion | None:
    """After find / go to / ...: an unknown last word close to a known class word ("barrow" -> barrel)."""
    m = _OBJ_CONTEXT.search(text)
    if not m:
        return None
    toks = _tokens(m.group(1))
    known = _known_object_tokens()
    place_toks = {t for p in (known_places or []) for t in _tokens(p)}
    last = toks[-1]
    if last in known or last in _COMMON or last in place_toks or len(last) < 4:
        return None
    scored = sorted(((SequenceMatcher(None, last, w).ratio(), w) for w in known if len(w) >= 3), reverse=True)
    if not scored or scored[0][0] < 0.66:
        return None
    # accept only a clear winner (the runner-up must be clearly worse)
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.03 and scored[1][1] != scored[0][1]:
        return None
    w = scored[0][1]
    sug = Suggestion(last, [w], kind="object")
    sug.corrected[w] = re.sub(rf"\b{re.escape(last)}\b", w, text, count=1, flags=re.I)
    return sug
