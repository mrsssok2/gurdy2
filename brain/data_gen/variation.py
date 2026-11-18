"""Surface-form variation for SFT data v2: casual, slang, typos, indirect / polite wrappers, run-on clauses, fillers.

Pure text -> text, seeded `random.Random`, no model, no audio. Gold plans are NOT changed by a variation, so every
transformation keeps the slot words (numbers, colours, directions, class words, place words) recoverable:
  * slot words that are short (colours, directions, number words) are never touched by typos;
  * long slot words (extinguisher, doorway, ...) only get ONE small edit and only sometimes;
  * stop/halt/emergency words and unsafe verbs are never introduced or removed (the e-stop fast path and the unsafe-text rule
    run before the LLM and must keep their meaning).
The generator (generate.py) still runs the real agent on every varied dialogue, so a variation that changes the observed
behaviour (e.g. breaks a save cue) is dropped there, not here.
"""
from __future__ import annotations

import random
import re

_SLOT_SHORT = {"room", "that", "this", "them", "those", "other", "second", "again", "once", "more", "same", "then", "there", "red", "blue", "green", "yellow", "orange", "white", "black", "brown", "grey", "gray", "left", "right", "forward", "forwards", "backward", "backwards", "back", "ahead", "straight", "degrees", "metres", "meters",
               "home", "box", "a", "b", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
_NEVER = re.compile(r"\b(stop|halt|freeze|emergency|e-?stop|disable|bypass|ignore|override|safety|hit|ram|push|crash|smash|kick|knock)\b", re.I)
_IMPERATIVE = {"go", "walk", "move", "turn", "rotate", "find", "look", "search", "locate", "inspect", "check", "explore", "map", "patrol", "head",
               "come", "return", "save", "remember", "keep", "avoid", "navigate", "drive", "travel", "proceed", "approach", "scout", "spin", "step",
               "take", "show", "tell", "give", "list", "mark", "call", "name", "store", "allow", "remove", "pause", "resume", "continue", "cancel"}

PREFIXES = ["yo ", "hey ", "ok so ", "um ", "uh ", "so ", "right ", "alright ", "hey gurdy can you ", "gurdy buddy ", "oh and ", "quick one, ", "hmm ",
            "umm ", "okay gurdy, ", "listen, ", "right then, ", "ok ", "well ", "hey gurdy, ", "sooo ", "alright gurdy "]
SUFFIXES = [" pls", " plz", " thanks", " thx", " mate", " buddy", " real quick", " when you can", " if you can", " asap", " ya", " cheers", " would ya",
            " thank you", " if thats ok", " dude"]
WRAPPERS = ["i would really like it if you could {x}", "it would be great if you could {x}", "would you be able to {x}", "do you think you could {x}",
            "i need you to {x}", "if it is not too much trouble, {x}", "i was wondering if you could {x}", "any chance you could {x}", "maybe {x}",
            "try to {x}", "just {x}", "{x}, that would be great", "{x} if that is ok", "could you possibly {x}", "i want you to {x}", "can you {x}",
            "be a dear and {x}", "i would appreciate it if you would {x}", "hey, can you {x}", "you should {x}", "lets {x}", "{x} for me"]
SLANG = [(r"\bplease\b", ["pls", "plz", "pleeease"]), (r"\byou\b", ["u", "ya"]), (r"\byour\b", ["ur"]), (r"\bwhat is\b", ["whats", "what's"]),
         (r"\bwhere is\b", ["wheres", "where's"]), (r"\bwant to\b", ["wanna"]), (r"\bgoing to\b", ["gonna"]), (r"\bgot to\b", ["gotta"]),
         (r"\bokay\b", ["ok", "k"]), (r"\bthank you\b", ["thx", "ty"]), (r"\bcould you\b", ["could ya", "can u"]), (r"\bare you\b", ["r u", "are u"]),
         (r"\bhow much\b", ["how much"]), (r"\bkind of\b", ["kinda"]), (r"\bgive me\b", ["gimme"]), (r"\blet me\b", ["lemme"]),
         (r"\bright now\b", ["rn", "right now"]), (r"\bmetres\b", ["m", "meters", "metres"]), (r"\bdegrees\b", ["deg", "degrees"]),
         (r"\bforward\b", ["forward", "fwd", "forwards"])]
JOINERS = [(r" and then ", [" then ", " n then ", ", ", " and ", " after that ", ", then ", " and after that "]),
           (r", then ", [" then ", " and then ", " n then "]), (r" and ", [" n ", " and "]), (r" after that ", [" then ", ", "])]
FILLERS = ["uh", "um", "like", "you know", "basically", "i mean"]
_KEYS = {"a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
         "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy", "u": "yhji",
         "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"}


def _typo(rng: random.Random, w: str) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "double", "key"])
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "drop":
        return w[:i] + w[i + 1:]
    if op == "double":
        return w[:i] + w[i] + w[i:]
    c = w[i]
    return w[:i] + (rng.choice(_KEYS[c]) if c in _KEYS else c) + w[i + 1:]


def _typos(rng: random.Random, text: str, n: int) -> str:
    toks = text.split(" ")
    idx = [i for i, t in enumerate(toks) if re.fullmatch(r"[a-z']{4,}", t) and t not in _SLOT_SHORT and not _NEVER.search(t)
           and not t.isdigit()]
    rng.shuffle(idx)
    for i in idx[:n]:
        toks[i] = _typo(rng, toks[i])
    return " ".join(toks)


def vary(rng: random.Random, say: str, p: float = 0.8) -> str:
    """One varied surface form of `say` (the original with probability 1 - p)."""
    if rng.random() > p or _NEVER.search(say):
        return say
    t = say.strip()
    low = t.rstrip("?.!").lower()
    is_q = t.endswith("?") or low.split(" ")[0] in ("what", "where", "which", "who", "how", "why", "is", "are", "do", "did", "have", "can", "any", "anything")
    first = low.split(" ")[0] if low else ""
    out = low
    r = rng.random()
    # 1. indirect / polite wrapper (imperatives only)
    if first in _IMPERATIVE and not is_q and r < 0.28:
        out = rng.choice(WRAPPERS).format(x=out)
    # 2. slang / abbreviations
    if rng.random() < 0.45:
        for pat, subs in rng.sample(SLANG, k=len(SLANG)):
            if re.search(pat, out) and rng.random() < 0.6:
                out = re.sub(pat, rng.choice(subs), out, count=1)
    # 3. run-on / joiner changes
    if rng.random() < 0.5:
        for pat, subs in JOINERS:
            if re.search(pat, out):
                out = re.sub(pat, rng.choice(subs), out, count=1)
                break
    # 4. disfluency
    if rng.random() < 0.12:
        toks = out.split(" ")
        toks.insert(rng.randrange(1, max(2, len(toks))), rng.choice(FILLERS))
        out = " ".join(toks)
    # 5. casual prefix / suffix
    if rng.random() < 0.3:
        out = rng.choice(PREFIXES) + out
    if rng.random() < 0.22:
        out = out + rng.choice(SUFFIXES)
    # 6. typos (0-2)
    k = rng.choices([0, 1, 2], [0.62, 0.28, 0.10])[0]
    if k:
        out = _typos(rng, out, k)
    # 7. dropped / added punctuation + capitalisation
    r2 = rng.random()
    if r2 < 0.12:
        out = out[:1].upper() + out[1:] + ("?" if is_q else ".")
    elif r2 < 0.2 and is_q:
        out += "?"
    elif r2 < 0.24:
        out = out.upper() if len(out) < 24 else out
    return out.strip()
