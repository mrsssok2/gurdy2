"""Words, number rendering, phrasing banks, STT confusion table. Pure data + tiny helpers (no random state of its own)."""
from __future__ import annotations

import hashlib
import re

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def words(n: int) -> str:
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return TENS[t * 10] + (" " + UNITS[u] if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return UNITS[h] + " hundred" + (" and " + words(r) if r else "")
    return str(n)


def num(rng, n: int) -> str:
    """An integer as digits or words."""
    return str(n) if rng.random() < 0.45 else words(n)


def dist_phrase(rng, m: float) -> str:
    """Metres as people say them: '2 metres', 'two meters', 'fifty centimetres', 'one and a half metres', 'half a metre'."""
    unit = rng.choice(["metres", "meters"])
    if m < 1.0 and abs(m * 100 - round(m * 100)) < 1e-6 and rng.random() < 0.6:
        cm = int(round(m * 100))
        return f"{num(rng, cm)} {rng.choice(['centimetres', 'centimeters'])}"
    if m == 0.5 and rng.random() < 0.5:
        return "half a metre"
    if m == 1.0 and rng.random() < 0.4:
        return "a metre"
    if m == 1.5 and rng.random() < 0.6:
        return "one and a half " + unit
    if abs(m - round(m)) < 1e-9:
        return f"{num(rng, int(round(m)))} {unit if round(m) != 1 else unit[:-1] if False else unit}"
    return f"{m:g} {unit}"


OBJECT_WORDS = {
    "box": ["box", "box", "box", "crate", "cardboard box"],
    "barrel": ["barrel", "barrel", "drum"],
    "fire_extinguisher": ["fire extinguisher", "fire extinguisher", "extinguisher"],
    "traffic_cone": ["traffic cone", "cone", "traffic cone"],
    "toolbox": ["toolbox", "toolbox", "tool box"],
    "mannequin": ["mannequin", "mannequin", "dummy"],
    "chair": ["chair"],
}
COLORS_BY_CLASS = {"box": ["red", "blue", "yellow", "green", "white", "orange"], "barrel": ["green", "blue", "red", "yellow"],
                   "fire_extinguisher": ["red"], "traffic_cone": ["orange"], "toolbox": ["red", "blue", "yellow"],
                   "mannequin": ["grey"], "chair": ["black", "brown", "grey", "white"]}
PLACES = ["home", "room A", "room B", "doorway"]
SAVED_PLACE_NAMES = ["workbench", "charging dock", "storage corner", "window", "lab bench", "meeting spot", "parking spot"]
UNKNOWN_PLACES = ["kitchen", "bathroom", "garage", "office", "corridor", "basement", "balcony", "table", "roof", "garden"]
DIRS_WALK = {"forward": ["forward", "forwards", "ahead", "straight ahead"], "backward": ["backward", "backwards", "back"],
             "left": ["left"], "right": ["right"]}
LEGS = ["front_left", "middle_left", "rear_left", "front_right", "middle_right", "rear_right"]


def place_say(rng, place: str) -> str:
    if place == "home":
        return rng.choice(["home", "home", "the start", "the home position"])
    if place == "doorway":
        return rng.choice(["the doorway", "the doorway", "the door"])
    return place if rng.random() < 0.9 else place.lower()


def pick(rng, templates: list[str], split: str) -> str:
    """Choose a phrasing template. Train and val use disjoint templates (same mix, other templates, design 6.2):
    a template belongs to val iff its stable hash mod 6 == 5."""
    def is_val(t: str) -> bool:
        return int(hashlib.md5(t.encode()).hexdigest(), 16) % 6 == 5
    pool = [t for t in templates if is_val(t) == (split == "val")]
    return rng.choice(pool or templates)


PREFIXES = ["", "", "", "", "please ", "could you ", "can you ", "i want you to ", "hey gurdy ", "gurdy ", "okay ", "now "]
SUFFIXES = ["", "", "", "", "", " please", " now", " for me"]


def polite(rng, text: str) -> str:
    return rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)


def surface(rng, text: str, question: bool = False) -> str:
    """Typed-vs-Whisper surface form: mostly lower-case without punctuation (like the test set), sometimes capitalised."""
    r = rng.random()
    if r < 0.62:
        return text
    t = text[:1].upper() + text[1:]
    if r < 0.85:
        return t + ("?" if question else ".")
    return t


# STT confusions: word the speech recogniser tends to produce -> the word that was meant (used by the slot check, design 6.1.5)
CONFUSIONS = {
    "lift": "left", "laughed": "left", "write": "right", "rite": "right", "four": "for", "for": "four", "to": "two", "too": "two",
    "read": "red", "bloom": "blue", "blew": "blue", "yello": "yellow", "grain": "green", "comb": "cone", "cohn": "cone",
    "petrol": "patrol", "hole": "home", "whole": "home", "be": "b", "bee": "b", "ay": "a", "stob": "stop", "stap": "stop",
    "fine": "find", "bare": "barrel", "barrow": "barrel", "metres": "meters", "tool": "toolbox", "mannequin": "mannequin",
    "ex-tinguisher": "extinguisher", "explorer": "explore", "ate": "eight", "won": "one", "tree": "three", "sex": "six",
    "forty": "forty", "nein": "nine", "tin": "ten",
}
_NORM = re.compile(r"[^a-z0-9 ]+")


def norm(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())
