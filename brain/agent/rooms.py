"""Room names by voice. Whisper cannot reliably hear a single letter ("D" -> "B", or dropped: "Go to room."), so rooms can be
said phonetically ("room Delta", "room D as in dog") and the robot reads the letter back ("Going to room B, Bravo.").
Pure strings: used by Agent.say before anything else (planner, guard, fuzzy matcher all see "room D")."""
from __future__ import annotations

import re

NATO = {"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo", "F": "Foxtrot", "G": "Golf", "H": "Hotel"}
_WORD2LETTER = {w.lower(): k for k, w in NATO.items()} | {"alfa": "A", "charly": "C"}
_WORDS = "|".join(sorted(_WORD2LETTER, key=len, reverse=True))
_AS_IN = r"(?:\s*,?\s*(?:as in|for|like)\s+[a-z]+)?"
_ROOM_PHON = re.compile(rf"\broom\s*,?\s+(?:({_WORDS})|([a-h])){_AS_IN}\b", re.I)
_ANSWER = re.compile(rf"^\W*(?:the\s+)?(?:room\s*,?\s+)?(?:({_WORDS})|([a-h])){_AS_IN}\W*$", re.I)
_CORRECTION = re.compile(r"^\W*(?:no|nope|wrong|not that|not that one|i said|i meant|no i said|no i meant)\W+(.+)$", re.I)
_MISSING = re.compile(r"\b(?:to|into|toward|towards)\s+(?:the\s+)?room\b(?!\s*,?\s*(?:[a-z0-9]+\b))", re.I)


def phonetic(letter: str) -> str | None:
    return NATO.get(letter.upper())


def normalize(text: str) -> str:
    """"room delta" / "room D as in dog" / "room d" -> "room D" (a room word followed by a single letter or a NATO word)."""
    def sub(m):
        return "room " + (_WORD2LETTER[m.group(1).lower()] if m.group(1) else m.group(2).upper())
    return _ROOM_PHON.sub(sub, text)


def answer_letter(text: str) -> str | None:
    """The whole utterance is a room answer: "B", "bravo", "room Delta", "D as in dog" -> "B" / "D"; anything else -> None."""
    m = _ANSWER.match(text.strip())
    if not m:
        return None
    return _WORD2LETTER[m.group(1).lower()] if m.group(1) else m.group(2).upper()


def correction_letter(text: str) -> str | None:
    """"no, Delta" / "no I said room D" -> "D"."""
    m = _CORRECTION.match(text.strip())
    return answer_letter(m.group(1)) if m else None


def mentioned_letters(text: str) -> list[str]:
    return re.findall(r"\broom ([A-Z])\b", normalize(text))


def known_letters(places: list[str]) -> list[str]:
    return sorted(m.group(1) for p in places if (m := re.fullmatch(r"room ([A-Z])", p)))


def listing(letters: list[str]) -> str:
    return letters[0] if len(letters) == 1 else ", ".join(letters[:-1]) + " or " + letters[-1]


def missing_letter_question(text: str, places: list[str]) -> str | None:
    """"Go to room." (the letter was not heard) -> "Which room: A, B or D?"."""
    letters = known_letters(places)
    if letters and _MISSING.search(text):
        return f"Which room: {listing(letters)}?"
    return None


def unknown_room_message(text: str, places: list[str]) -> str | None:
    """A room letter that is not on the map but could be a real room label (between A and two past the last known letter,
    at least up to D): "There is no room C. I know rooms A, B and D." Other letters stay with the "Did you mean ...?" repair."""
    letters = known_letters(places)
    if not letters:
        return None
    top = max(ord(letters[-1]) + 2, ord("D"))
    for ch in mentioned_letters(text):
        if ch not in letters and ord(ch) <= top:
            plural = "rooms" if len(letters) > 1 else "room"
            known = letters[0] if len(letters) == 1 else ", ".join(letters[:-1]) + " and " + letters[-1]
            return f"There is no room {ch}. I know {plural} {known}."
    return None
