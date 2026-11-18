"""Whisper vocabulary bias: an `initial_prompt` / `hotwords` string built from what the robot actually knows
(place names, object classes with their aliases, colours). Whisper then prefers "room B" over "room V".
Pure strings, no engine import. Used by FasterWhisperSTT, scripts/mic_transcribe.py and brain.voice.cli."""
from __future__ import annotations

from ..data_gen.lexicon import COLORS_BY_CLASS, OBJECT_WORDS, PLACES


COMMANDS = ["go to", "go home", "go back", "come back", "return home", "find", "look for", "search for", "where is", "what is near you",
            "what do you see", "stop", "patrol", "explore", "turn left", "turn right", "walk forward", "walk back", "save this place",
            "what is the battery", "continue", "cancel"]


def vocabulary(places=None, extra=()) -> list[str]:
    """Place names, object words (all aliases) and colours, de-duplicated, order kept."""
    words = list(places if places is not None else PLACES)
    words += [w for ws in OBJECT_WORDS.values() for w in ws]
    words += sorted({c for cs in COLORS_BY_CLASS.values() for c in cs})
    words += COMMANDS + list(extra)
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def initial_prompt(places=None) -> str:
    """A short example command in the robot's own vocabulary (Whisper treats the prompt as preceding text)."""
    p = list(places if places is not None else PLACES)
    rooms = [x for x in p if x.lower().startswith("room")] or p
    return (f"Commands for the robot Gurdy: go to {', '.join(rooms)}, the doorway or home. Go to room A and then go home. "
            f"Where is the red box? What is near you? Look for the barrel. Patrol. Stop. "
            f"Find the red box, blue box, yellow box, green box, barrel, fire extinguisher, traffic cone, toolbox, mannequin or chair.")


def hotwords(places=None) -> str:
    return " ".join(vocabulary(places))
