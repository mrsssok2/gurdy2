"""Typed chat that survives Gurdy's asynchronous messages (mission progress, alerts, new objects).

Plain `input()` + a background thread that prints would mangle the line being typed. Here the thread erases the prompt line,
prints the message, then redraws the prompt and whatever was typed so far (`readline.get_line_buffer()`), so typing is never lost.
Only when stdin is a terminal; with a pipe (scripted runs) messages are printed plainly.
The agent is not thread-safe: one lock covers `say()` and `pump()` (a `say()` with an LLM call holds it ~1 s; messages wait).
"""
from __future__ import annotations

import re
import sys
import threading
import time
from typing import Callable


def capitalise_places(text: str, names: list[str]) -> str:
    """'in room a' -> 'in Room A' for the given place names (display only; the agent still works with the lower-case names)."""
    for n in sorted(names, key=len, reverse=True):
        shown = " ".join(w.upper() if len(w) == 1 else w.capitalize() for w in n.replace("_", " ").split())
        text = re.sub(rf"(?<!\w){r'[ _]'.join(map(re.escape, n.replace('_', ' ').split(' ')))}(?!\w)", shown, text, flags=re.I)
    return text


def redraw_line(message: str, prompt: str, typed: str) -> str:
    """Terminal bytes: clear the current line, print `message`, put the prompt and the typed text back."""
    return f"\r\x1b[2K{message}\n{prompt}{typed}"


class Chat:
    def __init__(self, agent, place_names: Callable[[], list[str]], prompt: str = "> ", pump_s: float = 0.25, out=None, tty: bool | None = None):
        self.agent, self.names, self.prompt, self.pump_s = agent, place_names, prompt, pump_s
        self.out = out or sys.stdout
        self.tty = sys.stdin.isatty() if tty is None else tty
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _typed(self) -> str:
        try:
            import readline
            return readline.get_line_buffer()
        except ImportError:
            return ""

    def say_async(self, message: str) -> None:
        message = capitalise_places(message, self.names())
        if self.tty:
            self.out.write(redraw_line(message, self.prompt, self._typed()))
        else:
            self.out.write(message + "\n")
        self.out.flush()

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                time.sleep(self.pump_s)
                with self.lock:
                    msgs = self.agent.pump()
                for m in msgs:
                    self.say_async("Gurdy: " + m)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def ask(self, text: str) -> str:
        with self.lock:
            reply = self.agent.say(text).reply
        return capitalise_places(reply, self.names())
