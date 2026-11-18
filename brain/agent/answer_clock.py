"""Answer timeouts in WALL-CLOCK time (never the sim clock: the mock runs 3x, Gazebo != 1x).

A question ("Shall I go?", "Did you mean ...?", "Which room?") can be answered until `timeout` seconds after the robot FINISHED
speaking it, and the answer counts if the user STARTED speaking within that time. Slow speech recognition (Whisper on CPU: 10 s) or
a slow LLM therefore never expires an answer. Typed input: the moment the line arrives."""
from __future__ import annotations

import time
from typing import Callable


class AnswerClock:
    def __init__(self, wall: Callable[[], float] = time.monotonic):
        self.wall = wall
        self.spoken_at: float | None = None      # when the robot last finished speaking
        self.turn_at: float | None = None        # when the user started the utterance being processed

    def begin_turn(self, started_at: float | None = None) -> None:
        self.turn_at = self.wall() if started_at is None else started_at

    def spoken_done(self, at: float | None = None) -> None:
        self.spoken_at = self.wall() if at is None else at

    def stamp(self) -> float:
        """Call when a question is created."""
        return self.wall()

    def expired(self, asked_at: float, timeout_s: float) -> bool:
        start = self.spoken_at if self.spoken_at is not None and self.spoken_at >= asked_at else asked_at
        now = self.turn_at if self.turn_at is not None else self.wall()
        return now - start > timeout_s
