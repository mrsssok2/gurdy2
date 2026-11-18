"""Per-key token-bucket rate limiter (used per client IP in demo mode). Thread-safe; the clock is injectable for tests."""
from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    """`burst` requests at once, refilled at one per `every_s` seconds. `allow(key)` -> (allowed, retry_after_s)."""

    def __init__(self, burst: int, every_s: float, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000):
        self.burst, self.every_s, self.clock, self.max_keys = float(burst), float(every_s), clock, max_keys
        self._lock = threading.Lock()
        self._b: dict[str, list[float]] = {}          # key -> [tokens, last time]

    def allow(self, key: str) -> tuple[bool, float]:
        now = self.clock()
        with self._lock:
            tokens, last = self._b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) / self.every_s)
            if tokens >= 1.0:
                self._b[key] = [tokens - 1.0, now]
                ok, wait = True, 0.0
            else:
                self._b[key] = [tokens, now]
                ok, wait = False, (1.0 - tokens) * self.every_s
            if len(self._b) > self.max_keys:                          # memory bound against many source addresses
                full = [k for k, (t, _) in self._b.items() if t >= self.burst - 1e-9]
                for k in full:
                    del self._b[k]
                if len(self._b) > self.max_keys:
                    for k in sorted(self._b, key=lambda k: self._b[k][1])[: len(self._b) // 2]:
                        del self._b[k]
            return ok, wait
