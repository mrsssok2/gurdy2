"""One visitor's simulation for the STATIC demo: the real agent (keyword planner with the smart phrasings, safety, grounding, executor), the real mock
flat_lab world and the real dashboard telemetry, driven by a timer instead of server threads. Runs in the browser (Pyodide, in a Web Worker) or in
plain Python (tests). Everything is JSON-in / JSON-out through `call()` so the JavaScript side stays a thin transport.

The page is the normal dashboard (web/app); `static_backend.js` answers its REST / WebSocket calls from here. Each visitor automatically has a private
world because it lives in their own tab; a reload starts a new one."""
from __future__ import annotations

import json
import time

MAX_TEXT = 200
CHAT_KEEP = 200


class DemoSession:
    def __init__(self):
        from brain.voice.host import AgentHost
        from web.api.demo import make_agent
        from web.api.telemetry import MockTelemetry
        self.host = AgentHost(make_agent(), sim_speed=3.0)
        self.telemetry = MockTelemetry(self.host)
        self.chat: list[dict] = []
        self._seq = 0
        self.host.on_message(lambda text: self._add("robot", text))
        self._last = time.monotonic()

    # ------------------------------------------------------------------ chat log (same item shape as the server's ChatLog)
    def _add(self, role: str, text: str) -> dict:
        self._seq += 1
        item = {"seq": self._seq, "t": round(time.time(), 1), "role": role, "text": text}
        self.chat.append(item)
        del self.chat[:-CHAT_KEEP]
        return item

    def chat_since(self, seq: int = 0) -> list[dict]:
        return [dict(i) for i in self.chat if i["seq"] > seq]

    # ------------------------------------------------------------------ what the dashboard asks for
    def hello(self) -> dict:
        return {"type": "hello", "version": "static", "demo": True, "voice": False, "map": self.telemetry.map(), "chat": self.chat_since(0)}

    def state(self) -> dict:
        return {**self.telemetry.snapshot(), "demo": True}

    def tick(self) -> dict:
        """Advance the simulated world by the wall time since the last tick (x3, like the server demo) and collect what the robot announced."""
        now = time.monotonic()
        dt, self._last = min(now - self._last, 2.0), now          # a background tab must not fast-forward the robot by minutes
        before = self._seq
        self.host.step(dt)
        return {"state": self.state(), "chat": self.chat_since(before)}

    def say(self, text) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            return {"error": f"'text' must be a non-empty string of at most {MAX_TEXT} characters"}
        user = self._add("user", text.strip())
        reply = self._add("robot", self.host.say(text.strip()))
        return {"user": user, "reply": reply, "state": self.state()}

    def estop(self) -> dict:
        self.host.estop()
        return {"item": self._add("system", "Emergency stop sent."), "state": self.state()}


_session: DemoSession | None = None


def call(op: str, args_json: str = "[]") -> str:
    """Entry point used by the Web Worker: `op` is a DemoSession method name (or "reset"); returns JSON."""
    global _session
    if op == "reset" or _session is None:
        _session = DemoSession()
        if op == "reset":
            return "{}"
    return json.dumps(getattr(_session, op)(*json.loads(args_json)))
