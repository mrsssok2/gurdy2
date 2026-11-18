"""One interface between the voice loop and the agent, in-process or over TCP.

`AgentHost` wraps an Agent: one lock (the agent is not thread-safe), a pump thread that turns executor events / alerts into
spoken messages, and the fast stop path (`estop()` does NOT wait for the lock: a stop must not queue behind a 1 s LLM call).
`serve()` exposes an AgentHost as JSON lines over TCP, `RemoteHost` is the client with the same methods. That split exists
because the ROS agent lives in the sim image (rclpy) while the microphone / Whisper / Piper live in the `voice` image.

Protocol (one JSON object per line):  client -> {"op":"say","id":n,"text":..,"conf":..} | {"op":"estop"} | {"op":"state"}
                                      server -> {"op":"reply","id":n,"text":..} | {"op":"msg","text":..} | {"op":"state","moving":b,"places":[..]}
No authentication: bind it to the host only (docker -p 127.0.0.1:PORT:PORT)."""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable


class AgentHost:
    def __init__(self, agent, pump_s: float = 0.25, sim_speed: float | None = None):
        self.agent, self.pump_s = agent, pump_s
        self.sim_speed = sim_speed          # mock world only: advance its virtual clock this many times faster than wall time (None: someone else does)
        self.lock = threading.Lock()
        self.listeners: list[Callable[[str], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- interface shared with RemoteHost
    def say(self, text: str, conf: float | None = None) -> str:
        with self.lock:
            return self.agent.say(text, conf).reply

    def estop(self) -> None:
        """Fast path: cancel the mission and stop the robot without taking the agent lock (idempotent)."""
        try:
            self.agent.executor.cancel()
        except Exception:
            pass
        self.agent.layer.call("stop")

    def is_moving(self) -> bool:
        return self.agent.executor.state == "running"

    def places(self) -> list[str]:
        return list(self.agent.layer.backend.known_places())

    def on_message(self, fn: Callable[[str], None]) -> None:
        self.listeners.append(fn)

    def start(self) -> None:
        def loop():
            last = time.monotonic()
            while not self._stop.is_set():
                time.sleep(self.pump_s)
                now = time.monotonic()
                with self.lock:
                    be = self.agent.layer.backend
                    if self.sim_speed and hasattr(be, "advance"):      # without this a mock trip never ends
                        be.advance((now - last) * self.sim_speed)
                    msgs = self.agent.pump()
                last = now
                for m in msgs:
                    for fn in list(self.listeners):
                        fn(m)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()


def serve(host: AgentHost, port: int, bind: str = "0.0.0.0") -> threading.Thread:
    """Serve `host` to one client at a time (a reconnect replaces the old one). Returns the accept thread (daemon)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(1)
    clients: list = []

    def push(m: str):
        for f in list(clients):
            try:
                f.write(json.dumps({"op": "msg", "text": m}) + "\n"); f.flush()
            except OSError:
                pass
    host.on_message(push)

    def handle(conn):
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        clients.append(f)
        wlock = threading.Lock()

        def send(o):
            with wlock:
                try:
                    f.write(json.dumps(o) + "\n"); f.flush()
                except OSError:
                    pass
        try:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                op = r.get("op")
                if op == "estop":
                    host.estop(); send({"op": "ok"})
                elif op == "state":
                    send({"op": "state", "moving": host.is_moving(), "places": host.places()})
                elif op == "say":
                    # off the reading thread: a slow LLM turn must not delay an {"op":"estop"} that arrives meanwhile
                    threading.Thread(target=lambda r=r: send({"op": "reply", "id": r.get("id"), "text": host.say(r.get("text", ""), r.get("conf"))}),
                                     daemon=True).start()
        finally:
            if f in clients:
                clients.remove(f)
            conn.close()

    def accept():
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    return t


class RemoteHost:
    """Client for `serve()`. `is_moving()` / `places()` use a state cache refreshed every `state_s` seconds in the background."""

    def __init__(self, addr: str, state_s: float = 0.5, timeout_s: float = 60.0):
        h, p = addr.rsplit(":", 1)
        self.sock = socket.create_connection((h, int(p)), timeout=10)
        self.sock.settimeout(None)
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self.wl = threading.Lock()
        self.timeout_s, self.state_s = timeout_s, state_s
        self.listeners: list[Callable[[str], None]] = []
        self._replies: dict[int, str] = {}
        self._cv = threading.Condition()
        self._id = 0
        self._state = {"moving": False, "places": []}
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._poll, daemon=True).start()

    def _send(self, o):
        with self.wl:
            self.f.write(json.dumps(o) + "\n"); self.f.flush()

    def _read(self):
        try:
            for line in self.f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["op"] == "reply":
                    with self._cv:
                        self._replies[r["id"]] = r["text"]; self._cv.notify_all()
                elif r["op"] == "msg":
                    for fn in list(self.listeners):
                        fn(r["text"])
                elif r["op"] == "state":
                    self._state = r
        except (OSError, ValueError):
            pass
        self._closed = True
        with self._cv:
            self._cv.notify_all()

    def _poll(self):
        while not self._closed:
            try:
                self._send({"op": "state"})
            except OSError:
                return
            time.sleep(self.state_s)

    def say(self, text: str, conf: float | None = None) -> str:
        with self._cv:
            self._id += 1
            i = self._id
        self._send({"op": "say", "id": i, "text": text, "conf": conf})
        end = time.monotonic() + self.timeout_s
        with self._cv:
            while i not in self._replies and not self._closed and time.monotonic() < end:
                self._cv.wait(0.5)
            return self._replies.pop(i, "I lost the connection to the robot." if self._closed else "The robot did not answer in time.")

    def estop(self) -> None:
        self._send({"op": "estop"})

    def is_moving(self) -> bool:
        return bool(self._state.get("moving"))

    def places(self) -> list[str]:
        return list(self._state.get("places", []))

    def on_message(self, fn) -> None:
        self.listeners.append(fn)

    def start(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
