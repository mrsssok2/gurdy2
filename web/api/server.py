"""Mission-control API server (Phase 7 step 1). Standard library only: threading HTTP server + WebSocket (web/api/ws.py).

It sits in front of an `AgentHost` / `RemoteHost` (brain/voice/host.py): every command goes through `host.say` / `host.estop`, so Phase 5 safety
(checks, confirmations, e-stop fast path) is untouched. Telemetry comes from a `Telemetry` source. Design: docs/phase7_design.md.

Auth: one shared token (env GURDY_WEB_TOKEN, else random and printed once by the CLI, never written anywhere). Everything under /api and the
WebSocket needs it except /api/health; static files are public (they contain no secrets)."""
from __future__ import annotations

import hmac
import json
import mimetypes
import os
import queue
import secrets
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import ws
from .telemetry import MockTelemetry, Telemetry

VERSION = "0.1"
MAX_BODY = 64 * 1024
MAX_TEXT = 500
MAX_WS_CLIENTS = 8
CHAT_KEEP = 200
DEFAULT_STATIC = Path(__file__).resolve().parents[1] / "app"


class ChatLog:
    """Shared conversation + announcements, newest last, monotonically numbered, so every device shows the same history and a client that
    reconnects asks only for what it missed (`since`)."""

    def __init__(self, on_add=None):
        self._lock = threading.Lock()
        self._items: deque[dict] = deque(maxlen=CHAT_KEEP)
        self._seq = 0
        self.on_add = on_add

    def add(self, role: str, text: str) -> dict:
        with self._lock:
            self._seq += 1
            item = {"seq": self._seq, "t": round(time.time(), 1), "role": role, "text": text}
            self._items.append(item)
        if self.on_add:
            self.on_add(item)
        return item

    def since(self, seq: int = 0) -> list[dict]:
        with self._lock:
            return [dict(i) for i in self._items if i["seq"] > seq]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


class _WsClient:
    """One writer thread + bounded queue per client: a stalled phone can never block the broadcast to the others (it is dropped when full)."""

    def __init__(self, sock: socket.socket, on_dead):
        self.sock, self.on_dead = sock, on_dead
        self.q: queue.Queue = queue.Queue(maxsize=50)
        self.dead = False
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, frame: bytes) -> None:
        if self.dead:
            return
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.kill()

    def _run(self) -> None:
        while True:
            frame = self.q.get()
            if frame is None:
                return
            try:
                self.sock.sendall(frame)
            except OSError:
                self.kill()
                return

    def kill(self) -> None:
        if self.dead:
            return
        self.dead = True
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.on_dead(self)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):      # a phone going out of Wi-Fi range is normal, not a traceback
            return
        super().handle_error(request, client_address)


class GurdyWebServer:
    def __init__(self, host, telemetry: Telemetry | None = None, token: str | None = None, static_dir: str | Path | None = DEFAULT_STATIC,
                 bind: str = "127.0.0.1", port: int = 8080, state_s: float = 0.5):
        self.host, self.state_s = host, state_s
        self.telemetry = telemetry or MockTelemetry(host)
        self.token = token or os.environ.get("GURDY_WEB_TOKEN") or secrets.token_urlsafe(16)
        self.static_dir = Path(static_dir).resolve() if static_dir else None
        self.chat = ChatLog(on_add=lambda item: self.broadcast({"type": "chat", **item}))
        self._clients: list[_WsClient] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        if hasattr(host, "on_message"):
            host.on_message(lambda text: self.chat.add("robot", text))        # pump announcements / alerts
        server = self
        handler = type("Handler", (_Handler,), {"app": server})
        self.httpd = _Server((bind, port), handler)
        self.port = self.httpd.server_address[1]
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "GurdyWebServer":
        for fn in (lambda: self.httpd.serve_forever(poll_interval=0.05), self._broadcast_loop):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def close(self) -> None:
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        for c in list(self._clients):
            c.kill()

    # ------------------------------------------------------------------ helpers used by the handler
    def check_token(self, given: str | None) -> bool:
        return bool(given) and hmac.compare_digest(given.encode("utf-8", "replace"), self.token.encode())

    def state(self) -> dict[str, Any]:
        try:
            return self.telemetry.snapshot()
        except Exception as e:                       # noqa: BLE001  a broken source must not kill the stream
            return {"connected": False, "error": f"{type(e).__name__}: {e}"}

    def say(self, text: str) -> dict:
        self.chat.add("user", text)
        reply = self.host.say(text)
        return self.chat.add("robot", reply)

    def estop(self) -> None:
        self.host.estop()
        self.chat.add("system", "Emergency stop sent.")

    def broadcast(self, obj: dict) -> None:
        frame = ws.encode_frame(ws.OP_TEXT, json.dumps(obj, separators=(",", ":")).encode())
        with self._clients_lock:
            clients = list(self._clients)
        for c in clients:
            c.send(frame)

    def add_client(self, sock: socket.socket) -> _WsClient | None:
        with self._clients_lock:
            if len(self._clients) >= MAX_WS_CLIENTS:
                return None
            c = _WsClient(sock, self._drop_client)
            self._clients.append(c)
            return c

    def _drop_client(self, c: _WsClient) -> None:
        with self._clients_lock:
            if c in self._clients:
                self._clients.remove(c)

    @property
    def n_clients(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def _broadcast_loop(self) -> None:
        while not self._stop.wait(self.state_s):
            if self.n_clients:
                self.broadcast({"type": "state", **self.state()})


class _Handler(BaseHTTPRequestHandler):
    app: GurdyWebServer
    protocol_version = "HTTP/1.1"
    server_version = "GurdyWeb/" + VERSION

    def log_message(self, *a) -> None:              # never log: WebSocket URLs carry the token in the query string
        pass

    # ---- plumbing
    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def _authed(self, query: dict) -> bool:
        h = self.headers.get("Authorization", "")
        given = h[7:] if h.startswith("Bearer ") else (query.get("token") or [None])[0]
        return self.app.check_token(given)

    def _body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True                 # the body was not read: it must not be parsed as the next request
            self._err(413 if n > MAX_BODY else 400, "bad or too large body")
            return None
        try:
            obj = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self._err(400, "body must be JSON")
            return None
        if not isinstance(obj, dict):
            self._err(400, "body must be a JSON object")
            return None
        return obj

    # ---- routes
    def do_GET(self) -> None:
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/health":
            return self._json({"ok": True, "version": VERSION})
        if u.path == "/ws":
            return self._websocket(q)
        if u.path.startswith("/api/"):
            if not self._authed(q):
                return self._err(401, "missing or wrong token")
            if u.path == "/api/state":
                return self._json(self.app.state())
            if u.path == "/api/map":
                return self._json(self.app.telemetry.map())
            if u.path == "/api/messages":
                try:
                    since = int((q.get("since") or ["0"])[0])
                except ValueError:
                    return self._err(400, "since must be an integer")
                return self._json({"messages": self.app.chat.since(since), "seq": self.app.chat.last_seq})
            return self._err(404, "unknown route")
        self._static(u.path)

    def do_POST(self) -> None:
        u = urlsplit(self.path)
        if not u.path.startswith("/api/"):
            return self._err(404, "unknown route")
        if not self._authed(parse_qs(u.query)):
            return self._err(401, "missing or wrong token")
        if u.path == "/api/estop":
            self.app.estop()
            return self._json({"ok": True})
        if u.path == "/api/say":
            body = self._body()
            if body is None:
                return
            text = body.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
                return self._err(400, f"'text' must be a non-empty string of at most {MAX_TEXT} characters")
            item = self.app.say(text.strip())
            return self._json({"reply": item["text"], "seq": item["seq"]})
        self._err(404, "unknown route")

    def _static(self, path: str) -> None:
        root = self.app.static_dir
        if root is None:
            return self._err(404, "no static files")
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (root / rel).resolve()
        if not f.is_relative_to(root) or not f.is_file():
            return self._err(404, "not found")
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    # ---- websocket
    def _websocket(self, q: dict) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
            return self._err(400, "WebSocket upgrade required")
        if not self._authed(q):
            return self._err(401, "missing or wrong token")
        client = self.app.add_client(self.connection)
        if client is None:
            return self._err(503, "too many clients")
        self.close_connection = True
        self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Accept: {ws.accept_key(key)}\r\n\r\n").encode())
        self.wfile.flush()
        send_json = lambda o: client.send(ws.encode_frame(ws.OP_TEXT, json.dumps(o).encode()))
        send_json({"type": "hello", "version": VERSION, "map": self.app.telemetry.map(), "chat": self.app.chat.since(0)})
        send_json({"type": "state", **self.app.state()})
        try:
            while True:
                msg = ws.read_message(self.rfile, client.send)
                if msg is None:
                    break
                op, payload = msg
                if op == ws.OP_TEXT:
                    self._ws_command(payload, send_json)
        except ws.WsError as e:
            client.send(ws.encode_frame(ws.OP_CLOSE, e.code.to_bytes(2, "big")))
        except (EOFError, OSError):
            pass
        finally:
            client.send(ws.encode_frame(ws.OP_CLOSE, (1000).to_bytes(2, "big")))
            client.kill()

    def _ws_command(self, payload: bytes, send_json) -> None:
        try:
            cmd = json.loads(payload)
            op = cmd["op"]
        except (ValueError, KeyError, TypeError):
            return send_json({"type": "error", "error": "expected a JSON object with 'op'"})
        if op == "estop":                                             # inline: no thread, no lock
            self.app.estop()
            send_json({"type": "ack", "op": "estop"})
        elif op == "say":
            text = cmd.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
                return send_json({"type": "error", "error": "bad 'text'"})
            # off this reading thread: a slow LLM turn must not delay an estop arriving on the same socket.
            threading.Thread(target=self.app.say, args=(text.strip(),), daemon=True).start()
            send_json({"type": "ack", "op": "say"})
        else:
            send_json({"type": "error", "error": f"unknown op {op!r}"})
