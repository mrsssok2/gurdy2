# Phase 7 design — Mission control (web dashboard + phone)

Status (2026-11-18): **design + step 1 only** (API skeleton on the mock world). No number below is measured. English only.
Goal: one page, usable on a laptop and on a phone on the same Wi-Fi, that shows what the robot is doing and lets you talk to it and stop it.

## 1. What the dashboard shows

| Panel | Content | Source |
|---|---|---|
| Map + pose | room walls / obstacles, no-go zones, known objects (the robot's memory, not ground truth), robot pose + heading, last ~200 poses as a trail, saved places | `GET /api/map` once, pose + objects in the live stream |
| Camera + detections | latest camera frame (JPEG, ≤ 320x240, ~2 fps) with boxes, list of confirmed objects | ROS image topic via the ROS adapter (step 5); mock: none (panel says "no camera in mock") |
| Chat | typed messages to the agent, replies, spoken-style announcements and alerts, same history on every connected device | `AgentHost.say` / pump messages |
| Mission status | mode (idle / moving / paused / estop), current task, speed, uptime | `status()` |
| Alerts | active alerts, newest first, severity colour | `status().alerts_active`, pump messages |
| Battery / legs | battery % (labelled SIMULATED while it is), six leg states | `status()` |
| E-stop | always visible, big, one tap, no confirmation dialog; uses the fast stop path (does not wait for the agent lock) | `POST /api/estop`, WS `{"op":"estop"}` |
| Incidents (Phase 6) | list from the incident store | later, after Phase 6 step 2+; page is a stub until then |

## 2. How it talks to the robot

```
 browser / phone ──HTTP+WS──►  web/api server (stdlib, one process)
                               ├─ chat, e-stop ─►  AgentHost  (in-process)  or  RemoteHost (the existing JSON-lines TCP bridge)
                               └─ telemetry     ─►  Telemetry source: MockTelemetry (reads the MockBackend)
                                                    RosTelemetry   (step 5: rclpy subscriptions: odom/TF, /map, joint/leg health, battery, /gurdy/alerts, camera)
```

- The server does **not** talk to the LLM or ROS directly for commands: every command goes through `AgentHost.say` / `estop`, so safety checks, confirmations and the e-stop fast path stay exactly as in Phase 5.
- **Telemetry** is a small protocol (`snapshot()` -> dict, `map()` -> dict). The mock version reads the backend (under `host.lock`, with a short timeout so a slow LLM turn does not freeze the stream; a stale snapshot is flagged `stale: true`). The ROS version runs inside the sim image (it needs rclpy), so for the real robot the API server runs in the same container as `run_text --serve`, not behind the TCP bridge (the bridge only carries say / estop / state).
- **REST**: `GET /api/health` (no auth), `/api/state`, `/api/map`, `/api/messages?since=N`, `POST /api/say`, `POST /api/estop`.
- **WebSocket** `/ws?token=...`: server pushes `{"type":"state",...}` every 0.5 s and `{"type":"chat",...}` for every chat / alert entry; the client may send `{"op":"say","text":...}` and `{"op":"estop"}`. REST stays complete, so a client without WebSockets (or a test) works with polling.
- Frames and sizes: state ~1–2 KB at 2 Hz; map sent once (vector geometry, not a grid); camera frames are separate binary messages, added in step 5.

## 3. Tech choice (simple, offline-capable)

- **Server: Python standard library only** (`http.server` threading server + a ~100-line RFC 6455 WebSocket). Reasons: nothing to install, runs in the existing `brain` and ROS images and on the host, no CDN, works with no internet. FastAPI is planned for Phase 6 (ops API); the web server does not need it. If we later want OpenAPI / typed models, the routes are small enough to move.
- **Frontend: static HTML + plain ES modules + canvas, no build step, no npm, no CDN.** The map is a 2D canvas (the robot is a 2D robot in the mock / Nav2 world). Installable on a phone as a PWA (manifest + service worker caches the shell, so the page opens with the robot offline and shows "disconnected").
- **This deviates from the ROADMAP wording** ("Next.js/TypeScript, rosbridge, three.js"). Why: Node is not in our images, a Next.js build adds ~1 GB of tooling to an 8 GB machine, and rosbridge adds a second protocol next to the agent bridge. The 3D three.js robot view moves to an optional step 8 (vendored three.js file, still offline). If you prefer Next.js for the portfolio, the API stays identical and the frontend can be swapped — decision for the user.

## 4. Phone access and hosting

1. **Same Wi-Fi first.** Server binds `127.0.0.1` by default; `--bind 0.0.0.0` for LAN. WSL2 detail: the Windows host must forward the port to WSL (`netsh interface portproxy` or mirrored networking in `.wslconfig`) and allow it in the firewall; documented with exact commands in step 6, tested with a real phone.
2. **Hosting later** (not core): the robot stays at home; a small tunnel (Cloudflare Tunnel / Tailscale) exposes the same server over HTTPS. Needs HTTPS (service worker + secure cookies), so no plain-HTTP exposure to the internet — ever.

## 5. Auth basics

- One **shared access token**, read from the environment variable `GURDY_WEB_TOKEN`; if unset the server generates a random one at start and prints it to the terminal only (never written to a file, a log or git; hard rule 1). Compared in constant time.
- Every `/api/*` route except `/api/health` and the WebSocket needs it: `Authorization: Bearer <token>`, or `?token=` for the WebSocket (browsers cannot set headers there). The page asks for the token once and keeps it in `localStorage`.
- Request bodies ≤ 64 KB, WebSocket messages ≤ 64 KB, max 8 simultaneous WebSocket clients, JSON only, static files served from one directory with path-traversal checks.
- **E-stop needs the token too** (an unauthenticated stop endpoint is a DoS lever), but the token is on the page after first login, so it is one tap. A hardware / voice stop remains the primary safety path.
- Not in scope: user accounts, roles, TLS termination (done by the tunnel). Honest limit: a shared token over plain HTTP on a LAN can be sniffed; fine at home, not for hosting without HTTPS.

## 6. Step plan

| # | Step | Test |
|---|---|---|
| 1 | **API server skeleton + MockTelemetry**: REST, WebSocket, token auth, chat log, e-stop, static files (placeholder page) | unit tests with the real `MockBackend`, no Docker |
| 2 | Frontend v1: map + pose + trail, status / battery / legs, alerts, e-stop (static page, canvas) | JS-free parts unit-tested; manual check in a browser |
| 3 | Chat panel + mission status + "token / disconnected" UX; reconnect with backoff | manual, plus server tests for `since` replay |
| 4 | `compose` profile `web` (+ `scripts/start_web.sh`): mock demo with the keyword agent; measure RAM | measured RAM, one-command start |
| 5 | **RosTelemetry** (pose, map, legs, battery, alerts) + camera frames with detection boxes; runs next to `run_text --serve` in the sim image | live Gazebo run, after the 5.11 eval is DONE |
| 6 | Phone: PWA manifest + service worker, WSL2 LAN forwarding commands, test on a real phone | user's phone, same Wi-Fi |
| 7 | Incidents page (after Phase 6 resumes) + CI job for the web tests + README GIFs | CI green |
| 8 | Optional: three.js 3D view (vendored), tunnel hosting notes | — |

Phase 6 is paused after its step 1 (`405c10c`) and resumes at step 2 after Phase 7.

## 7. Risks

- Threading: the agent is not thread-safe (`AgentHost.lock`); telemetry reads must never block e-stop. Mitigation above (lock timeout, stale flag, estop without lock). Tested.
- WSL2 networking for the phone is the most likely time sink (step 6).
- A stdlib server has no HTTP/2, no compression and little hardening; acceptable for a LAN robot, so keep the "no internet exposure without a tunnel + HTTPS" rule.
