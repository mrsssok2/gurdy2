# Hosting the public demo (`python3 -m web.api --demo`)

Status (2026-11-18): the demo mode is built and unit-tested; **nothing is deployed and no host was tried**. Prices and free-tier limits below are from my memory of early 2026 — **check each provider's current page before choosing**, they change often.

## What gets hosted
Only the **simulated** robot: the mock world + keyword planner (no LLM, no GPU, no ROS, no real robot). `--demo` refuses to start with anything else. No token, so anyone with the link can use it; protections are per-IP rate limits (5 chat messages burst then 1 per 4 s, 10 e-stops burst then 1/s, 60 API requests burst then 20/s), 200-character messages, 4 KB request bodies, 3 WebSocket connections per IP and 20 in total, an idle timeout, and a full world reset every 10 minutes (`--demo-reset-min`).
Needs: Python + the `brain` package + `langgraph` + `numpy` (image ≈ 340 MB, the `docker/brain` image). **RAM and idle CPU are not measured yet** (guess: 150–250 MiB); step 4 of the Phase 7 plan measures them.
Start command: `python3 -m web.api --demo --bind 0.0.0.0 --trust-proxy` (the port comes from `$PORT`, which most hosts set; `--trust-proxy` only when exactly one reverse proxy sits in front).
Still to do before deploying (needs Docker, so after the eval): `docker/web/Dockerfile` (small, non-root), health check on `/api/health`, a measured RAM number.

Limits to remember: the rate limiter and the world live **in one process** — run exactly one instance. Behind several proxy hops the "last X-Forwarded-For" is the proxy, so everybody shares one bucket (it fails safe: too strict, not too open).

## Options

| Host | Cost (to verify) | Card needed | WebSockets | Idle behaviour | Effort | Notes |
|---|---|---|---|---|---|---|
| **Hugging Face Spaces (Docker)** | free CPU tier (2 vCPU / 16 GB) | no | yes | sleeps after a period of inactivity, wakes on visit | low: push a Dockerfile; app must listen on port 7860 | good fit for an ML portfolio; public by default; set `PORT`/`--port 7860` |
| Render (free web service) | free | no | yes | spins down after ~15 min idle, ~1 min cold start; 512 MB | low: connect the GitHub repo + Dockerfile | cold start is fine for a demo; world resets anyway |
| Google Cloud Run | pay per use, monthly free allowance | yes | yes (request timeout up to 60 min) | scales to zero; keep max instances = 1 | medium: container registry + `gcloud` | background threads are throttled when no request is open; an open WebSocket counts as one, so it works |
| Fly.io | small monthly cost (no free tier for new accounts, as far as I know) | yes | yes | always on, or auto-stop | medium: `flyctl` + `fly.toml` | good global edge; single machine only |
| Railway | trial credit, then about $5/month | yes | yes | always on | low | simple, not free long-term |
| Cheap VPS (Hetzner / similar) | about €4–6/month | yes | yes | always on | medium: you run Docker + a proxy (Caddy gives automatic https) | most predictable; you own the patching |
| Oracle Cloud "Always Free" VM | free | yes | yes | always on, but idle instances can be reclaimed | high: sign-up friction, own proxy + https | generous, fiddly |
| Home PC + Cloudflare Tunnel | free | no (account) | yes | only while the PC is on | low | https for free, but it exposes a process on your home machine: run **only** the demo container, never the real robot stack on the same tunnel |
| GitHub Pages / Netlify static | free | no | no | — | — | **not possible**: the demo needs the Python server |

## Recommendation
1. **Hugging Face Spaces (Docker)** — free, no card, WebSockets, fits a portfolio.
2. Fallback: **Render free** (same idea, cold starts) or **Cloud Run** if you want a custom domain and do not mind a card.
3. If it must always be instant: a small VPS with Caddy.

## Private access to the *real* dashboard (not the demo)
Use Tailscale (private network, https via `tailscale serve`) or a Cloudflare Tunnel with Cloudflare Access in front, **with the token enabled**. Never use `--demo` for that: demo mode has no authentication by design. See `docs/phase7_phone.md` for the LAN setup.
