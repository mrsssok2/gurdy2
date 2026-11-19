# Public demo: a static page that runs the simulation in the browser

**Why this changed:** Hugging Face now needs a PRO subscription for Docker Spaces on free CPU (static Spaces stay free). So the public demo no longer needs a server at all: the dashboard plus the **real Python simulation** (mock `flat_lab` world, agent with safety / grounding / executor, keyword planner with the smart phrasings) run inside the visitor's browser. It is published for free on **GitHub Pages** (`https://mrsssok2.github.io/gurdy2/`) and can also be uploaded as a **static Hugging Face Space**.

What a visitor gets: the same dashboard (map, planned path, chat with suggestion chips, status, alerts, demo banner, E-STOP), no mic, no token, no server, nothing sent anywhere. Every visitor automatically has a private robot (it lives in their own tab); a reload starts a new one.

## The two options I compared

| | **A. Pyodide (chosen)**: the repo's Python runs in the browser | **B. JavaScript port** |
|---|---|---|
| Fidelity | The same files: agent, safety checker, grounding, executor, mock world, keyword parser, phrasing, telemetry. | A second implementation of ~4,000 lines of Python (agent ~2,200, mock world ~950, tools ~400, parsers) that must be kept in sync by hand; parity could only be spot-checked on a fixed command list. |
| First load | **5.6 MiB** from the jsDelivr CDN (Pyodide runtime, brotli) + 0.23 MiB our site. Measured cold: **8-11 s** to a working page (desktop-class, Docker Chromium), **2.7-3.4 s** with the runtime cached; on a throttled phone profile (4x slower CPU, 1.6 Mbit/s, 150 ms latency) cold **35 s**. | Instant (< 0.2 MB). |
| Risk | `langgraph` cannot be installed in Pyodide (checked in headless Chromium with micropip: it fails because its dependencies orjson / ormsgpack / xxhash / uuid_utils have no Pyodide wheels; pydantic 2.10 and pyyaml do load). Solved with a 40-line stand-in for the two things `brain/agent/graph.py` uses (`StateGraph`, `END`). | Drift between the JS and Python behaviour over time. |
| Proof of equivalence | `web/tests/test_static_demo.py` runs 30 commands on the repo with the **real langgraph** and on the unzipped build with **only the stand-in**: identical replies, announcements and robot poses. | A parity test would have to be written for every behaviour. |

Pyodide wins on fidelity and upkeep; its only cost is the one-time download. The page shows a loading screen ("Starting the simulated robot… everything runs in your browser") during it.

## How it works
`scripts/build_static_demo.py` copies `web/app` (the normal dashboard, unchanged apart from one line: no service worker in the static build), adds `static_backend.js` (answers the dashboard's `fetch("/api/…")` and `WebSocket("/ws")` calls locally), `demo_worker.js` (a Web Worker that loads Pyodide from the CDN, installs `pyyaml`, unpacks `gurdy_demo.zip` and runs `web/static_demo/session.py`) and `gurdy_demo.zip` (88 KiB of the repo's Python: no server, ROS, LLM client or audio code). The simulation advances 3x faster than wall time, like the server demo.
Pyodide version pinned: 0.27.7 (Python 3.12.7). It is loaded from `cdn.jsdelivr.net`; to host it yourself pass `--pyodide-base`.

Measured 2026-11-19 in headless Chromium: first visit loads **5.6 MiB (CDN) + 0.23 MiB**; chat "enter room B", "yes" walks the robot to room B, E-STOP works, no mic button, demo banner shown, a second browser context starts with an empty chat and a robot at the origin, a reload resets the world, no JavaScript errors, works from a sub-path (`/gurdy2/`) like GitHub Pages. Not tested on a real phone or on Safari/Firefox.

## Step by step: GitHub Pages (your repo `mrsssok2/gurdy2`)
1. **The repo must be public** (Pages is free for public repositories; for a private repo it needs a paid plan).
2. **Push** the commits (`git push`). The workflow `.github/workflows/pages.yml` runs on every push to `main` that touches the demo files.
3. In the repo on github.com: **Settings → Pages → Build and deployment → Source: "GitHub Actions"** (not "Deploy from a branch"). Do this once; it needs no token.
4. Open the **Actions** tab → workflow **"Publish static demo"**. If it did not start (for example because the push happened before step 3), click **Run workflow** (branch `main`). It has three jobs: `test` (runs the demo tests), `build`, `deploy`; all must be green (about 2-3 minutes).
5. The deploy job prints the page URL; it is **https://mrsssok2.github.io/gurdy2/** (give it a minute after the job finishes).
6. **Check it:** open the URL (laptop and phone): loading screen for a few seconds, then the dashboard with the banner "Demo: simulated robot"; type "go to room B", then "yes"; the robot walks. In the browser's developer tools (F12 → Network / Console) there must be no red 404 / CORS errors. `curl -I https://mrsssok2.github.io/gurdy2/` answers 200.
7. If a job fails: open it and read the red step. Typical causes: Pages not set to "GitHub Actions" (step 3); the repo is private; a test fails (the log shows which).

**Updating later:** change the code, `git push`; the workflow rebuilds and republishes by itself. Nothing else to do.

## Alternative: a static Hugging Face Space (free)
1. Create the account at https://huggingface.co/join and confirm the e-mail.
2. Token: https://huggingface.co/settings/tokens → **Create new token** → type **Write** → copy it once. Keep it out of the repo and the chat; you type it at a silent prompt below.
3. Tool (once): `python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install huggingface_hub`
4. Look at what will be uploaded (optional): `python3 scripts/demo_space.py stage` → `build/demo_space` (22 files, ~290 KiB, git-ignored).
5. Deploy:
   ```bash
   read -s HF_TOKEN; export HF_TOKEN        # paste the token, press Enter (nothing is shown or saved)
   ~/.venvs/hf/bin/python scripts/demo_space.py deploy
   unset HF_TOKEN
   ```
   It creates the public **static** Space `USER/gurdy-demo` if it does not exist, uploads, waits for RUNNING and fetches the page. It prints `LIVE: https://USER-gurdy-demo.static.hf.space/`.
6. Check it like step 6 above. To update: run the deploy command again (clean snapshot; old versions stay in the Space's Files → History). To stop it: Space → Settings → Pause / delete.
The Space page embeds the demo in an iframe; Web Workers and the CDN load work there. If something looks off in the iframe, open the direct `*.static.hf.space` URL.

## Limits and honest notes
- First visit: about 6 MB and several seconds (35 s on a slow phone profile). Later visits are quick because the browser caches the runtime.
- Needs a recent browser with WebAssembly and Web Workers. If `cdn.jsdelivr.net` is blocked (some company networks), the page shows an error message instead of the simulator.
- Same planner as before (keyword parser, no LLM), no voice. The world is the Gazebo `flat_lab` layout, not Gazebo itself.
- The per-visitor server mode (`python -m web.api --demo`, with sessions, caps and rate limits; tests in `web/tests/test_demo_sessions.py`) still exists for self-hosting on a machine you control; it is no longer what gets deployed.
- Safety: the page contains no robot connection, token, secret or backend address. Everything it can do is run a simulation inside the visitor's own browser.
