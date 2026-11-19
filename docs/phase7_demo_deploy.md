# Public demo on a Hugging Face Space (Docker, free CPU)

What gets deployed: `python -m web.api --demo` — the dashboard with a **simulated** robot in the Gazebo `flat_lab` layout (two rooms, doorway, red / blue / yellow boxes, green cylinder, table), the keyword planner with the smart phrasings ("enter room B", "take me to…", "find the yellow box"), **no token, per-IP rate limits (burst 5, then one message per 4 s), 200-character messages, world reset every 10 minutes, no voice**.
The server refuses to run the demo with anything but the mock backend, so this page can never reach a real robot.

Measured locally (Docker Desktop, 2026-11-19, `docker/demo/Dockerfile`): **image 366 MB**, **RAM 69 MiB** after a few commands, 5 quick commands answered, the 6th-8th in a burst get HTTP 429, `/api/voice` answers 503, no JavaScript errors in headless Chromium on laptop and phone sizes.
Not yet tested: the real Space (it needs your account).

Files: `docker/demo/Dockerfile` (python:3.12-slim, user 1000, `$PORT`=7860), `docker/demo/requirements.txt` (pyyaml, langgraph, numpy), `docker/demo/README_SPACE.md` (the Space's front matter), `scripts/demo_space.py` (stages exactly the files the demo needs — 77 files, ~0.5 MB — and deploys them).
Nothing from ROS, Gazebo, CUDA, Ollama, Whisper, your memory database or your TLS / Tailscale files is ever staged (the script refuses secrets and files over 5 MB; a test checks it).

## Step by step (in this order)

### 1. Hugging Face account
Go to https://huggingface.co/join, sign up, confirm the e-mail address. Your user name appears in the URLs below (`USER`).

### 2. Access token (write permission)
https://huggingface.co/settings/tokens → **Create new token** → type **Write** → name it `gurdy-demo-deploy` → Create → copy it now (it is shown once).
Keep it out of the repo, chat, notebooks and shell history: you will type it at a silent prompt in step 5. If it ever leaks, delete it on the same page and make a new one.

### 3. The Space (optional by hand; the script also creates it)
https://huggingface.co/new-space → Owner: you · Space name: `gurdy-demo` · License: MIT · SDK: **Docker** → **Blank** → Hardware: **CPU basic · free** → **Public** → Create Space.
(Without this step `scripts/demo_space.py deploy` creates the same public Docker Space named `USER/gurdy-demo`.)

### 4. Tool for the upload (once)
```bash
python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install huggingface_hub
```

### 5. Test the demo image locally first (recommended every time)
```bash
python3 scripts/demo_space.py stage                    # copies the demo files into build/demo_space (git-ignored)
docker build -t gurdy2/demo:test build/demo_space      # a minute or two
docker run --rm -p 127.0.0.1:7860:7860 gurdy2/demo:test
# in a browser: http://localhost:7860  ->  banner "Demo: simulated robot", chat "go to room B", then "yes"
# stop with Ctrl+C
```

### 6. Deploy
```bash
read -s HF_TOKEN; export HF_TOKEN        # paste the token, press Enter (nothing is shown, nothing is saved)
~/.venvs/hf/bin/python scripts/demo_space.py deploy
unset HF_TOKEN
```
The script stages the files, creates the Space if needed, uploads a clean snapshot, waits for the build (the first build takes a few minutes) and calls `/api/health`. It prints `LIVE: https://USER-gurdy-demo.hf.space`.
Use another name with `--space USER/other-name`.

*Alternative with git* (the token is typed at the prompt; do not put it in the URL, that would store it in `.git/config`):
```bash
git -c credential.helper= clone https://huggingface.co/spaces/USER/gurdy-demo build/space_git   # user name: USER, password: the token
rsync -a --delete --exclude .git build/demo_space/ build/space_git/ && cd build/space_git
git add -A && git commit -m "Update demo" && git -c credential.helper= push
```

### 7. Check that it is live
- Space page https://huggingface.co/spaces/USER/gurdy-demo : status **Running** (tab **Logs** shows the build / run log; `DEMO mode on http://0.0.0.0:7860/` is the line you want).
- `curl https://USER-gurdy-demo.hf.space/api/health` → `{"ok": true, "version": "0.1", "demo": true, "voice": false}`.
- Open the URL on your phone (mobile data): banner "Demo: simulated robot", no login box, no mic button; chat "go to room B" → "yes" → the robot walks.
- **Rate limits per visitor**: the Space sits behind a proxy and the demo reads the visitor's address from `X-Forwarded-For` (`--trust-proxy`). Test it: send 8 quick messages from your laptop (some answer "too many requests") and, at the same moment, one message from your phone on mobile data must still work. If the phone is also blocked, every visitor shares one bucket: tell me and I change how the address is read.
- Note: **all visitors share one simulated robot and one chat** (that is how the demo is built); it resets every 10 minutes and tells every open page to start over.

### 8. Update it later
Change the code in the repo, run step 5 (local test), then step 6 again. Each deploy replaces the Space's files with a clean snapshot (the old state stays in the Space's **Files → History**, so you can roll back in the web page) and the Space rebuilds by itself.
To stop the demo: Space → **Settings → Pause this Space** (or delete it). Free CPU Spaces also fall asleep when nobody visits for a while and wake up on the next visit (about a minute; I have not measured how long they stay awake, check Hugging Face's current rules).

### 9. Put the link in the README
In `README.md` replace `YOUR_HF_USER` in the "Try the live demo" line with your user name.

## Safety notes
- The Space is public and has **no login by design**; the protection is: mock world only (enforced in code), rate limits, short messages, 3 WebSocket connections per address, idle timeout, periodic reset, no voice, no GPU, no secrets inside (the image holds ~0.5 MB of source and the web page).
- Do not add `GURDY_WEB_TOKEN`, Ollama addresses or Tailscale settings to the Space; none of them are used in demo mode.
- Prices / limits of the free tier change: look at https://huggingface.co/docs/hub/spaces-overview before relying on it (see also `docs/phase7_hosting.md`, prices unverified there too).
