# Public demo of the Gurdy mission-control app (simulated robot only): Python + the agent, mock world and web server. No ROS, Gazebo, CUDA, Ollama or Whisper.
# Built from the staged folder made by `python3 scripts/demo_space.py stage` (that folder is also what gets pushed to the Hugging Face Space).
# Runs as UID 1000 (Hugging Face Spaces requirement); listens on $PORT (7860).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=7860 \
    PYTHONPATH=/home/user/app:/home/user/app/gurdy_core
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 8 --timeout 60 -r requirements.txt

COPY --chown=user . .
USER user
EXPOSE 7860
# --demo: MockBackend only, no token, per-IP rate limits, 200-char messages, no voice; every browser session gets its own isolated world / agent / chat
# (max 30 at once, freed after 10 idle minutes, "demo is busy" when full).
# --trust-proxy: the Space sits behind ONE reverse proxy; the client address comes from X-Forwarded-For (otherwise every visitor shares one rate-limit bucket).
CMD ["python", "-m", "web.api", "--demo", "--bind", "0.0.0.0", "--max-sessions", "30", "--session-idle-min", "10", "--trust-proxy"]
