# Gurdy 2.0 — voice image (Phase 5.8 live test): brain image + PulseAudio client tools (WSLg mic/speaker) + faster-whisper (CPU).
# No ROS, no GPU libraries (the CPU build of ctranslate2 is what pip ships; CUDA use stays in the vision image for benchmarks).
FROM gurdy2/brain:cpu
USER root
RUN apt-get update && apt-get install -y --no-install-recommends pulseaudio-utils && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir faster-whisper==1.2.1
USER ubuntu
ENV PULSE_SERVER=unix:/mnt/wslg/PulseServer
