# Gurdy 2.0 — brain image (Phase 5): agent, tool layer, mock world, evaluation harness. No ROS, no torch, no GPU.
# Same Ubuntu 24.04 / Python 3.12 as the other images. Runs as `ubuntu` (UID 1000), repo bind-mounted.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv $VIRTUAL_ENV && pip install --no-cache-dir -r /tmp/requirements.txt pytest==8.3.4

USER ubuntu
WORKDIR /home/ubuntu/gurdy2
ENV PYTHONPATH=/home/ubuntu/gurdy2:/home/ubuntu/gurdy2/gurdy_core PYTHONDONTWRITEBYTECODE=1
CMD ["bash"]
