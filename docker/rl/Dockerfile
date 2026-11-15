# Gurdy 2.0 — RL training image (headless, no ROS): MuJoCo + PyTorch (CPU) + Stable-Baselines3.
# Ubuntu 24.04 / Python 3.12 like the sim image; MuJoCo pinned to the SAME version (3.3.0) so the
# physics is identical to the parity-tested model. Runs as `ubuntu` (UID 1000).
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      libgl1 libegl1 libgl1-mesa-dri libglfw3 libosmesa6 ffmpeg fonts-dejavu-core \
      procps less nano \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv $VIRTUAL_ENV \
 && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      numpy==1.26.4 pyyaml==6.0.2 scipy==1.14.1 \
      mujoco==3.3.0 gymnasium==1.0.0 stable-baselines3==2.4.1 \
      tensorboard==2.18.0 onnx==1.17.0 onnxruntime==1.20.1 \
      psutil==6.1.0 imageio==2.36.1 imageio-ffmpeg==0.5.1 pillow==11.0.0 pytest==8.3.4

USER ubuntu
WORKDIR /home/ubuntu/gurdy2
ENV PYTHONPATH=/home/ubuntu/gurdy2/gurdy_core:/home/ubuntu/gurdy2/sim_mujoco:/home/ubuntu/gurdy2/rl \
    MUJOCO_GL=egl
CMD ["bash"]
