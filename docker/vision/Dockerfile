# Gurdy 2.0 — vision image: YOLO training + ONNX export/eval on the RTX 4050 (CUDA 12.1, PyTorch 2.5.1).
# No ROS and no Gazebo here (dataset rendering runs in the `sim` image, the ROS node in the sim image too).
# Ultralytics is AGPL-3.0: it lives only in this training image (see docs/vision_objects.md).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 procps git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ultralytics==8.3.40 onnx==1.17.0 onnxslim==0.1.42 onnxruntime-gpu==1.20.2 \
      opencv-python-headless==4.10.0.84 pyyaml scipy pillow pytest "numpy<2"
RUN useradd -m -u 1000 -s /bin/bash ubuntu || true
USER ubuntu
WORKDIR /home/ubuntu/gurdy2
ENV PYTHONPATH=/home/ubuntu/gurdy2:/home/ubuntu/gurdy2/gurdy_core YOLO_CONFIG_DIR=/home/ubuntu/.config/Ultralytics
CMD ["bash"]
