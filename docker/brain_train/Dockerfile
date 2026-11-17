# Gurdy 2.0 — Phase 5.5 SFT image: QLoRA training of Qwen3 on the RTX 4050 (6 GB).
# Built ON TOP of gurdy2/vision:cuda (PyTorch 2.5.1 + CUDA 12.1 already inside, so nothing big is downloaded again).
# No TRL / datasets: brain/train/sft.py has its own small loop (assistant-only loss mask, token-budget batching).
FROM gurdy2/vision:cuda
USER root
RUN pip install --no-cache-dir transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 bitsandbytes==0.45.5 \
      sentencepiece safetensors "numpy<2" \
    && python -c "import torch;print(torch.__version__)"
USER ubuntu
WORKDIR /home/ubuntu/gurdy2
CMD ["bash"]
