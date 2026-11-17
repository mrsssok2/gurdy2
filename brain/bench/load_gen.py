#!/usr/bin/env python3
"""Milestone 5.1 set A load: GPU detector at 15 Hz on dataset images + Whisper transcribing a clip every 6 s (CUDA int8_float16).
Run in the vision image (PYTHONPATH=/cv:/pk:/t:/w/gurdy_core), for DURATION seconds. The LLM load runs on the host (llm_bench-like loop)."""
import glob, sys, time, threading
import numpy as np
from PIL import Image
from gurdy_core.vision.detector import OnnxDetector
from faster_whisper import WhisperModel
DUR = float(sys.argv[1]); stop = time.time() + DUR
files = sorted(glob.glob("/w/datasets/synth_v1/images/test/*.png"))[:40]
imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
det = OnnxDetector("/w/models/detector/det_v2_s.onnx", providers="auto"); print("detector provider", det.provider, flush=True)
wh = None if (len(sys.argv) > 2 and sys.argv[2] == "nowhisper") else WhisperModel("/m/whisper-turbo-ct2", device="cuda", compute_type="int8_float16")
import soundfile as sf
clip = np.random.RandomState(0).randn(16000 * 4).astype(np.float32) * 0.01
def whisper_loop():
    n = 0
    while time.time() < stop:
        t0 = time.time(); list(wh.transcribe(clip, language="en", beam_size=1)[0]); n += 1; time.sleep(max(0, 6 - (time.time() - t0)))
    print("whisper runs", n, flush=True)
if len(sys.argv) < 3 or sys.argv[2] != "nowhisper": threading.Thread(target=whisper_loop, daemon=True).start()
n = 0; lat = []
while time.time() < stop:
    t0 = time.time(); det(imgs[n % len(imgs)]); lat.append(time.time() - t0); n += 1; time.sleep(max(0, 1 / 15 - (time.time() - t0)))
print("detector frames", n, "median latency ms", round(1000 * float(np.median(lat)), 1), flush=True)
