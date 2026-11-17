#!/usr/bin/env python3
"""Milestone 5.1: Whisper large-v3-turbo (CTranslate2) latency and Piper vs Kokoro time-to-first-audio.
Run in the vision image with PYTHONPATH=/cv:/pk:/t (see docs/results/phase5_budget.md for the exact command).
Prints one JSON line per measurement. Audio for STT is synthesised by Piper (clean speech; real mic audio is slower/noisier)."""
import json, subprocess, sys, time
import numpy as np

TEXTS = ["Go to the red box and stay there.",
         "Explore the room, then come back to the start and tell me what you found."]
M = "/m"

def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.split()[0])

def out(**k): print(json.dumps(k), flush=True)

def pstats(xs): return dict(median=round(float(np.median(xs)), 3), min=round(min(xs), 3), max=round(max(xs), 3), n=len(xs))

# ---- Piper
from piper import PiperVoice
v = PiperVoice.load(f"{M}/tts/piper-lessac.onnx", use_cuda=False)
sr = v.config.sample_rate
list(v.synthesize("warm up"))
audio = {}
for i, t in enumerate(TEXTS):
    first, total = [], []
    for _ in range(5):
        t0 = time.perf_counter(); it = iter(v.synthesize(t)); c = next(it); f = time.perf_counter() - t0
        chunks = [c] + list(it); tt = time.perf_counter() - t0
        first.append(f); total.append(tt)
    a = np.concatenate([c.audio_float_array for c in chunks]); dur = len(a) / sr
    audio[i] = (a, sr)
    out(what="piper_cpu", text_len=len(t), audio_s=round(dur, 2), first_audio_s=pstats(first), total_s=pstats(total), rtf=round(float(np.median(total)) / dur, 3))

# ---- Kokoro
import os
if os.path.exists(f"{M}/tts/voices-v1.0.bin") and os.path.getsize(f"{M}/tts/kokoro-v1.0.onnx") > 3e8:
    from kokoro_onnx import Kokoro
    for prov in ("CPU", "CUDA"):
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider" if prov == "CUDA" else "CPUExecutionProvider"
        try:
            k = Kokoro(f"{M}/tts/kokoro-v1.0.onnx", f"{M}/tts/voices-v1.0.bin")
            _ = k.create("warm up", voice="af_heart", lang="en-us")
            for t in TEXTS:
                first, total = [], []
                for _ in range(5):
                    async_first = None
                    import asyncio
                    async def run():
                        t0 = time.perf_counter(); f = None
                        async for s, _sr in k.create_stream(t, voice="af_heart", lang="en-us"):
                            if f is None: f = time.perf_counter() - t0
                        return f, time.perf_counter() - t0
                    f, tt = asyncio.run(run()); first.append(f); total.append(tt)
                out(what=f"kokoro_{prov.lower()}", text_len=len(t), first_audio_s=pstats(first), total_s=pstats(total))
        except Exception as e:
            out(what=f"kokoro_{prov.lower()}", error=repr(e)[:200])
else:
    out(what="kokoro", error="model file not complete")

# ---- Whisper
from faster_whisper import WhisperModel
base = vram()
for dev, ct in (("cuda", "int8_float16"), ("cuda", "float16"), ("cpu", "int8")):
    try:
        m = WhisperModel(f"{M}/whisper-turbo-ct2", device=dev, compute_type=ct, cpu_threads=6)
    except Exception as e:
        out(what=f"whisper_{dev}_{ct}", error=repr(e)[:200]); continue
    for i, (a, s) in audio.items():
        x = a
        if s != 16000: x = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / s)), np.arange(len(a)), a).astype(np.float32)
        segs, _ = m.transcribe(x, language="en", beam_size=1, vad_filter=False); " ".join(sg.text for sg in segs)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); segs, _ = m.transcribe(x, language="en", beam_size=1, vad_filter=False); txt = " ".join(sg.text for sg in segs).strip(); ts.append(time.perf_counter() - t0)
        out(what=f"whisper_{dev}_{ct}", audio_s=round(len(x) / 16000, 2), latency_s=pstats(ts), text=txt, ref=TEXTS[i], vram_mib_above_idle=vram() - base if dev == "cuda" else None)
    del m
