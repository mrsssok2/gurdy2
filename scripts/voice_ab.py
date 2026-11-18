#!/usr/bin/env python3
"""Re-transcribe the saved utterances (brain/runs/voice_debug/*.wav) with different Whisper settings, to see what hurts short phrases.
  docker compose run --rm voice python3 scripts/voice_ab.py [DIR]
Configs: no_bias | prompt_only | full (initial_prompt + hotwords, what the live loop uses) | lead0.3 (full, 0.3 s of silence prepended)."""
import json, sys
from pathlib import Path

import numpy as np

from brain.voice.audio import SAMPLE_RATE, read_wav
from brain.voice.stt import FasterWhisperSTT

d = Path(sys.argv[1] if len(sys.argv) > 1 else "brain/runs/voice_debug")
stt = FasterWhisperSTT("/models/whisper-turbo-ct2", "cpu", bias="full")   # "full" so that stt.hotwords is really set (the default is prompt-only)
full = (stt.prompt, stt.hotwords)
cfgs = {"no_bias": (None, None, 0), "prompt_only": (full[0], None, 0), "full": (*full, 0), "lead0.3": (*full, 0.3)}
rows = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines()] if (d / "index.jsonl").exists() else [{"wav": p.name} for p in sorted(d.glob("*.wav"))]
for r in rows:
    audio = read_wav(d / r["wav"])
    print(f"\n{r['wav']}  live: {r.get('text')!r} conf {r.get('confidence')}  ({len(audio) / SAMPLE_RATE:.1f} s, rms {r.get('rms', 0):.3f})")
    for name, (pr, hw, lead) in cfgs.items():
        stt.prompt, stt.hotwords = pr, hw
        a = np.concatenate([np.zeros(int(lead * SAMPLE_RATE), np.float32), audio])
        t = stt.transcribe(a)
        print(f"  {name:12s} {t.text!r:40s} conf {t.confidence if t.confidence is None else round(t.confidence, 2)}")
