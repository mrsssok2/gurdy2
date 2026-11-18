"""Record N seconds from the mic with the stream opened and warmed up BEFORE ">>> SPEAK NOW" (WSLg RDPSource wakes in ~1 s).
  python3 scripts/mic_record.py out.wav [seconds=5] [warmup_s=1.5] [--cold]     (--cold = old behaviour, no warm-up, for comparison)"""
import sys, time
from brain.voice.audio import ArecordSource, FRAME_N, write_wav
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
out, secs = args[0], float(args[1]) if len(args) > 1 else 5.0
src = ArecordSource(warmup_s=0.0 if "--cold" in sys.argv else float(args[2]) if len(args) > 2 else 1.5)
t0 = time.monotonic()
src.start()
print(f"stream open, warm-up {time.monotonic() - t0:.1f} s discarded (rms {src.warmup_rms:.4f})", flush=True)
print(">>> SPEAK NOW  (e.g. go to room B and then find the red box)", flush=True)
fr = []
for f in src:
    fr.append(f)
    if len(fr) * FRAME_N >= secs * 16000:
        break
src.close()
write_wav(out, np.concatenate(fr))
