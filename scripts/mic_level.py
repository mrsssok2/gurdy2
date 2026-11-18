"""Print level of a WAV and warn on silence / clipping.  python3 scripts/mic_level.py clip.wav"""
import sys, wave
import numpy as np
w = wave.open(sys.argv[1])
a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
peak, rms = abs(a).max(), np.sqrt((a ** 2).mean())
print(f"recorded {len(a) / w.getframerate():.1f} s, peak {peak:.3f}, rms {rms:.4f}")
if peak >= 0.99:
    print("WARNING: clipping (peak >= 0.99). Lower the microphone volume in Windows (Sound settings -> Input) and record again.")
elif peak < 0.01:
    print("WARNING: mic silent or muted (peak < 0.01).")
