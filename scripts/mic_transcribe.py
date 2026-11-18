"""Transcribe a WAV with Whisper turbo on CPU (int8).  python3 scripts/mic_transcribe.py clip.wav [model_dir]"""
import sys, time
from faster_whisper import WhisperModel
from brain.voice.audio import read_wav     # own WAV reader: faster-whisper's file decoder (av) is broken with the installed av 19
t = time.time()
m = WhisperModel(sys.argv[2] if len(sys.argv) > 2 else "/models/whisper-turbo-ct2", device="cpu", compute_type="int8", cpu_threads=4)
print(f"load {time.time() - t:.1f} s")
t = time.time()
segs, _ = m.transcribe(read_wav(sys.argv[1]), language="en", beam_size=1, vad_filter=True)
print("TEXT:", " ".join(s.text for s in segs).strip(), f"({time.time() - t:.1f} s)")
