#!/bin/bash
# Microphone check through WSLg PulseAudio, in the `voice` container (CPU only, does not touch the GPU).
#   scripts/mic_test.sh            # list devices, record 5 s, play it back, transcribe with Whisper (CPU)
#   scripts/mic_test.sh list       # only list devices
# Say (when "RECORDING" appears): "go to room B and then find the red box"
set -eo pipefail
cd "$(dirname "$0")/.."
docker compose run --rm voice bash -c '
  echo "== PulseAudio server"; pactl info | grep -E "Server Name|Default Source|Default Sink" || { echo "no PulseAudio at $PULSE_SERVER (is WSLg running? try: wsl --shutdown, then reopen)"; exit 1; }
  echo "== sources (mic = non-monitor)"; pactl list short sources
  echo "== sinks"; pactl list short sinks
  [ "'"${1:-}"'" = list ] && exit 0
  echo; echo "RECORDING 5 s in 2 s ... say: go to room B and then find the red box"; sleep 2; echo ">>> SPEAK NOW"
  parecord --rate=16000 --channels=1 --format=s16le --file-format=wav /tmp/mic_test.wav & P=$!
  sleep 5; kill -INT $P; wait $P 2>/dev/null || true
  python3 -I - <<PY
import wave, numpy as np
w = wave.open("/tmp/mic_test.wav"); a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
print(f"recorded {len(a)/w.getframerate():.1f} s, peak {abs(a).max():.3f}, rms {np.sqrt((a**2).mean()):.4f}  (peak < 0.01 = mic silent / muted)")
PY
  echo ">>> PLAYBACK"; paplay /tmp/mic_test.wav
  echo "== Whisper turbo, CPU int8"
  python3 - <<PY
import time; from faster_whisper import WhisperModel
t=time.time(); m = WhisperModel("/models/whisper-turbo-ct2", device="cpu", compute_type="int8", cpu_threads=4); print(f"load {time.time()-t:.1f} s")
t=time.time(); segs,_ = m.transcribe("/tmp/mic_test.wav", language="en", beam_size=1, vad_filter=True); print(\"TEXT:\", \" \".join(s.text for s in segs).strip(), f\"({time.time()-t:.1f} s)\")
PY
'
