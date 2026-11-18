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
  python3 -I scripts/mic_level.py /tmp/mic_test.wav
  echo ">>> PLAYBACK"; paplay /tmp/mic_test.wav
  echo "== Whisper turbo, CPU int8"
  python3 scripts/mic_transcribe.py /tmp/mic_test.wav
'
