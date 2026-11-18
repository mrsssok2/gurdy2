#!/bin/bash
# Talk to Gurdy in the MOCK world (CPU only: Whisper turbo int8 + Piper; no GPU, no Gazebo). Needs the mic through WSLg.
#   scripts/voice_chat.sh              # push-to-talk: Enter, speak, Enter (a typed line also works)
#   VAD=true scripts/voice_chat.sh     # always listening (use a headset: the mic hears the robot's own voice)
#   MODEL=gurdy-sft-400 OLLAMA_URL=http://$(hostname -I|cut -d' ' -f1):11435 scripts/voice_chat.sh   # with the LLM instead of the keyword planner
# Say: "go to room A" | "find the red box" | "open room B" (-> "Did you mean: go to room B?" -> say "yes" / "no") | "stop"
# Needs piper-tts for Python 3.12 in ~/p5_piper312 (pip install --target; compose mounts it at /pk).
set -eo pipefail
cd "$(dirname "$0")/.."
[ -d "$HOME/p5_piper312/piper" ] || { echo "piper missing: docker run --rm --entrypoint pip -v ~/p5_piper312:/pk gurdy2/voice:cpu install --no-deps --target /pk piper-tts"; exit 1; }
docker compose run --rm -e MODEL -e OLLAMA_URL voice python3 -m brain.voice.cli --mic --mock $([ "$VAD" = true ] && echo --vad)
