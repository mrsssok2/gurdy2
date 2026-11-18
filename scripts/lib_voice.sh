# Source me. voice_service -> `voice_gpu` (Whisper on the GPU, ~0.3 s per utterance) when the CUDA voice image, the mounted faster-whisper packages and a GPU exist,
# else `voice` (CPU, 6-10 s per utterance). Override: VOICE_GPU=false forces the CPU service.
voice_service() {
  if [ "${VOICE_GPU:-true}" != false ] && docker image inspect gurdy2/voice:cuda >/dev/null 2>&1 \
     && [ -d "$HOME/p5_pkgs/ctranslate2" ] && [ -d "$HOME/p5_tts/piper" ] && nvidia-smi -L >/dev/null 2>&1; then
    echo voice_gpu
  else
    echo voice
  fi
}
