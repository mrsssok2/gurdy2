# Phase 5.8 prep - voice pipeline (code + CPU tests only, 2026-11-17)

Package `brain/voice` (no ROS; numpy only, engines lazy): `audio` (WAV, 20 ms frames, `ArecordSource` / `AplaySink` for WSLg),
`vad` (`EnergyVad` adaptive floor + `Segmenter`: always-listening or push-to-talk, partial windows), `stt`
(`FasterWhisperSTT` CPU/GPU, `ScriptedSTT`), `tts` (`PiperTTS`, `Speaker` = sentence queue played one chunk per mic frame),
`pipeline` (`VoicePipeline`), `cli` (WAV through the mock agent).

Reaction order: speech start -> TTS stops (barge-in) | while moving, every 0.6 s the last 1.5 s of the running utterance is
transcribed and checked against the stop regex -> `on_estop` | end of utterance -> stop check BEFORE the agent (no LLM), then the
agent still gets the text (its router answers from a template). `on_estop` = `RosBackend.estop(True)` in the real run.

Tests: `docker compose run --rm brain python3 -m pytest -q brain/tests` (248 pass, 2 skip). WAVs in `brain/tests/data/voice`
(Piper, `scripts/make_voice_test_wavs.py`). Real engines (CPU Whisper turbo + Piper) on those WAVs, measured while SFT ran:
"Stop." -> "top.", "Halt!" -> "Holt." (added to the fast-path confusions), others exact; Whisper CPU 16-25 s per clip under
load (5.7 s earlier on an idle machine), Piper 0.26 s for 0.8 s of speech. Optional real-engine test: `VOICE_STT_MODEL`,
`VOICE_TTS_MODEL` (needs faster-whisper / piper, i.e. the vision image, not the brain image).

NOT done / limits: no live mic or speaker run (WSLg audio untested); energy VAD (Silero not used) cannot tell speech from
other loud sounds; no echo cancellation (only a +8 dB VAD threshold while the robot talks; use a headset); the stop-word
latency numbers in the tests use a scripted STT, so real e-stop latency (STT time on GPU ~0.3 s) is still to be measured in 5.8.
`gait_node` now obeys latched `/estop` like `policy_node` (stand ramp, ignore /cmd_vel; test `gurdy_gait/test`).
