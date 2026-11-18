# Voice test 4: last word cut, one-word answers (2026-11-18, real clips in brain/runs/voice_debug, not committed)

`scripts/voice_ab.py` on the clips that came out as "Go to room" (121417_001, 121601_005, 121650_007) and the one-word answers.

* **Last word is not in the audio.** In all three clips the level in the last 0.3 s is the loudest part of the clip (-26..-32 dB, floor -42 dB): the recording ends while the word is still being said.
  Appending 0.5 s of silence (`tail0.5`), switching the VAD filter off (`no_vad`) or normalising the level (`norm`) does NOT bring the word back (still "Go to room", conf 0.41-0.61), so Whisper / VAD / trim are
  not dropping it; the audio is cut at Enter. Fix: push-to-talk keeps recording 0.5 s after Enter (`VoiceConfig.ptt_postroll_s`). It cannot be proven on these clips (the missing audio was never recorded): re-record "Go to room A / Delta".
* Mic level is low (speech peaks around -26 dB, rms 0.02): raising the Windows mic input level will help Whisper more than any setting here. Peak normalisation was mixed (better on 2 clips, worse on 2), so it is not applied.
* Hotwords ("full") are harmful on short clips (121707_008: "Don't forget to subscribe to the channel." conf 0.25): keep `prompt` as the default.
* `question_prompt` (expected answers as initial_prompt) gave "And." 0.60 / "Bye." 0.57-0.62 on the one-word clips: slightly higher confidence, but these 0.6-0.9 s clips contain very little speech energy and "Alpha" is not
  recognised in any configuration. The room answer is therefore also matched fuzzily (`rooms.fuzzy_answer`) and ALWAYS read back ("Room A, Alpha. Shall I go?") before anything moves.
