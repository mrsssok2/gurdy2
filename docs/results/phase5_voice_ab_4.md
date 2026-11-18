# Voice test 4: last word cut, one-word answers (2026-11-18, real clips in brain/runs/voice_debug, not committed)

`scripts/voice_ab.py` on the clips that came out as "Go to room" (121417_001, 121601_005, 121650_007) and the one-word answers.

* **Last word is not in the audio.** In all three clips the level in the last 0.3 s is the loudest part of the clip (-26..-32 dB, floor -42 dB): the recording ends while the word is still being said.
  Appending 0.5 s of silence (`tail0.5`), switching the VAD filter off (`no_vad`) or normalising the level (`norm`) does NOT bring the word back (still "Go to room", conf 0.41-0.61), so Whisper / VAD / trim are
  not dropping it; the audio is cut at Enter. Fix: push-to-talk keeps recording 0.5 s after Enter (`VoiceConfig.ptt_postroll_s`). It cannot be proven on these clips (the missing audio was never recorded): re-record "Go to room A / Delta".
* Mic level is low (speech peaks around -26 dB, rms 0.02): raising the Windows mic input level will help Whisper more than any setting here. Peak normalisation was mixed (better on 2 clips, worse on 2), so it is not applied.
* Hotwords ("full") are harmful on short clips (121707_008: "Don't forget to subscribe to the channel." conf 0.25): keep `prompt` as the default.
* `question_prompt` (expected answers as initial_prompt) gave "And." 0.60 / "Bye." 0.57-0.62 on the one-word clips: slightly higher confidence, but these 0.6-0.9 s clips contain very little speech energy and "Alpha" is not
  recognised in any configuration. The room answer is therefore also matched fuzzily (`rooms.fuzzy_answer`) and ALWAYS read back ("Room A, Alpha. Shall I go?") before anything moves.

## Voice test 5: "Where are you?" loses its soft start (clips 125332_008 .. 125423_011)
`voice_ab.py` (prompt_only = what the loop used, VAD filter ON, vs `no_vad`, vs `norm`):
| clip | live text | prompt_only (VAD on) | no_vad | norm |
|---|---|---|---|---|
| 125332_008 | That. Are. You. 0.74 | That. Are. You. 0.75 | That. Are. You. 0.75 | That. Are. You. 0.77 |
| 125349_009 | What are you? 0.75 | What are you? 0.72 | **Where are you? 0.89** | What are you? 0.75 |
| 125405_010 | R U. 0.48 | R U. 0.48 | **Where are you? 0.85** | Where are you? 0.82 |
| 125423_011 (good) | Where are you? 0.82 | 0.84 | 0.80 | 0.78 |
Whisper's VAD filter (added against hallucinations) clips the "Wh" of two of the three bad clips; the "Wh" is in the audio. Push-to-talk now runs with the VAD filter OFF (the key marks the speech; always-listening mode keeps it).
Clip 008 is unchanged by every setting: its audio starts in the middle of the word (the first loud frame is frame 0), so the start was lost before Whisper (pre-roll / trim). Push-to-talk pre-roll is now 0.5 s (was 0.3), the push-to-talk trim keeps 0.5 s before the first loud frame (was 0.24 s) and uses a threshold 6 dB lower. Not provable on old clips: re-record.
