# Read-aloud voice evaluation

Measures how well the voice pipeline understands YOU, with the same mic path as the live loop, and lets you compare Whisper settings on identical recordings.

## 1. Record (needs the mic, about 10 minutes)
```bash
scripts/voice_eval.sh record
```
It shows one sentence at a time from `brain/eval/voice_prompts_v1.txt` (43 sentences: room commands A/B/D, phonetic rooms Alpha/Bravo/Charlie/Delta, objects, home, status questions, stop, yes/no, long and casual sentences; none copied from `testset_ood_v1`).
* **Enter** = start recording, say the sentence, **Enter** = send (the recording continues 0.5 s after Enter, so the last word is kept). One recording at a time.
* `r` + Enter = redo the previous sentence, `s` + Enter = skip, `q` + Enter = finish early.
* The segmenter is the live push-to-talk one: 0.5 s pre-roll, 0.5 s post-roll, the same gentle trim.
* Output: `brain/runs/voice_eval/<timestamp>/` with `001.wav ...` and `meta.jsonl` (id, category, reference text, level). `brain/runs/` is not committed. Use a quiet room and your usual mic distance; read naturally, do not spell out letters.

Edit the prompt file to add sentences (`category | sentence`); keep the categories if you want them in the per-category table.

## 2. Score (offline, no mic)
```bash
scripts/voice_eval.sh score brain/runs/voice_eval/<timestamp> --bias none prompt full
```
* Runs Whisper with the push-to-talk settings (VAD filter off, no question prompt) once per bias mode: `none` = no prompt, `prompt` = initial prompt with the robot's words (the default), `full` = prompt + hotwords.
* Device `--device auto` (default): GPU when CTranslate2 sees one with at least 2 GiB VRAM free, otherwise CPU. The `voice` image is CPU-only, so it is CPU there (about 6-10 s per clip: 43 clips x 3 modes is 15-20 minutes; use one mode for a quick run). Do not score while an SFT / training job needs the RAM.
* Prints and saves (`score_<time>.json` in the folder):
  * **WER %** = word errors / reference words (lower case, punctuation ignored, "2" = "two"; "Alpha" and "A" are different words here, so WER is strict).
  * **Exact-command accuracy %**: the transcript parses to the same command as the reference. Room names are normalised like the agent does (Alpha = room A, "D as in dog" = room D), so "Go to room Bravo" heard as "Go to room B" counts as a correct command with one word error. The parser is the keyword baseline (`brain/eval/keyword_parser.py`); sentences it cannot parse (many casual / long ones) are compared word for word instead (`cmd_basis: text` in the JSON). Hallucination rejects (repeated words, silence) count as "heard nothing".
  * Per category and per bias mode, plus every command mismatch (reference -> what Whisper heard, confidence).
* Re-score the same folder after any change (bias, trimming, a new Whisper model) to compare on identical audio.

## Limits
* One speaker, one mic, one room: a regression check and a tuning aid, not a general accuracy number.
* WER on short sentences is noisy: one word is 25-100 % of a sentence. Look at command accuracy and the mismatch list.
