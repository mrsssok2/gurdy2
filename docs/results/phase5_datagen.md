# Phase 5.4 — SFT data generator (built 2026-11-17)

Code: `brain/data_gen/` (`lexicon`, `worlds`, `families`, `generate`, `noise_synth`, `merge_noise`, `validate`, `token_stats`), driver `scripts/p5_datagen.sh`, tests `brain/tests/test_data_gen.py`.
Output (not in git, `brain/data/` is ignored): `brain/data/full/` — `system.txt`, `train_all.jsonl`, `val_all.jsonl`, `stats.json`, `noise_stats.json`, `validation.json`, `token_stats.json`.

## How it works
1. **Worlds**: flat_lab geometry, but a random object set per seed (3-7 objects, 7 classes, colours, duplicates for ambiguity, positions) + random robot state (pose, battery incl. low / critical,
   failed leg, no-go zone, partly explored map, saved places, history tasks). Test world objects are fixed, so train / val / test do not share worlds.
2. **44 intent families** (single motion, find / inspect / explore / patrol, 2-4 step missions, conditional query-then-act, 12 question types, no-go zones / save place, 10 follow-up dialogues,
   clarify / ambiguous / unknown place, refuse, confirm, unsafe text) with several phrasings each, numbers as digits or words, polite / casual wrappers, lower-case or Whisper-style capitals.
3. **Gold by construction**: every dialogue is run through the REAL agent (graph, safety checker, executor, tool layer, grounding check) with a `GoldLLM` that returns the scripted gold output.
   The prompts are built by the same `OllamaLLM` code that serves the model (train / serve parity), one example per LLM call. A dialogue is kept only if the OBSERVED behaviour (execute / answer /
   clarify / refuse / confirm) is in the family's allowed set, all calls pass the schema validator, the grounding check never fired on the gold reply, and the scripted outputs were all consumed.
   Replies are the template phrasing of the real tool results (grounded by construction).
4. **Noisy STT** (stage B, GPU): each unique utterance is synthesised by Piper under 4 conditions (mild / harsh / phone band-limit + SNR 3 dB / light reverb + clipping), transcribed by the
   deployed Whisper (large-v3-turbo CT2 int8_float16). Stage C keeps the clean gold if the transcript still carries the same slots (numbers, directions, colours, classes, places; known
   confusions and near spellings count as preserved; whole-sentence similarity >= 0.7), otherwise the gold becomes "Sorry, I did not catch that" (ask_user). Capped: clarify golds <= 25 % of the
   noisy set, confusion-table variants (lift / left, read / red, petrol / patrol ...) <= 25 %, surface-only (digits, capitals) <= 15 %. Noisy share of the final file: 30 %.
5. **Validation**: schema replay of every plan, grounding re-check of every reply, duplicates, exact-utterance overlap with the 100-command test set.

## Held out from the test set (and how strictly)
Any dialogue containing an utterance that equals a test-set utterance (normalised) is dropped (747 of 7,486 train dialogues, 22 of 345 val). Test world = fixed flat_lab objects, training worlds are random.
This is **exact-utterance + world** hold-out, not template hold-out: a test phrasing such as "navigate to the green barrel" can still occur with other slots in training. Val uses disjoint phrasing
templates for the banked families (hash of the template string mod 6), but not for inline templates (clarify, follow-ups).

## Stage A result (clean examples, 3 minutes with 10 workers)
| | train | val |
|---|---|---|
| examples | 8,400 | 420 |
| plan / respond | 6,968 / 1,432 | 341 / 79 |
| single / multi-step / multi-turn / question / behaviour / config | 2,189 / 1,449 / 1,450 / 1,914 / 1,277 / 121 | 106 / 70 / 84 / 109 / 47 / 4 |
| observed behaviour: execute / answer / clarify / refuse / confirm / none | 3,496 / 2,825 / 698 / 805 / 536 / 40 | 182 / 155 / 24 / 38 / 20 / 1 |
| unique utterances for the noise round trip | 5,134 (train + val) | |

Dialogues dropped (train, of 7,486): equals test utterance 747, family not applicable to the world 252, duplicate example 154, observed behaviour differs from intended 69 (e.g. a confirm
that the safety layer turned into a refuse), gold reply tripped the grounding check 9.

## Findings
- **The system prompt alone is ~2,400 Qwen3 tokens** (24 tool specs + rules); every example is 2.5-3.0 k tokens, only ~1 % of them are loss tokens. Training needs sequence length >= 3.1 k (the design
  said 2048) and spends almost all compute on the shared prefix. Options for 5.5, to be decided with a pilot: train at 3.1 k with batch 1, or shorten the tool descriptions, or two-stage tool selection.
- **Piper speech is almost never mis-transcribed by Whisper turbo** (mild / harsh: surface differences only, digits / capitals / "meters"). Real recognition errors appear only with band-limit + low SNR,
  and those are harsher than most real microphones. The noisy set therefore mixes real round-trip errors with confusion-table variants and is deliberately partly synthetic; **one Piper voice only**,
  so speaker variety is limited to speed / pitch / noise. Real microphone transcripts (5.5, category 9 replacement) remain the honest check.
- Not generated yet: re-plan examples (failed mission step), the `patrol places: "all"` form (design lists it, the schema / backend do not implement it), paraphrases by the base LLM (design 6.1 step 4),
  several reply styles (replies are the grounded templates, which is what we want the model to reproduce).
- Bugs found by the generator: `schedule` calls with inner arguments were rejected by the validator (fixed).

## Smoke test (before the full run)
300 + 40 examples, noise round trip on 120 utterances (5 min), merge + validation: no invalid plans, 0 duplicates, 0 test-set utterances. Noisy share 29 %, clarify golds 20 % of the noisy set.

## Full run
Started 2026-11-17 11:26 UTC: `scripts/p5_datagen.sh brain/data/full 8400 420`. Stage B (5,134 utterances x 4 conditions) measured ~2.4 s per utterance, so roughly 3.4 h. Final numbers (noisy share, clarify
share, validation, token stats) are written by the pipeline to `brain/data/full/noise_stats.json`, `validation.json`, `token_stats.json`; they are NOT in this document yet.
