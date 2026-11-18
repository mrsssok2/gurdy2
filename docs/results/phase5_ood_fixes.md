# Phase 5: fixes for the OOD weaknesses (code guard + SFT data v2)

Source of the problems: `docs/results/phase5_eval_ood.md` (gurdy-sft-400: 26/40 on `ood_v1`).
**`ood_v1` stays held out**: no sentence of it, and no near-paraphrase (difflib ratio >= 0.75), is in the v2 training data (checked, see below).
Nothing here was re-evaluated with a model yet (no Ollama / GPU use). The model-level effect is unknown until SFT v2 is trained and `run --set ood_v1` is repeated.

## A. Code fixes (no training) - `brain/agent/guard.py`, wired into `graph.py`

| Weakness (ood_v1 case) | Fix | Where |
|---|---|---|
| invented tool `continue` for "carry on" (o19) | aliases `continue / resume / carry_on / keep_going / proceed ...` -> `mission_control(resume)`; pause / cancel aliases too | `normalize_call` |
| invented tools `get`, ... (o27) | any other tool name -> "I can't do that. I don't have a way to do it." (verdict refuse, nothing runs) | `guard_plan` |
| `save_place home` as a side effect (o31) | `save_place` / `no_go_zone add|remove` run only if the user's words carry a request cue (save / remember / mark / name ... near place / spot / here, typo tolerant; keep out / avoid / off limits / don't go into ...). Otherwise the call is dropped; the rest of the plan runs and the reply adds "I did not save a place, because you did not ask for that."; if it was the only call, the robot asks ("Do you want me to save this place as X?", confirmation gate) or explains how to ask. `no_go_zone list` needs no cue. A re-plan after a failed step never changes saved state. | `guard_plan`, `_on_mission_failed` |
| unknown objects replaced by a known one (o09 cylinder, o12 table) | pre-LLM check (no LLM call): everyday nouns the robot has no concept of (table, cylinder, wall, window ... ~90 words) -> "I can't do that. I don't know what a table is, and I have not seen one." A class outside the 7 in a tool call is declined the same way. | `unknown_noun`, `graph.n_route` |
| unknown rooms / destinations (kitchen ...) | "I can't go there. I do not know a place or object called kitchen." + the known places (clarify, as before) | `n_route` |
| class never named but filled in by the model (o13 "whatever is green" -> green box) | if the user did not name the class (fuzzy: typos / Whisper variants count), is not using "it / that one / the other", and the class is not the FOCUS object (or the only seen object of the named colour) -> ask "Which kind of object do you mean?" | `_substitution` |
| raw validator text shown ("args: give at most one of ...") (o16) | INVALID_ARGS / UNKNOWN_TOOL results are phrased as "I could not work out how to do that. Could you say it another way?" | `templates.error_text` |

Judgement calls: `chair` IS one of the 7 object classes, so it is not declined (the task text listed it as an example of unknown). Words with safety meaning (person, door, wall in a safety context) are left to the planner / safety layer. A destination that is an object noun ("go to the table") is asked like an unknown place, as in `testset_v1` c7_08.

Tests: `brain/tests/test_guard.py` (16, aliases, cues, state changes, unknown objects, substitution, end to end) and `test_data_gen_v2.py`.
Regression: full brain suite 293 passed; oracle 100/100 on v1 and ood_v1 (scoring still consistent); keyword baseline v1 57 %, ood_v1 37.5 % (unchanged in kind).

## B. SFT data v2 - no new audio

`python3 -m brain.data_gen.generate_v2 --out brain/data/v2_new --train 3500 --val 450` then `python3 -m brain.data_gen.merge_v2` (data dirs are gitignored like v1; regenerate in ~2 min on CPU).

**Size:** train **6390** examples (plan 5532 / respond 858; 1785 from multi-turn dialogues; 62 families), val 799. Composition (train): v1 data re-used 2100 (a random subsample of `train_all.jsonl`, which already contains the real Piper->Whisper examples), v2 new 3491, v2 text-noise 799. Largest groups: multi-turn 1639, single 1224, question 907, behaviour 773, multistep 716, unknown/decline 587, state_neg 355, config 189. 400 steps x 16 = one epoch.

**What is new**
- Wider phrasing (`variation.py`, applied to every v1 family at p = 0.85 and to the new ones): casual prefixes / suffixes ("yo", "ok so", "pls", "real quick"), slang ("u", "wanna", "gimme", "whats", "fwd"), typos (1-2 per sentence on non-slot words; directions, colours, numbers, "room", pronouns are never touched), indirect and polite wrappers ("would you be able to ...", "any chance you could ..."), run-on / changed joiners ("n then", "after that"), fillers ("uh", "you know"), very short forms (extra terse phrase banks: "battery?", "{d} {m} pls", "home please"), ~15 new templates per intent.
- New families: `unknown_object` (321 ex: ~100 nouns x 25 intents, 4-6 reply wordings, 25 % as 2nd turn after a normal answer; nothing is called), `unknown_capability` (166: carry / operate things / media / body / settings -> "I can't do that ..."), `state_neg` (355: a place or spot is only commented on, gold has no save_place / no_go_zone; also "inspect X then come back" sequences), `resume` (220: 25 wordings), `follow_other` (335: "and the other box" -> asks for the whole class, the tool layer lists / asks), `follow_it` (347: "check it out", "is it far" after a where-is), `ref_unclear` ("the green thing": ask for the kind, never guess), `unknown_place2` (rooms the router does not know -> go_to, safety asks). `f_zone` / `f_save_place` get 2.5x weight with ~20 more explicit phrasings, so requested state changes stay well covered.
- The system prompt is unchanged (the v2 model trains with the same prompt as v1, so the serving code needs no change).

**Noise without audio:** (1) real Whisper outputs: the v1 subsample keeps its existing noisy examples; (2) `asr_learned`: 238 word substitutions (107 words) learned from the 5k real Piper->Whisper round trips in `noisy.jsonl` (clean word -> what Whisper wrote, seen >= 3 times) applied to new utterances; (3) the v1 `CONFUSIONS` table. Gold stays the clean plan only if the slots survive (`meaning_kept`), else the "did not catch that" clarification (20 % of the text-noise, composition: asr_learned kept 331-ish / clarify ~160, table 309). No new audio was needed, so no extra GPU / Piper / Whisper time.

**Hold-out check:** train utterances equal to a `testset_v1` sentence are dropped (185 dialogues), and so is anything with difflib ratio >= 0.75 to any `ood_v1` sentence (also for noisy variants and the v1 subsample). Measured on the final train file: 0 exact matches, max similarity to any ood sentence **0.745**, 0 above 0.75 (115 unique utterances are between 0.70 and 0.75: short generic commands such as "go to room b" variants). Samples: `docs/results/data/phase5/sft_v2_samples.txt`.
**Gold audits:** every gold plan was produced by the real agent (guard included) and its observed behaviour checked; every gold plan is replayed through `guard_plan` (9 + 1 dropped); the declining replies pass the grounding check (test).

## C. SFT v2 run (prepared, NOT started)

`scripts/p5_sft_v2.sh` - base Qwen3-1.7B, LoRA r 32 / alpha 64, lr 2e-4 cosine, effective batch 16, `--max-steps 400`, saves every 50 steps to `brain/runs/sft_v2` (container `gurdy2-sft2`, log `brain/runs/sft_v2.log`). Same command resumes. The script first unloads the models this project's Ollama holds in VRAM (right now 3.7 GiB used) and refuses to start if more than 2.5 GiB are still used or any container runs. Smoke-tested on CPU with the tiny model on the v2 files: all 6390 examples encode (p50 1463 tokens, max 2134, none over 3328). Expected duration is an estimate from v1 (about 60 s/step): ~6.5-7.5 h.

After training: convert (`docs/results/phase5_eval_sft.md`, llama.cpp b12494), then run `run --set v1` and `run --set ood_v1` with the new agent, compare with gurdy-sft-400 (94 / 65 %).
