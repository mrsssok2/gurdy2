# Eval re-score (offline, no LLM calls)

Scorer fixes (2026-11-17): meters/metres/m, "not been seen"/"haven't seen"/"not seen", "81.8%"/"82 percent", `{"reply": ...}` unwrapped; facts (numbers, names, directions) still strict. Command: `docker compose run --rm brain python3 -m brain.eval.rescore <saved json>`.
Only reply checks changed; all other results are as saved. Old = as reported on the day, new = current scorer.


## qwen3-1.7b  (docs/results/data/phase5/eval_base_qwen3-1.7b.json)
| category | n | old | new |
|---|---|---|---|
| 1 single motion / navigation | 20 | 15 | 15 |
| 2 search and inspection | 12 | 9 | 9 |
| 3 spatial questions | 14 | 8 | 8 |
| 4 status, why, memory | 8 | 2 | 3 |
| 5 multi-step missions | 10 | 0 | 0 |
| 6 multi-turn dialogues | 8 | 3 | 3 |
| 7 ambiguous requests | 8 | 3 | 3 |
| 8 unsafe / risky | 10 | 8 | 8 |
| 9 noisy STT | 10 | 5 | 5 |
| **total** | 100 | **53** | **54** |
changed: c4_01 fail->ok

## qwen3-0.6b  (docs/results/data/phase5/eval_base_qwen3-0.6b.json)
| category | n | old | new |
|---|---|---|---|
| 1 single motion / navigation | 20 | 8 | 8 |
| 2 search and inspection | 12 | 5 | 5 |
| 3 spatial questions | 14 | 7 | 7 |
| 4 status, why, memory | 8 | 2 | 3 |
| 5 multi-step missions | 10 | 0 | 0 |
| 6 multi-turn dialogues | 8 | 1 | 1 |
| 7 ambiguous requests | 8 | 2 | 2 |
| 8 unsafe / risky | 10 | 7 | 7 |
| 9 noisy STT | 10 | 4 | 4 |
| **total** | 100 | **36** | **37** |
changed: c4_01 fail->ok

## qwen3-1.7b  (docs/results/data/phase5/eval_base_compact_qwen3-1.7b.json)
| category | n | old | new |
|---|---|---|---|
| 1 single motion / navigation | 20 | 19 | 19 |
| 2 search and inspection | 12 | 9 | 9 |
| 3 spatial questions | 14 | 7 | 9 |
| 4 status, why, memory | 8 | 0 | 1 |
| 5 multi-step missions | 10 | 2 | 2 |
| 6 multi-turn dialogues | 8 | 2 | 3 |
| 7 ambiguous requests | 8 | 3 | 3 |
| 8 unsafe / risky | 10 | 7 | 7 |
| 9 noisy STT | 10 | 6 | 6 |
| **total** | 100 | **55** | **59** |
changed: c3_10 fail->ok, c3_11 fail->ok, c4_01 fail->ok, c6_05 fail->ok
