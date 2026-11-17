# Phase 5.3 part 2 — base Qwen3 (prompt only) on the 100-command test set

Date 2026-11-17. Mock flat_lab world (virtual clock), real Ollama on the RTX 4050 (Qwen3 Q4_K_M from the Ollama registry, thinking off,
temperature 0, `num_ctx` 4096). Command: `OLLAMA_URL=http://<wsl-ip>:11435 docker compose run --rm brain python3 -m brain.eval.run --agent qwen3-1.7b|qwen3-0.6b`.
Raw data: `docs/results/data/phase5/eval_base_qwen3-1.7b.json`, `..._qwen3-0.6b.json`. Single run each (temperature 0, so deterministic apart from GPU
non-determinism; no variance measured).

## Results (success = whole case passes: behaviour, tool trace, reply facts, end state)

| # | Category | keyword baseline | Qwen3-1.7B base | Qwen3-0.6B base |
|---|---|---|---|---|
| 1 | single motion / navigation (20) | 75.0 | 75.0 | 40.0 |
| 2 | search and inspection (12) | 75.0 | 75.0 | 41.7 |
| 3 | spatial questions (14) | 50.0 | 57.1 | 50.0 |
| 4 | status, why, memory (8) | 37.5 | 25.0 | 25.0 |
| 5 | multi-step missions (10) | 80.0 | **0.0** | **0.0** |
| 6 | multi-turn dialogues (8) | 12.5 | 37.5 | 12.5 |
| 7 | ambiguous requests (8) | 12.5 | 37.5 | 25.0 |
| 8 | unsafe / risky (10) | 80.0 | 80.0 | 70.0 |
| 9 | noisy STT (10, hand-written errors) | 30.0 | 50.0 | 40.0 |
| | **Total (100)** | **55** | **53** | **36** |

| Metric | keyword | 1.7B base | 0.6B base |
|---|---|---|---|
| tool-sequence match % | – | 79.1 | 55.7 |
| argument accuracy % | – | 87.4 | 74.7 |
| direction errors | – | 0 | 0 |
| clarification asked when needed % / unnecessary (of 95 clear turns) | – | 40 / 5 | 50 / 8 |
| refused when needed % / over-refusals | – | 85.7 / 0 | 85.7 / 0 |
| risky actions executed without confirmation or despite refusal | 1 | 0 | 0 |
| grounding: replies checked / rejected before fallback / violations in final reply | – | 27 / 16 / 0 | 51 / 29 / 0 |
| latency p50 / p95 per turn (wall, incl. prompt cache hits) | ~0 ms | 404 / 1454 ms | 335 / 1384 ms |
| LLM calls per turn | 0 | 1.30 | 1.61 |

## Reading it
- **Base 1.7B ties the keyword parser (53 vs 55); base 0.6B is clearly worse (36).** Prompt-only is not enough; this is the bar SFT (5.4 / 5.5) has to clear.
- **Multi-step: 0 / 10 for both.** The base models plan the first step only ("go to room B" for "go to room B and then come back home"). This is the largest, cleanest SFT target; the
  keyword parser scores 80 here because it splits on "then".
- Safety layer works independently of the model: 0 risky actions executed. Grounding: the models' phrased replies are rejected by the grounding check in 59 % (1.7B) / 57 % (0.6B)
  of the cases and replaced by the template after one retry, so **no ungrounded reply reached the user (0 final violations)**; the cost is that the reply is the template, not the model's words.
- 1.7B direction errors 0, but arguments are often wrong in structure (object given as `place`, `object: null`), which the schema validator rejects (behaviour "none").
- Latency: 0.4 s per turn at p50 with the prompt prefix cached; p95 1.4 s (turns with a second call).
- `c4_01` fails for both because the test requires "NN percent" in the reply and the model writes "95%": a strict reply check, not a wrong fact (TTS reads "percent").

## What was fixed while building the client (all would have affected any model)
1. **Plan schema forced a tool call after results** (`minItems: 1`), so after a query the model could not say "done" and looped until the 4-call cap (1.7B spatial questions 0/14).
   Plan output now also allows `{"done": true}` when tool results are in the prompt; the graph routes it to `respond`. Plus a prompt example that had taught `continue: true` for plain questions.
   Effect: spatial 0/14 -> 8/14 (1.7B).
2. **Safety checker crashed on malformed arguments** (`go_to` with `object: null` reached `backend.path_check`). It now validates against the tool schema first; invalid calls are
   left to the tool layer, which rejects them before any backend code runs (tests added).
3. **Validator rejected every `schedule` call with inner arguments** (a free-form `{"type":"object"}` was treated as "no allowed keys"). Fixed (test added). This bug predates today and was
   found by the data generator.
4. Infrastructure: a Windows-side Ollama also answers on `host.docker.internal:11434` and has other models, which gave a baffling 404. Our Ollama now listens on **11435**
   (`OLLAMA_HOST=0.0.0.0:11435`), the container reaches it through the WSL IP (`OLLAMA_URL`, see `compose.yaml`).

## Client design (`brain/agent/ollama_llm.py`)
One byte-identical system message (role, output rules, 3 examples, all 24 tool specs; ~2.4 k tokens) for plan / respond / replan; everything per turn (state, tool results, task, the user's words) is in the
user message. Plan / re-plan use JSON-schema constrained output (`anyOf`: calls + continue | reply | done); respond is free text. The system prompt is deliberately not tuned further for the base models:
the same prompt and the same prompt-building code produce the SFT data (5.4), so training and serving match by construction.

## Caveats
- One prompt, not a prompt search; a better hand-tuned prompt would raise the base numbers somewhat (multi-step in particular), so "SFT vs base" should be read as "SFT vs this base prompt".
- Category 9 uses hand-written STT errors (replaced by real Whisper transcripts in 5.5).
- Single runs; with 100 cases a difference of ~5 points is within noise.
