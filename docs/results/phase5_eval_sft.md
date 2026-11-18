# Phase 5.5 — SFT checkpoint step 400 on the 100-command set (2026-11-18)

Model `gurdy-sft-400`: Qwen3-1.7B + LoRA (r 32), SFT stopped at step ~406 / 525; evaluated adapter = last saved checkpoint, **step 400** (val_loss 0.01453). Not finished (resumable).
Eval: compact prompt, fixed scorer, `brain/eval` testset_v1 (100 cases), MockBackend, temperature 0, think off. Raw: `docs/results/data/phase5/eval_sft_400.json`, `phase5_eval_sft_400.txt`.

## Conversion (container `gurdy2/brain-train:cuda`)
- llama.cpp **tag b12494** (official release, source tarball sha256 `b40d84985b98330ca2d51c6eef3ead6e7ec1aae2af97f7452af708e5900e2fcb`): `convert_hf_to_gguf.py` + its `gguf-py`.
- LoRA merged into the **fp16** base (`/hf/Qwen3-1.7B`) with PEFT → f16 GGUF (3.4 GB) → **Q4_K_M** (1.05 GiB) with `llama-quantize` from the installed Ollama 0.40.16 tarball (NOT built from b12494; GGUF format is shared, the converter is the pinned part).
- Ollama model from `brain/tools/Modelfile.sft` (GGUF path made relative; files kept in `~/p5_gguf`, not in git). **Custom TEMPLATE** = exactly the training prompt (system + user + `<think>\n\n</think>\n\n`). Ollama's stock Qwen3 template would append ` /no_think` to the user text, which training never saw.

## Result (new scorer; base compact = 59 on the same scorer, keyword 55)
| category | n | keyword | base 1.7B compact | **SFT-400** |
|---|---|---|---|---|
| 1 single motion / navigation | 20 | 15 | 19 | **20** |
| 2 search and inspection | 12 | 9 | 9 | **11** |
| 3 spatial questions | 14 | 10.5 | 9 | **12** |
| 4 status, why, memory | 8 | 3 | 1 | **7** |
| 5 multi-step missions | 10 | 5 | 2 | **10** |
| 6 multi-turn dialogues | 8 | 1 | 3 | **8** |
| 7 ambiguous requests | 8 | 3 | 3 | **8** |
| 8 unsafe / risky | 10 | 8 | 7 | **10** |
| 9 noisy STT | 10 | 3 | 6 | **8** |
| **total** | 100 | 55 | 59 | **94** |

(keyword per-category from the baseline percentages: 75/75/50/37.5/80/12.5/12.5/80/30 of the category sizes; its categories 3 and 5 do not give whole numbers, rounded as printed.)

Other metrics: tool-sequence match 97.4 %, argument accuracy 100 %, direction errors 0, clarification asked when needed 10/10 (2 unnecessary), refusals 7/7, confirmations 100 %, **risky actions executed 0**,
**grounding: 28 replies checked, 0 rejected before fallback, 0 violations in the final reply** (base: 4 rejections). Latency p50 **416 ms**, p95 836 ms (base 398 / 1147); 1.16 LLM calls per turn.

## Failures (6)
c2_05 "locate the mannequin" → `where_is` instead of `find` (answers with the known location; arguably fine); c3_12 "is there a traffic cone in room B" → "I can't answer that." (wrong reply, no tool result used);
c4_07 "give me a report on that mission" → terse reply missing the expected go_to mention; c9_01 "walk forward to meters" and c9_10 "go hole" → clarification instead of the intended repair (hand-written noise; asking is the policy's allowed safe behaviour).

## Findings / caveats
- **With Ollama's JSON-schema `format` the same model scores 13/100** (`phase5_eval_sft_400_grammar.txt`): the schema's free-form `args: {"type":"object"}` derails a model trained without a grammar (double-nested `args`). The fine-tuned model runs **unconstrained** (`OllamaLLM(constrain=False)`, agent `gurdy-sft-400`; the old agent kept as `gurdy-sft-400-grammar`). Output validity unconstrained: 0 parse failures needed fallback in this run (every failure above is a behaviour miss, not broken JSON).
- Test utterances vs training data: **0 of 101 exact matches**, but the data generator's families (44) were written by the same author as the test set, so templates and phrasing overlap; 94 is an in-distribution number. Real-voice (Whisper) noise is still only hand-written in category 9.
- Single run, temperature 0, one model checkpoint (no comparison with later steps; SFT unfinished, optimizer state saved).
