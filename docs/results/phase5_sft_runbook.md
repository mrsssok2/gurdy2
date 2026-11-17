# Phase 5.5 — QLoRA SFT: setup, tool-block measurement, start command

## Setup (built, CPU smoke test passed; GPU path NOT run yet — Stage B had the GPU)
- Image `gurdy2/brain-train:cuda` (`docker/brain_train/Dockerfile`) = `gurdy2/vision:cuda` + transformers 4.51.3, peft 0.15.2,
  accelerate 1.6.0, bitsandbytes 0.45.5 (only pip packages were downloaded; PyTorch 2.5.1+cu121 reused). No TRL/datasets.
- `brain/train/sft.py`: 4-bit NF4 + double quant, bf16 compute, LoRA r 32 / alpha 64 / dropout 0.05 on all 7 linear layers, gradient
  checkpointing, batch 1 × accumulation 16 (token-weighted loss), AdamW lr 2e-4 cosine to 10 %, 3 % warm-up, grad clip 1.0.
  Loss only on the answer + `<|im_end|>`. LM head evaluated only on those positions (`logits_to_keep`; verified equal to the full-logit
  loss) — full fp32 logits at 3.1k tokens would be ~1.9 GB. Prompt = Qwen3 chat template, thinking off (empty think block), checked by test.
  Examples > 3328 tokens are dropped and counted (system prompt alone is 2423 tokens).
- Resumable (adapter + optimizer + step every 100 steps; just re-run the same command). Writes `brain/runs/sft_v1/{log.jsonl,status.txt,best,final}`.
- Smoke test: `brain/tests/test_sft_smoke.py` (tiny random Qwen3 on CPU, real tokenizer + real data: format, mask, 4 steps, resume) passes.
  It proves the loop, not the 4-bit/GPU path: bitsandbytes on the GPU and the real memory peak are **unmeasured**. Watch the first steps
  (`vram_peak_gib` in the log; stop if it is near 5.8). Fallback if OOM: `--max-len 3072`, `--lora-r 16`.

## Tool block: how much shorter (measured with the Qwen3 tokenizer, `python3 -m brain.train.compact_tools`)
Lossless compact notation (`walk(direction: forward|..., distance_m: num[0.1,3.0]) -- description`), decoded back to the JSON schema and
compared for equality for all 24 tools. Eval prompt NOT changed.

| tool block | tokens | vs current |
|---|---|---|
| current JSON schema | 2010 | — |
| compact, lossless | 964 | −52 % (−1046 tokens) |
| + `CLASS` alias for the repeated 7-class enum | 828 | −59 % |
| descriptions removed (upper bound, loses information) | 720 | — |

System prompt 2423 → 1377 tokens (−43 %): every training example gets ~40 % shorter, so SFT time drops roughly in proportion and
the 3.1k length limit becomes ~2k. **Decided 2026-11-17: compact + CLASS alias everywhere** (`build_system()` default; `build_system("json")` / agent `qwen3-1.7b-json` keep the old prompt for reference;
`python3 -m brain.train.render_system brain/data/full` writes `system.txt` (compact) + `system_json.txt`). The 5.3 base numbers (1.7B 53, 0.6B 36) used the JSON prompt;
the fair base baseline with the compact prompt is run by `scripts/p5_tonight.sh` (`docs/results/phase5_eval_base_compact.txt`).

## Overnight chain: `nohup scripts/p5_tonight.sh > brain/runs/tonight.log 2>&1 &` (waits for DONE + empty docker ps, renders prompt, base eval, starts SFT; re-runnable)

## Start by hand (after `brain/data/full/status.txt` says DONE and `docker ps` is empty)
    scripts/p5_sft.sh --epochs 2                 # detached container gurdy2-sft, checks both conditions itself
Check first that the final files are `train.jsonl` / `val.jsonl` (else pass `--train-file`, `--val-file`).

## Monitor
    cat brain/runs/sft_v1/status.txt                     # RUNNING step n/N loss .. val ..  /  DONE
    docker logs --tail 5 gurdy2-sft                      # loss, lr, tok_per_s, vram_peak_gib, eta_h
    tail -n 3 brain/runs/sft_v1/log.jsonl
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
Stop: `docker stop gurdy2-sft` (resume = run the script again). Early stop leaves the LR high; `brain/runs/sft_v1/best` = lowest val loss.

## Duration — ESTIMATE, not measured
~8400 examples × ~2.5k tokens ≈ 21M tokens/epoch, 525 optimizer steps/epoch (2 epochs = 1050). QLoRA with checkpointing on a 4050 laptop
is probably 1–2.5k tokens/s → **~2.5–6 h per epoch, 5–12 h for 2 epochs** (compact prompt: ~40 % less). The log prints `eta_h` after the
first step; trust that after ~10 steps and decide between `--epochs 2` and `--epochs 1`.
