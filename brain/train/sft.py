#!/usr/bin/env python3
"""QLoRA SFT of Qwen3 on the Gurdy agent data (milestone 5.5). Small own loop, no TRL.

Memory design for a 6 GB GPU at ~3.1k tokens:
  * base weights 4-bit NF4 (double quant) + bf16 compute, LoRA on all linear layers, gradient checkpointing, batch 1
    + gradient accumulation (token-weighted loss over the accumulated examples);
  * the LM head is only evaluated on the assistant positions (`logits_to_keep`): full logits for 3.1k tokens x 152k vocab
    in fp32 would be ~1.9 GB by themselves;
  * loss only on assistant tokens (answer + <|im_end|>). The prompt is exactly what Ollama sends at inference:
    system + user + assistant header + empty think block (thinking off).
Resumable: adapter + optimizer + step every --save-every steps; the example order depends only on (seed, epoch).

Full run:   see docs/results/phase5_sft_runbook.md
Smoke test: brain/tests/test_sft_smoke.py (tiny random model, CPU)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def encode_example(tok, system: str, ex: dict, max_len: int):
    """-> (input_ids, n_answer_tokens) or None if longer than max_len."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": ex["user"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(ex["assistant"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        return None
    return p + a, len(a)


def order(n: int, seed: int, epoch: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def lm_loss(model, ids, n_ans, device):
    """Summed cross-entropy over the n_ans answer tokens (hidden states -> logits only for those positions)."""
    import torch
    import torch.nn.functional as F
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, logits_to_keep=n_ans + 1, use_cache=False)
    logits = out.logits[0, :-1].float()
    target = x[0, -n_ans:]
    return F.cross_entropy(logits, target, reduction="sum")


@__import__("torch").no_grad()
def evaluate(model, data, device, limit=None):
    model.eval()
    tot, n = 0.0, 0
    for ids, n_ans in data[:limit]:
        tot += lm_loss(model, ids, n_ans, device).item()
        n += n_ans
    model.train()
    return tot / max(n, 1)


def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen3Config
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if args.tiny:  # CPU smoke test: random 2-layer model, real tokenizer vocab
        cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, max_position_embeddings=8192, tie_word_embeddings=True)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0}, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lc)
    model.config.use_cache = False
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="brain/data/full", help="dir with train.jsonl, val.jsonl, system.txt")
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--val-file", default="val.jsonl")
    ap.add_argument("--model", default="/hf/Qwen3-1.7B")
    ap.add_argument("--tokenizer", default=None, help="default: --model")
    ap.add_argument("--out", default="brain/runs/sft_v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=3328)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=100, help="val examples per periodic eval (full val at the end)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test with a tiny random model")
    ap.add_argument("--system-file", default=None, help="default: <data>/system.txt")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    device = "cpu" if args.tiny else "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.data)
    system = Path(args.system_file or d / "system.txt").read_text()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    def enc(path):
        ex = load_examples(path)
        if args.max_examples:
            ex = ex[:args.max_examples]
        e = [encode_example(tok, system, x, args.max_len) for x in ex]
        return [x for x in e if x], sum(1 for x in e if x is None)
    train, dropped = enc(d / args.train_file)
    val, vdropped = enc(d / args.val_file)
    lens = sorted(len(i) for i, _ in train)
    print(f"train {len(train)} (dropped {dropped} > {args.max_len} tokens), val {len(val)} (dropped {vdropped}); "
          f"tokens p50 {lens[len(lens)//2]} max {lens[-1]}; answer tokens mean {sum(a for _, a in train)/len(train):.1f}", flush=True)

    model = build_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params)/1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = len(train) // args.accum
    total = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(total * args.warmup))
    lr_at = lambda s: args.lr * (s + 1) / warm if s < warm else args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    step = 0
    state = out / "state.pt"
    if state.exists():
        st = torch.load(state, map_location="cpu")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(out / "adapter_model.safetensors")))
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed at step {step}", flush=True)

    log = open(out / "log.jsonl", "a")
    status = out / "status.txt"

    def write_status(msg):
        status.write_text(msg + "\n")

    def save(final=False):
        model.save_pretrained(out)
        torch.save({"opt": opt.state_dict(), "step": step}, state)
        if final:
            model.save_pretrained(out / "final")

    model.train()
    start_step = step
    t0, tok_seen, last = time.time(), 0, time.time()
    write_status(f"RUNNING step {step}/{total}")
    best = float("inf")
    while step < total:
        epoch, within = divmod(step * args.accum, len(train))
        ord_ = order(len(train), args.seed, epoch)
        batch = [train[ord_[(within + k) % len(train)]] for k in range(args.accum)]
        ntok = sum(n for _, n in batch)
        loss_sum = 0.0
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for ids, n_ans in batch:
            l = lm_loss(model, ids, n_ans, device)
            (l / ntok).backward()
            loss_sum += l.item()
            tok_seen += len(ids)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        el = time.time() - t0
        rec = {"step": step, "loss": loss_sum / ntok, "lr": lr_at(step - 1), "grad_norm": gn, "tok_per_s": tok_seen / max(el, 1e-9),
               "elapsed_s": el}
        if device == "cuda":
            rec["vram_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if step % args.eval_every == 0 or step == total:
            rec["val_loss"] = evaluate(model, val, device, None if step == total else args.eval_n)
        # ETA from this process's own speed (steps done since start / elapsed)
        rec["eta_h"] = (total - step) * el / (step - start_step) / 3600
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 5 == 0 or step == 1 or "val_loss" in rec:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if v is not None), flush=True)
        write_status(f"RUNNING step {step}/{total} loss {rec['loss']:.4f}" + (f" val {rec['val_loss']:.4f}" if "val_loss" in rec else ""))
        if step % args.save_every == 0 and step < total:
            save()
        if "val_loss" in rec and rec["val_loss"] < best:
            best = rec["val_loss"]
            model.save_pretrained(out / "best")
    save(final=True)
    write_status(f"DONE step {step}/{total} best_val {best:.4f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
