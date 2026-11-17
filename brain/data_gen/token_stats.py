#!/usr/bin/env python3
"""Token lengths with the Qwen3 tokenizer (vision image has `tokenizers`):
  docker run --rm -v $PWD:/w -v ~/hf_models:/hf -v ~/p5_pkgs:/pk -e PYTHONPATH=/pk -w /w --entrypoint python3 gurdy2/vision:cuda \
      brain/data_gen/token_stats.py brain/data/full
Counts system + user + assistant + ~30 chat-template tokens per example (ChatML, thinking off)."""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

d = Path(sys.argv[1])
tok = Tokenizer.from_file("/hf/Qwen3-1.7B/tokenizer.json")
sys_n = len(tok.encode((d / "system.txt").read_text()).ids)
res = {"system_tokens": sys_n}
for split in ("train", "val"):
    ex = [json.loads(l) for l in open(d / f"{split}_all.jsonl")]
    n = np.array([sys_n + 30 + len(tok.encode(e["user"]).ids) + len(tok.encode(e["assistant"]).ids) for e in ex])
    asst = np.array([len(tok.encode(e["assistant"]).ids) for e in ex])
    res[split] = {"total_tokens": {"p50": int(np.percentile(n, 50)), "p95": int(np.percentile(n, 95)), "max": int(n.max())},
                  "assistant_tokens": {"p50": int(np.percentile(asst, 50)), "p95": int(np.percentile(asst, 95)), "max": int(asst.max())},
                  "over_2048": int((n > 2048).sum()), "over_3072": int((n > 3072).sum()), "over_4096": int((n > 4096).sum()),
                  "loss_token_share_pct": round(100 * float(asst.sum()) / float(n.sum()), 2)}
(d / "token_stats.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
