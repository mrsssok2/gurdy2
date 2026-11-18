"""SFT data v2, stage A: clean examples from the v2 family table (varied phrasing + decline / no-state-change / context families).

  python3 -m brain.data_gen.generate_v2 --out brain/data/v2_new --train 4400 --val 500 [--workers 8]

Same machinery as generate.py (real agent + safety + guard run on every dialogue, gold = scripted LLM). Differences:
  * family table = families_v2.FAMILIES_V2
  * HOLD-OUT: utterances equal to a v1-test sentence, or equal / near-paraphrase (difflib ratio >= 0.75) of a sentence of
    testset_ood_v1.yaml, are dropped. Counted in stats.json as `equals_test_utterance`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from . import families_v2
from .generate import TEST, VAL_SEED0, generate
from ..agent.ollama_llm import build_system
from .lexicon import norm

OOD = Path(__file__).resolve().parent.parent / "eval" / "testset_ood_v1.yaml"
NEAR = 0.75


class Holdout(set):
    """`x in holdout`: exact match with a v1-test sentence, or a near paraphrase of an ood_v1 sentence."""

    def __init__(self, exact: set[str], ood: list[str]):
        super().__init__(exact)
        self.ood = ood

    def __contains__(self, x):
        if set.__contains__(self, x):
            return True
        for o in self.ood:
            m = SequenceMatcher(None, x, o)
            if m.real_quick_ratio() >= NEAR and m.quick_ratio() >= NEAR and m.ratio() >= NEAR:
                return True
        return False

    def __reduce__(self):
        return (Holdout, (set(self), self.ood))


def load_holdout() -> Holdout:
    v1 = {norm(t["say"]) for c in yaml.safe_load(TEST.read_text())["cases"] for t in c["turns"]}
    ood = sorted({norm(t["say"]) for c in yaml.safe_load(OOD.read_text())["cases"] for t in c["turns"]})
    return Holdout(v1, ood)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=4400)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system.txt").write_text(build_system())
    hold = load_holdout()
    families_v2.install()
    stats: dict = {"holdout_exact": len(hold), "holdout_ood_sentences": len(hold.ood), "near_ratio": NEAR, "splits": {}}
    for split, n, seed0 in (("train", a.train, a.seed), ("val", a.val, VAL_SEED0 + a.seed)):
        ex, drops, nd = generate(split, n, seed0, a.workers, hold)
        with open(out / f"{split}.jsonl", "w") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        stats["splits"][split] = {"examples": len(ex), "dialogues_tried": nd, "dropped": dict(drops),
                                  "by_group": dict(Counter(e["group"] for e in ex)), "by_family": dict(sorted(Counter(e["family"] for e in ex).items())),
                                  "by_kind": dict(Counter(e["kind"] for e in ex)), "by_behaviour": dict(Counter(e["behaviour"] for e in ex)),
                                  "multi_turn_examples": sum(1 for e in ex if e["n_turns"] > 1)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({s: {k: v for k, v in x.items() if k in ("examples", "dialogues_tried", "dropped", "by_group")} for s, x in stats["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
