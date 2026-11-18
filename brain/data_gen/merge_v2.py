"""SFT data v2, final stage: v1 data (re-used, NOT regenerated) + v2 new examples + text-level speech noise, no new audio.

  python3 -m brain.data_gen.merge_v2 --v1 brain/data/full --new brain/data/v2_new --out brain/data/v2 \
      --v1-n 2100 --v2-noisy 800 [--val-v1 250]

* v1 part: a random subsample of `train_all.jsonl` (clean + the REAL Piper->Whisper noisy examples that already exist),
  minus anything equal / near-paraphrase to the held-out test sentences.
* v2 noise, all text-level, from what already exists:
    - `asr_learned`  word substitutions LEARNED from the real Whisper outputs in `noisy.jsonl` (clean word -> what Whisper wrote,
                     counted over all 5k utterance round trips, kept when seen >= 3 times)
    - `confusion_table`  lexicon.CONFUSIONS reversed (the v1 table)
  Gold stays the clean plan if the slots survive (merge_noise.meaning_kept), otherwise it becomes the "did not catch that"
  clarification (<= 20 % of the noisy examples).
* A final guard replay: every gold plan is run through guard_plan with its own utterance and focus; a gold the guard would
  change (e.g. a save_place without a save cue) is dropped, so training never teaches what the runtime blocks.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from ..agent.guard import guard_plan
from .generate_v2 import load_holdout
from .lexicon import norm
from .merge_noise import CLARIFY, apply, meaning_kept, table_variant


def learn_asr_table(noisy_path: Path, min_count: int = 3) -> dict[str, list[tuple[str, int]]]:
    """clean word -> [(whisper word, count)] from 1:1 token replacements in the real round trips."""
    cnt: Counter = Counter()
    for line in open(noisy_path):
        r = json.loads(line)
        c = norm(r["say"]).split()
        for v in r["variants"]:
            n = norm(v["text"]).split()
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, c, n).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    for a, b in zip(c[i1:i2], n[j1:j2]):
                        if a != b and len(a) >= 3 and a.isalpha() and b.isalpha() and SequenceMatcher(None, a, b).ratio() >= 0.5:
                            cnt[(a, b)] += 1
    table: dict[str, list] = defaultdict(list)
    for (a, b), k in cnt.most_common():
        if k >= min_count:
            table[a].append((b, k))
    return {a: v[:5] for a, v in table.items()}


def asr_variant(rng: random.Random, say: str, table) -> str | None:
    toks = say.split()
    idx = [i for i, t in enumerate(toks) if re.sub(r"\W", "", t.lower()) in table]
    if not idx:
        return None
    for i in rng.sample(idx, k=min(len(idx), rng.choice([1, 1, 2]))):
        core = re.sub(r"\W", "", toks[i].lower())
        alts = table[core]
        toks[i] = rng.choices([b for b, _ in alts], [k for _, k in alts])[0]
    out = " ".join(toks)
    return out if out != say else None


def _focus_class(user: str) -> dict | None:
    m = re.search(r"FOCUS\s+object=#\d+ (.*?)\s{2,}place=", user)
    return {"object": {"class": m.group(1).split()[-1]}} if m else None


class _Any(set):
    def __eq__(self, other):
        return True

    __hash__ = None


def guard_ok(ex: dict) -> bool:
    """Would the runtime guard leave this gold plan alone?"""
    if ex["kind"] != "plan":
        return True
    try:
        out = json.loads(ex["assistant"])
    except json.JSONDecodeError:
        return False
    calls = out.get("calls")
    if not calls:
        return True
    g = guard_plan(ex["say"], calls, _focus_class(ex["user"]), recall=lambda color: _Any())   # class-by-colour needs the world: not judged offline
    return g.reply is None and g.clarify is None and g.confirm is None and not g.notes and g.calls == calls


def noisy_from(rng, plans, table, target, hold):
    pool = []
    for ex in plans:
        a = asr_variant(rng, ex["say"], table)
        if a:
            pool.append((ex, a, "asr_learned"))
        t = table_variant(rng, ex["say"])
        if t:
            pool.append((ex, t, "confusion_table"))
    pool = [p for p in pool if norm(p[1]) not in hold]        # noisy forms obey the hold-out too
    rng.shuffle(pool)
    kept = [p for p in pool if meaning_kept(p[0]["say"], p[1])]
    lost = [p for p in pool if not meaning_kept(p[0]["say"], p[1])]
    chosen = kept[: int(target * 0.8)] + lost[: int(target * 0.2)]
    rng.shuffle(chosen)
    out, seen, c = [], set(), Counter()
    for ex, noisy, kind in chosen[:target]:
        if (ex["id"], noisy) in seen:
            continue
        seen.add((ex["id"], noisy))
        ok = meaning_kept(ex["say"], noisy)
        gold = ex["assistant"] if ok else json.dumps(CLARIFY, separators=(",", ":"))
        e = apply(ex, noisy, gold, kind, f"{ex['id']}-n{len(out)}")
        out.append(e)
        c[(kind, "kept" if ok else "clarify")] += 1
    return out, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="brain/data/full")
    ap.add_argument("--new", default="brain/data/v2_new")
    ap.add_argument("--out", default="brain/data/v2")
    ap.add_argument("--v1-n", type=int, default=2100)
    ap.add_argument("--v2-noisy", type=int, default=800)
    ap.add_argument("--val-v1", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    v1, new, out = Path(a.v1), Path(a.new), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hold = load_holdout()
    table = learn_asr_table(v1 / "noisy.jsonl")
    stats: dict = {"asr_learned_words": len(table), "asr_learned_pairs": sum(len(v) for v in table.values()), "splits": {}}
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]          # noqa: E731
    for split, n_v1, n_noisy in (("train", a.v1_n, a.v2_noisy), ("val", a.val_v1, max(60, a.v2_noisy // 8))):
        v1_all = [e for e in load(v1 / f"{split}_all.jsonl") if norm(e["say"]) not in hold and guard_ok(e)]
        by_say = defaultdict(list)
        for e in v1_all:
            by_say[e["say"]].append(e)
        # keep dialogue-complete groups out of it: sample examples, the prompt of each is self-contained (state block + focus)
        rng.shuffle(v1_all)
        v1_part = v1_all[:n_v1]
        new_all = load(new / f"{split}.jsonl")
        n0 = len(new_all)
        new_ok = [e for e in new_all if guard_ok(e)]
        plans = [e for e in new_ok if e["kind"] == "plan" and e["family"] != "unsafe_text" and not e["assistant"].startswith('{"reply"')]
        for i, e in enumerate(new_ok):
            e["id"] = f"v2-{split}-{i:06d}"
        for i, e in enumerate(v1_part):
            e["id"] = f"v1-{e['id']}"
        noisy, comp = noisy_from(rng, plans, table, n_noisy, hold)
        allx = v1_part + new_ok + noisy
        rng.shuffle(allx)
        with open(out / f"{split}_all.jsonl", "w") as f:
            for e in allx:
                f.write(json.dumps(e) + "\n")
        fam = Counter(e["family"] for e in allx)
        stats["splits"][split] = {"total": len(allx), "v1_reused": len(v1_part), "v2_new": len(new_ok), "v2_new_dropped_by_guard_replay": n0 - len(new_ok),
                                  "v2_text_noise": len(noisy), "noise_composition": {f"{k}/{o}": v for (k, o), v in sorted(comp.items())},
                                  "by_group": dict(Counter(e["group"] for e in allx)), "by_kind": dict(Counter(e["kind"] for e in allx)),
                                  "multi_turn_examples": sum(1 for e in allx if e["n_turns"] > 1), "families": len(fam), "by_family": dict(sorted(fam.items()))}
    (out / "system.txt").write_text((new / "system.txt").read_text())
    (out / "merge_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({s: {k: v for k, v in x.items() if k not in ("by_family",)} for s, x in stats["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
