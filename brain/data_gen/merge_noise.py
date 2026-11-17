"""Stage C: build the noisy-STT examples from stage B transcripts + the confusion table, and write the final files.

  python3 -m brain.data_gen.merge_noise --dir brain/data/full --noisy brain/data/full/noisy.jsonl [--frac 0.3]

Gold rule (design 6.1 step 5): the gold plan stays the CLEAN intent if the noisy transcript still carries the same slots
(numbers, directions, colours, class words, places). A slot counts as preserved if the word is there, if the transcript word is a
known confusion of it (table), or if it is a near spelling (difflib >= 0.75). If a slot was lost or a different slot appeared,
the gold becomes a clarification ("did not catch that"). Plus confusion-table variants of clean utterances (recoverable by construction).
Writes train_all.jsonl / val_all.jsonl (clean + noisy) and noise_stats.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .lexicon import CONFUSIONS, OBJECT_WORDS, PLACES, UNITS, norm, words

SLOT_WORDS = set(["left", "right", "forward", "forwards", "backward", "backwards", "back", "ahead", "red", "blue", "yellow", "green", "white",
                  "orange", "black", "brown", "grey", "gray", "home", "doorway", "door", "room", "a", "b", "half", "twice", "metres", "metre",
                  "centimetres", "centimetre", "degrees", "around", "all", "box", "barrel", "extinguisher", "cone", "toolbox", "mannequin",
                  "chair", "crate", "drum", "dummy", "patrol", "explore", "find", "inspect", "stop"]
                 + [w for w in UNITS if w != "zero"] + ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"])
CLARIFY = {"calls": [{"tool": "ask_user", "args": {"question": "Sorry, I did not catch that. Could you say it again?"}}], "continue": False}
_REV = {}
for k, v in CONFUSIONS.items():
    if k != v and v not in ("for", "to", "a"):
        _REV.setdefault(v, []).append(k)


def canon(text: str) -> list[str]:
    t = norm(text)
    out = []
    for tok in t.split():
        if tok.isdigit() and int(tok) < 1000:
            out += words(int(tok)).split()
        else:
            out.append({"meters": "metres", "meter": "metre", "centimeters": "centimetres", "gray": "grey"}.get(tok, tok))
    s = " ".join(out).replace("tool box", "toolbox").replace("and ", "and ")
    return s.split()


def meaning_kept(clean: str, noisy: str) -> bool:
    c, n = canon(clean), canon(noisy)
    if SequenceMatcher(None, " ".join(c), " ".join(n)).ratio() < 0.7:      # not the same sentence any more, whatever the slots say
        return False
    cs, ns = [t for t in c if t in SLOT_WORDS], [t for t in n if t in SLOT_WORDS]
    cc, nc = Counter(cs), Counter(ns)
    for tok, k in cc.items():
        have = nc.get(tok, 0)
        if have >= k:
            continue
        near = any(CONFUSIONS.get(t) == tok or SequenceMatcher(None, t, tok).ratio() >= 0.75 for t in n if t not in cs or nc[t] > cc[t])
        if not near:
            return False
    extra = [t for t in nc if nc[t] > cc.get(t, 0)]
    for t in extra:                                         # a slot word appeared that the speaker did not say
        if not any(CONFUSIONS.get(t) == c_ or SequenceMatcher(None, t, c_).ratio() >= 0.75 for c_ in c):
            return False
    return True


def table_variant(rng: random.Random, say: str) -> str | None:
    toks = say.split()
    idx = [i for i, t in enumerate(toks) if re.sub(r"\W", "", t.lower()) in _REV]
    if not idx:
        return None
    i = rng.choice(idx)
    core = re.sub(r"\W", "", toks[i].lower())
    toks[i] = toks[i].lower().replace(core, rng.choice(_REV[core]))
    return " ".join(toks)


def apply(ex: dict, noisy_say: str, gold: str, kind: str, idn: str) -> dict:
    noisy_say = noisy_say.replace('"', "'")
    e = dict(ex)
    e["user"] = ex["user"].replace(f'USER SAID: "{ex["say"]}"', f'USER SAID: "{noisy_say}"')
    e.update(say_clean=ex["say"], say=noisy_say, assistant=gold, noisy=True, noise_kind=kind, id=idn)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--frac", type=float, default=0.3, help="noisy share of the FINAL file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    rng = random.Random(a.seed)
    trans = {}
    for line in open(a.noisy):
        r = json.loads(line)
        trans[r["say"]] = r["variants"]
    stats: dict = {}
    for split in ("train", "val"):
        clean = [json.loads(l) for l in open(d / f"{split}.jsonl")]
        plans = [e for e in clean if e["kind"] == "plan" and e["family"] != "unsafe_text"]
        target = int(round(len(clean) * a.frac / (1 - a.frac)))
        pool: list[tuple[dict, str, str]] = []                     # (example, noisy say, kind) candidates
        for ex in plans:
            vs = trans.get(ex["say"], [])
            for v in vs:
                t = v["text"].strip()
                if not t:
                    continue
                if canon(t) != canon(ex["say"]):
                    pool.append((ex, t, f"stt_{v['cond']}"))
                elif t != ex["say"]:
                    pool.append((ex, t, "stt_surface"))
            tv = table_variant(rng, ex["say"])
            if tv:
                pool.append((ex, tv, "confusion_table"))
        rng.shuffle(pool)
        real_kept = [p for p in pool if p[2].startswith("stt_") and p[2] != "stt_surface" and meaning_kept(p[0]["say"], p[1])]
        real_lost = [p for p in pool if p[2].startswith("stt_") and p[2] != "stt_surface" and not meaning_kept(p[0]["say"], p[1])]
        table = [p for p in pool if p[2] == "confusion_table"]
        surf = [p for p in pool if p[2] == "stt_surface"]
        # Mix: most noisy examples keep the clean gold (the speech is still understandable); a minority are "lost" -> clarify
        # (cap 25 %, else the model learns to clarify everything); confusion-table <= 25 %; surface-only forms <= 15 %.
        caps = [("real_kept", real_kept, 1.0), ("table", table, 0.25), ("surface", surf, 0.15), ("real_lost", real_lost, 0.25)]
        chosen: list = []
        for _, lst, cap in caps:
            chosen += lst[: min(int(target * cap), target - len(chosen))]
        real = real_kept + real_lost
        out_noisy, seen_keys, c = [], set(), Counter()
        for ex, noisy, kind in chosen:
            key = (ex["id"], noisy)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept = kind in ("confusion_table", "stt_surface") or meaning_kept(ex["say"], noisy)
            gold = ex["assistant"] if kept else json.dumps(CLARIFY, separators=(",", ":"))
            out_noisy.append(apply(ex, noisy, gold, kind, f"{ex['id']}-n{len(out_noisy)}"))
            c[(kind, "kept" if kept else "clarify")] += 1
        with open(d / f"{split}_all.jsonl", "w") as f:
            for e in clean + out_noisy:
                f.write(json.dumps(e) + "\n")
        stats[split] = {"clean": len(clean), "noisy": len(out_noisy), "noisy_target": target, "noisy_share": round(len(out_noisy) / (len(clean) + len(out_noisy)), 3),
                        "pool": {"real_errors_meaning_kept": len(real_kept), "real_errors_meaning_lost": len(real_lost), "confusion_table": len(table), "surface_only": len(surf)},
                        "composition": {f"{k}/{o}": v for (k, o), v in sorted(c.items())},
                        "gold_clarify_share_of_noisy": round(sum(v for (k, o), v in c.items() if o == "clarify") / max(1, len(out_noisy)), 3)}
    (d / "noise_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
