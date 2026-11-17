"""Re-score saved eval results offline (no LLM calls) with the current reply scorer.

Only reply checks are re-run (the saved reply text + the expectation from the test set); behaviour / call / argument /
state problems are kept as saved. Usage: python3 -m brain.eval.rescore docs/results/data/phase5/eval_base_*.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from brain.eval.harness import build_world, check_reply, load_cases

REPLY_PREFIXES = ("reply ", "no distance near")


def rescore(path: str) -> dict:
    cases = {c["id"]: c for c in load_cases()}
    d = json.load(open(path))
    old, new, n, flips = defaultdict(int), defaultdict(int), defaultdict(int), []
    for rec in d["cases"]:
        case = cases[rec["id"]]
        cat = rec["category"]
        ok_new = not rec["end_problems"]
        for turn, spec in zip(rec["turns"], case["turns"]):
            keep = [p for p in turn["problems"] if not p.startswith(REPLY_PREFIXES)]
            b = build_world(case.get("scenario", {}))
            fresh = check_reply(spec.get("expect", {}), turn["reply"], b)
            ok_new = ok_new and not keep and not fresh
        n[cat] += 1
        old[cat] += bool(rec["ok"])
        new[cat] += ok_new
        if ok_new != bool(rec["ok"]):
            flips.append((rec["id"], bool(rec["ok"]), ok_new))
    return {"agent": d["agent"], "n": n, "old": old, "new": new, "flips": flips,
            "names": {int(k): v["name"] for k, v in d["summary"]["by_category"].items()}}


def main(paths: list[str]) -> None:
    for p in paths:
        r = rescore(p)
        print(f"\n## {r['agent']}  ({p})")
        print("| category | n | old | new |\n|---|---|---|---|")
        for c in sorted(r["n"]):
            print(f"| {c} {r['names'].get(c, '')} | {r['n'][c]} | {r['old'][c]} | {r['new'][c]} |")
        print(f"| **total** | {sum(r['n'].values())} | **{sum(r['old'].values())}** | **{sum(r['new'].values())}** |")
        print("changed:", ", ".join(f"{i} {'ok->FAIL' if o else 'fail->ok'}" for i, o, _ in r["flips"]) or "none")


if __name__ == "__main__":
    main(sys.argv[1:])
