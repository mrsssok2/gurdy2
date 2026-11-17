"""Validation (design 6.1 step 7): every example replays through the tool validators; replies pass the grounding check;
no duplicates; no train/val/test leakage by exact utterance; report.

  python3 -m brain.data_gen.validate brain/data/full
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from ..agent.grounding import collect_facts, violations
from ..tools import TOOLS
from ..tools.schema import validate
from .generate import test_utterances
from .lexicon import norm


def check_plan(out: dict) -> list[str]:
    errs = []
    if "calls" in out:
        if not out["calls"] or not isinstance(out.get("continue"), bool):
            errs.append("calls/continue shape")
        for c in out["calls"]:
            if c.get("tool") not in TOOLS:
                errs.append(f"unknown tool {c.get('tool')}")
            else:
                errs += validate(c.get("args", {}), TOOLS[c["tool"]].parameters)
    elif "reply" not in out and "done" not in out:
        errs.append("no calls / reply / done")
    return errs


def main() -> None:
    d = Path(sys.argv[1])
    tests = test_utterances()
    rep: dict = {}
    train_say = set()
    for split in ("train", "val"):
        ex = [json.loads(l) for l in open(d / f"{split}_all.jsonl")]
        bad: Counter = Counter()
        keys = Counter()
        for e in ex:
            keys[(e["user"], e["assistant"])] += 1
            if e["kind"] in ("plan", "replan"):
                try:
                    errs = check_plan(json.loads(e["assistant"]))
                except json.JSONDecodeError:
                    errs = ["json"]
                for x in errs:
                    bad["plan:" + x[:60]] += 1
            else:
                m = re.search(r"TOOL RESULTS: (.*)\n", e["user"])
                try:
                    rows = json.loads(m.group(1))
                    entries = [{"tool": r["tool"], "args": r["args"], "result": {"ok": r["result"]["ok"], "data": r["result"]["data"], "error": r["result"]["error"]}} for r in rows]
                    v = violations(e["assistant"], collect_facts(entries, e["say"]), ["home", "room A", "room B", "doorway"])
                    if v:
                        bad["respond:grounding"] += 1
                except Exception as ex_:                       # noqa: BLE001
                    bad["respond:unparsable_results"] += 1
            if e["split"] == "val" or split == "val":
                pass
        says = {norm(e.get("say_clean", e["say"])) for e in ex}
        leak_test = len(says & tests)
        if split == "train":
            train_say = says
        rep[split] = {"examples": len(ex), "invalid": dict(bad), "exact_duplicates": sum(v - 1 for v in keys.values() if v > 1),
                      "utterances_equal_to_test": leak_test, "unique_utterances": len(says),
                      "noisy": sum(1 for e in ex if e.get("noisy")), "by_kind": dict(Counter(e["kind"] for e in ex)),
                      "by_group": dict(Counter(e["group"] for e in ex))}
        if split == "val":
            rep[split]["utterances_also_in_train"] = len(says & train_say)
    (d / "validation.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
