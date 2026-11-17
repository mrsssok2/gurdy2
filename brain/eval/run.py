"""Run the test set: python3 -m brain.eval.run --agent keyword [--cat 1 3] [--json out.json] [--failures]"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .agents import AGENTS
from .harness import CATEGORIES, load_cases, run_all, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="keyword", choices=sorted(AGENTS))
    ap.add_argument("--cat", type=int, nargs="*", help="only these categories")
    ap.add_argument("--json", help="write per-case records + summary here")
    ap.add_argument("--failures", action="store_true", help="print every failed case with the reasons")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = [c for c in load_cases() if not a.cat or c["cat"] in a.cat]
    recs = run_all(cases, AGENTS[a.agent], a.seed)
    s = summarize(recs)
    print(f"agent={a.agent}  cases={s['n_cases']}  success={s['success_pct']}%")
    for c, v in s["by_category"].items():
        print(f"  {c} {v['name']:<28} {v['ok']:>3}/{v['n']:<3} {v['pct']:>5}%")
    for k in ("tool_sequence_match_pct", "argument_accuracy_pct", "direction_errors", "clarification", "safety", "grounding",
              "latency_ms", "llm_calls_per_turn", "tool_calls_per_executed_turn", "mean_sim_time_s"):
        print(f"  {k}: {s[k]}")
    if a.failures:
        for r in recs:
            if not r.ok or any(t.grounding_final for t in r.turns):
                print(f"{'FAIL' if not r.ok else 'GROUNDING'} {r.id}")
                for t in r.turns:
                    if t.problems:
                        print(f"    '{t.say}' -> {t.behaviour}: {'; '.join(t.problems)} | reply: {t.reply[:90]}")
                for t in r.turns:
                    if t.grounding_final:
                        print(f"    GROUNDING '{t.say}' -> {t.reply[:120]}")
                for p in r.end_problems:
                    print(f"    end_state: {p}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"agent": a.agent, "summary": s, "cases": [asdict(r) for r in recs]}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
