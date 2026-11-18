"""Three-column report for the final Gazebo eval: first pass | after the Nav2-setup re-runs | fresh run after the agent / backend fixes.
  python3 -m brain.eval.final_compare --first-pass RUN_OLD/first_pass --rerun RUN_OLD --fresh RUN_NEW --notes docs/results/phase5_final_eval_fixes.md \
      --doc docs/results/phase5_final_eval.md
Every column is scored with the CURRENT scorer (verdicts are computed from the stored records at report time), so the columns are comparable; the numbers the
old scorer printed at the time are in the notes. Pure stdlib."""
from __future__ import annotations

import argparse
from pathlib import Path

from .final_report import load, markdown, pct, summarize


def result(r: dict) -> str:
    rec, v = r["rec"], r["verdict"]
    if rec is None:
        return "not run"
    if v is None or v["infra_failure"]:
        return "INFRA (%s)" % rec.get("status")
    if v["success"]:
        return "PASS"
    return "WALL TIMEOUT" if rec.get("status") == "wall_timeout" else "TIMEOUT" if v["timed_out"] else "FAIL"


def compare(cols: list[tuple[str, list[dict]]]) -> list[str]:
    sums = [summarize(rows) for _, rows in cols]
    L = ["## Three runs side by side", "", "| | " + " | ".join(name for name, _ in cols) + " |", "|---|" + "---|" * len(cols),
         "| **mission success rate** | " + " | ".join(f"**{pct(s['success'], s['valid'])}**" for s in sums) + " |",
         "| missions valid / infrastructure failures | " + " | ".join(f"{s['valid']} / {s['infra_failures']}" for s in sums) + " |",
         "| falls | " + " | ".join(str(s["falls"]) for s in sums) + " |",
         "| wrong claims | " + " | ".join(str(s["wrong_claims"]) for s in sums) + " |",
         "| grounding violations in final replies | " + " | ".join(str(s["grounding_violations"]) for s in sums) + " |",
         "| distance walked | " + " | ".join(f"{s['path_m_total']} m" for s in sums) + " |", "",
         "### Per mission", "", "| mission | category | " + " | ".join(name for name, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    by = [{r["m"]["id"]: r for r in rows} for _, rows in cols]
    for r in cols[-1][1]:
        i = r["m"]["id"]
        cells = [result(b[i]) for b in by]
        mark = " **<-**" if len(set(cells)) > 1 else ""
        L.append(f"| `{i}` | {r['m']['category']} | " + " | ".join(cells) + f" |{mark}")
    changed = [r["m"]["id"] for r in cols[-1][1] if len({result(b[r["m"]["id"]]) for b in by}) > 1]
    L += ["", f"Missions whose result differs between the runs: {', '.join(f'`{c}`' for c in changed) or 'none'}.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-pass", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--doc", default=None)
    a = ap.parse_args()
    first, _ = load(Path(a.first_pass))
    rerun, _ = load(Path(a.rerun))
    fresh, meta = load(Path(a.fresh))
    cols = [("first pass", first), ("after setup re-runs", rerun), ("fresh run after the fixes", fresh)]
    md = ["# Phase 5.11 - final Gazebo eval: first pass, setup re-runs, fresh run after fixes (gurdy-sft-400, typed input)", ""]
    if a.notes and Path(a.notes).exists():
        md += [Path(a.notes).read_text().rstrip(), ""]
    md += compare(cols)
    md += ["---", "", "# Fresh run: full report", "", markdown(fresh, meta, Path(a.fresh))]
    out = "\n".join(md)
    if a.doc:
        Path(a.doc).write_text(out)
    s = [summarize(r) for _, r in cols]
    print(" | ".join(f"{n}: {x['success']}/{x['valid']} falls {x['falls']} wrong {x['wrong_claims']}" for (n, _), x in zip(cols, s)))


if __name__ == "__main__":
    main()
