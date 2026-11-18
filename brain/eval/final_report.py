"""Aggregate a final-eval run directory into summary.json + a markdown report.
  python3 -m brain.eval.final_report brain/runs/final_eval/<timestamp> [--doc docs/results/phase5_final_eval.md]
Run dir layout (written by scripts/p5_final_eval.sh): meta.json, missions/<id>.json (runner output), attempts/<id>.json ({"attempts": n, "log": [...]}).
Pure stdlib; works on a partial run (missions without a result are listed as 'not run')."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .final_checks import evaluate
from .final_suite import SUITE


def load(run: Path) -> list[dict]:
    rows = []
    meta = json.load(open(run / "meta.json")) if (run / "meta.json").exists() else {}
    for m in SUITE:
        f, af = run / "missions" / f"{m['id']}.json", run / "attempts" / f"{m['id']}.json"
        att = json.load(open(af)) if af.exists() else {"attempts": 0, "log": []}
        if f.exists():
            rec = json.load(open(f))
            v = evaluate(rec) if rec.get("poses") or rec.get("status") not in (None, "error") else None
            rows.append({"m": m, "rec": rec, "verdict": v, "attempts": att})
        else:
            rows.append({"m": m, "rec": None, "verdict": None, "attempts": att})
    return rows, meta


def summarize(rows: list[dict]) -> dict:
    ran = [r for r in rows if r["rec"] is not None]
    valid = [r for r in ran if r["verdict"] and not r["verdict"]["infra_failure"]]
    ok = [r for r in valid if r["verdict"]["success"]]
    cats: dict[str, list] = {}
    for r in valid:
        cats.setdefault(r["m"]["category"], []).append(r)
    stops = [s["latency_s"] for r in ran for s in r["rec"].get("stops", []) if s.get("moving_before") and s.get("latency_s") is not None]
    stops_fail = sum(1 for r in ran for s in r["rec"].get("stops", []) if not s.get("moving_before") or s.get("latency_s") is None)
    sim = [r["rec"].get("sim_s", 0.0) for r in valid]
    wall = [r["rec"].get("wall_s", 0.0) for r in valid]
    return {
        "missions_total": len(rows), "missions_run": len(ran), "valid": len(valid), "infra_failures": len(ran) - len(valid),
        "not_run": len(rows) - len(ran), "success": len(ok),
        "success_rate": (len(ok) / len(valid)) if valid else None,
        "success_rate_incl_infra": (len(ok) / len(ran)) if ran else None,
        "by_category": {c: {"n": len(v), "success": sum(1 for r in v if r["verdict"]["success"])} for c, v in sorted(cats.items())},
        "falls": sum(r["verdict"]["falls"] for r in valid), "wrong_claims": sum(r["verdict"]["wrong_claims"] for r in valid),
        "grounding_violations": sum(r["verdict"]["grounding_violations"] for r in valid),
        "path_m_total": round(sum(r["rec"].get("path_m", 0.0) for r in valid), 1),
        "sim_s_total": round(sum(sim), 0), "wall_s_total": round(sum(wall), 0),
        "stop_latency_s": {"n": len(stops), "median": round(statistics.median(stops), 2) if stops else None, "max": max(stops) if stops else None,
                           "unmeasurable": stops_fail},
        "restarts": sum(max(0, r["attempts"].get("attempts", 1) - 1) for r in ran),
        "llm_latency_p50_s": _p50([t["latency_s"] for r in valid for t in r["rec"].get("turns", [])]),
    }


def _p50(x):
    return round(statistics.median(x), 2) if x else None


def pct(a, b) -> str:
    return "n/a" if not b else f"{100.0 * a / b:.0f} % ({a}/{b})"


def markdown(rows: list[dict], meta: dict, run: Path, partial_note: str = "") -> str:
    s = summarize(rows)
    L = ["# Phase 5.11 - final Gazebo eval (gurdy-sft-400, typed input)", ""]
    L.append(f"Run: `{run.as_posix()}`  ·  started {meta.get('started', '?')}  ·  commit `{meta.get('commit', '?')}`  ·  model `{meta.get('model', '?')}`  ·  suite: {s['missions_total']} missions.")
    if partial_note:
        L += ["", f"**{partial_note}**"]
    L += ["", "Method: every mission runs in a FRESH Gazebo + Nav2 + policy + detector stack (headless, flat_lab, robot at home, empty semantic map, simulated battery 100 %, "
          "no faults, no zones, fresh memory db), one stack at a time. The real agent (`brain.agent`, LLM `gurdy-sft-400`) gets TYPED text; success is judged against "
          "GROUND TRUTH (Gazebo pose of the robot, world object positions), not against what the robot says. Per-mission sim-time limit; a mission that is still "
          "working at its limit is a timeout = failure. A Gazebo hang (no /clock progress for 90 s) or a container crash restarts the stack and the mission "
          "(max 3 attempts); a mission that never ran counts as an infrastructure failure and is excluded from the success rate (listed separately).", ""]
    L += ["## Headline", "", "| metric | value |", "|---|---|",
          f"| **Mission success rate** | **{pct(s['success'], s['valid'])}** |",
          f"| missions run / valid / infrastructure failures / not run | {s['missions_run']} / {s['valid']} / {s['infra_failures']} / {s['not_run']} |",
          f"| falls (tilt >= 60 deg for 0.5 s) | {s['falls']} |",
          f"| wrong claims (false arrival, ungrounded final reply, false honesty / battery / leg statements) | {s['wrong_claims']} (of which grounding violations in final replies: {s['grounding_violations']}) |",
          f"| stop latency (wall, from sending the stop text until measured speed < 0.02 m/s) | median {s['stop_latency_s']['median']} s, max {s['stop_latency_s']['max']} s, n = {s['stop_latency_s']['n']}"
          f" (unmeasurable: {s['stop_latency_s']['unmeasurable']}) |",
          f"| total sim time / wall time / distance walked | {s['sim_s_total']:.0f} s / {s['wall_s_total']:.0f} s / {s['path_m_total']} m |",
          f"| LLM turn latency p50 | {s['llm_latency_p50_s']} s |",
          f"| stack restarts (hang / crash) | {s['restarts']} |", "", "## By category", "", "| category | success |", "|---|---|"]
    for c, v in s["by_category"].items():
        L.append(f"| {c} | {pct(v['success'], v['n'])} |")
    L += ["", "## Per mission", "", "| mission | category | result | sim s (limit) | wall s | path m | falls | wrong claims | attempts |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, rec, v = r["m"], r["rec"], r["verdict"]
        if rec is None:
            L.append(f"| `{m['id']}` | {m['category']} | not run | - | - | - | - | - | {r['attempts'].get('attempts', 0)} |")
            continue
        res = "INFRA FAIL (%s)" % rec.get("status") if v is None or v["infra_failure"] else ("PASS" if v["success"] else ("TIMEOUT" if v["timed_out"] else "FAIL"))
        L.append(f"| `{m['id']}` | {m['category']} | {res} | {rec.get('sim_s', '-')} ({m['limit_s']}) | {rec.get('wall_s', '-')} | {rec.get('path_m', 0):.1f} | "
                 f"{v['falls'] if v else '-'} | {v['wrong_claims'] if v else '-'} | {r['attempts'].get('attempts', '?')} |")
    bad = [r for r in rows if r["rec"] is not None and (r["verdict"] is None or not r["verdict"]["success"])]
    if bad:
        L += ["", "## Failures (what exactly failed)", ""]
        for r in bad:
            rec, v = r["rec"], r["verdict"]
            L.append(f"### `{r['m']['id']}` - {r['m']['name']}")
            if v is None or v["infra_failure"]:
                L += [f"- infrastructure: status `{rec.get('status')}`: {str(rec.get('error', ''))[:300]}", ""]
                continue
            if v["timed_out"]:
                L.append(f"- timed out (limit {r['m']['limit_s']} s sim)")
            for c in v["checks"]:
                if not c["ok"]:
                    L.append(f"- check {c['check'][0]}: {c['why']}")
            for e in v["arrival_errors"]:
                L.append(f"- false arrival claim: {e}")
            if v["falls"]:
                L.append(f"- falls: {v['falls']} (max tilt {rec.get('max_tilt')} deg)")
            L.append("- transcript:")
            for t in rec.get("messages", [])[:14]:
                L.append(f"  - `{t['t']}s` {t['who']}: {t['text'][:200]}")
            L.append("")
    wc = [(r["m"]["id"], d) for r in rows if r["verdict"] for d in r["verdict"]["wrong_claim_details"]]
    L += ["", "## Wrong claims (all)", ""] + ([f"- `{i}`: {d}" for i, d in wc] or ["- none"])
    sl = [(r["m"]["id"], st) for r in rows if r["rec"] for st in r["rec"].get("stops", [])]
    if sl:
        L += ["", "## Stop latency (every measured stop)", "", "| mission | text | moving before | say() s | speed < 0.02 after (s) |", "|---|---|---|---|---|"]
        for i, st in sl:
            L.append(f"| `{i}` | {st['text']} | {st['moving_before']} ({st['speed_before']} m/s) | {st['say_s']} | {st['latency_s']} |")
    L += ["", "## What this does and does not show", "",
          "- Single run per mission (no variance estimate); one world (flat_lab, 5 objects); one model; typed input - voice is evaluated separately (`phase5_voice*.md`).",
          "- Battery is SIMULATED (`battery_sim_node`, level set by `ros2 param set`); the leg fault is SYNTHETIC (`leg_health_node fake_failed_leg`): the robot's legs keep working. "
          "These cases test the alert -> agent path and the agent's honesty, not physical fault handling.",
          "- 'Wrong claims' are counted by regex rules and ground-truth arrival checks (see `brain/eval/final_suite.py`); a false statement that no rule covers is not counted. Read the transcripts of passing missions too.",
          "- Mission checks were written BEFORE the run and are not tuned to the results. Known open issue (not in this suite): 'the other box' reference resolution (`phase5_live_test.md`).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    run = Path(a.run)
    rows, meta = load(run)
    summ = summarize(rows)
    json.dump({"summary": summ, "missions": [{"id": r["m"]["id"], "verdict": r["verdict"], "attempts": r["attempts"]} for r in rows]},
              open(run / "summary.json", "w"), indent=1, default=str)
    md = markdown(rows, meta, run, a.note)
    (run / "report.md").write_text(md)
    if a.doc:
        Path(a.doc).write_text(md)
    print(f"success {summ['success']}/{summ['valid']} (run {summ['missions_run']}/{summ['missions_total']}), falls {summ['falls']}, wrong claims {summ['wrong_claims']}")


if __name__ == "__main__":
    main()
