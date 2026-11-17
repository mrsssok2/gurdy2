"""The evaluation machinery itself: test-set shape, the oracle must solve every case, scoring helpers."""
import collections

from brain.eval.agents import AGENTS
from brain.eval.harness import args_match, direction_errors, load_cases, run_all, summarize, trace_match
from brain.tools import TOOLS

DESIGN_COUNTS = {1: 20, 2: 12, 3: 14, 4: 8, 5: 10, 6: 8, 7: 8, 8: 10, 9: 10}


def test_testset_has_the_designed_shape():
    cases = load_cases()
    assert len(cases) == 100 and len({c["id"] for c in cases}) == 100
    assert collections.Counter(c["cat"] for c in cases) == DESIGN_COUNTS
    for c in cases:                                   # every expected call names a real tool; every turn has an utterance
        for t in c["turns"]:
            assert t["say"]
            for trace in t.get("expect", {}).get("calls", []):
                assert all(x["tool"] in TOOLS for x in trace), c["id"]
    assert all(c.get("noise") == "synthetic" for c in cases if c["cat"] == 9)


def test_oracle_solves_every_case():
    """If the oracle fails a case, the case (or the scoring) is broken, not the agent."""
    recs = run_all(load_cases(), AGENTS["oracle"])
    bad = [(r.id, [t.problems for t in r.turns if t.problems], r.end_problems) for r in recs if not r.ok]
    assert not bad, bad
    s = summarize(recs)
    assert s["safety"]["violations_executed_without_confirmation_or_despite_refusal"] == 0
    assert s["grounding"]["violations_in_final_reply"] == 0


def test_keyword_baseline_is_clearly_worse_than_oracle_and_deterministic():
    a = summarize(run_all(load_cases(), AGENTS["keyword"]))
    b = summarize(run_all(load_cases(), AGENTS["keyword"]))
    assert a["success_pct"] == b["success_pct"] and a["by_category"] == b["by_category"]
    assert a["success_pct"] < 80


def test_args_match_numbers_lists_and_subsets():
    assert args_match({"distance_m": 2}, {"distance_m": 2.04}) == (1, 1)
    assert args_match({"distance_m": 2}, {"distance_m": 1.0}) == (0, 1)
    assert args_match({"object": {"class": "box"}}, {"object": {"class": "box", "color": "red"}}) == (1, 1)
    assert args_match({"places": ["a", "b"]}, {"places": ["A", "b"]}) == (2, 2)
    assert args_match({"places": ["a", "b"]}, {"places": ["a"]})[0] == 0


def test_direction_errors_count_swapped_pairs_only():
    exp = [{"tool": "turn", "args": {"direction": "left", "angle_deg": 90}}]
    assert direction_errors(exp, [{"tool": "turn", "args": {"direction": "right", "angle_deg": 90}}]) == 1
    assert direction_errors(exp, [{"tool": "turn", "args": {"direction": "left", "angle_deg": 45}}]) == 0
    assert trace_match(exp, [{"tool": "ask_user", "args": {}}])[0] is False


def test_reply_normalisation_accepts_variants_but_not_wrong_facts():
    from brain.eval.harness import normalize_reply as n
    assert "4.7 metres" in n("about 4.7 meters") and "4.7 metres" in n("4.7 m away")
    assert "81.8 percent" in n('{"reply": "Battery 81.8%"}') and "82 percent" in n("82 per cent")
    for s in ("The toolbox has not been seen", "I haven't seen it", "not seen yet", "it was never seen"):
        assert "not seen" in n(s)
    assert "not seen" not in n("I do not see anything")          # different statement, not rewritten
    assert "left" in n("it is to the left") and "right" not in n("it is to the left")
