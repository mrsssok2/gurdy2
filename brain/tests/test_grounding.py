import pytest

from brain.agent import collect_facts, grounded_reply, violations
from brain.mock import MockBackend
from brain.tools import ToolLayer

PLACES = ["home", "room A", "room B", "doorway"]


def turn(utterance, *calls):
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    b.poll_events()
    l = ToolLayer(b)
    start = len(l.log)
    for name, args in calls:
        l.call(name, args)
    return b, l, l.log[start:], collect_facts(l.log[start:], utterance)


def kinds(v):
    return sorted(x["kind"] for x in v)


def test_true_reply_passes_and_invented_facts_fail():
    b, l, e, f = turn("where is the red box", ("where_is", {"object": {"class": "box", "color": "red"}}))
    o = e[0]["result"]["data"]["matches"][0]
    good = f"The red box is in {o['place']}, {o['distance_m']:.1f} metres {o['bearing_word']}."
    assert violations(good, f, PLACES) == []
    assert kinds(violations(good.replace("red box", "blue box"), f, PLACES)) == ["colour"]
    assert "number" in kinds(violations(f"The red box is {o['distance_m'] + 1.5:.1f} metres away.", f, PLACES))
    wrong_dir = "left" if o["bearing_word"] != "left" else "right"
    assert "direction" in kinds(violations(f"It is on the {wrong_dir}.", f, PLACES))
    assert "class" in kinds(violations("I also see a barrel.", f, PLACES))
    other = "room B" if o["place"] == "room A" else "room A"
    assert "place" in kinds(violations(f"It is in {other}.", f, PLACES))


def test_numbers_match_within_rounding_and_across_units():
    f = collect_facts([{"tool": "x", "args": {}, "result": {"ok": True, "error": None,
                        "data": {"distance_m": 4.23, "battery_pct": 82.4, "duration_s": 120, "angle": 90}}}])
    ok = ["4.2 metres", "4.3 m", "423 cm", "82 percent", "83%", "2 minutes", "120 seconds", "90 degrees", "90°"]
    for s in ok:
        assert violations(s, f) == [], s
    bad = ["5 metres", "50%", "4 minutes", "45 degrees", "10 seconds"]
    for s in bad:
        assert kinds(violations(s, f)) == ["number"], s


def test_user_words_and_args_count_as_facts():
    b, l, e, f = turn("walk left two metres", ("walk", {"direction": "left", "distance_m": 2.0}))
    assert violations("Walking left 2 metres.", f, PLACES) == []
    assert "direction" in kinds(violations("Walking right 2 metres.", f, PLACES))


def test_ids_and_counts():
    b, l, e, f = turn("what is nearby", ("whats_nearby", {"radius_m": 6.0}))
    objs = e[0]["result"]["data"]["objects"]
    assert violations(f"I see {len(objs)} objects.", f, PLACES) == []
    assert "number" in kinds(violations(f"I see {len(objs) + 3} objects.", f, PLACES))
    assert violations(f"Object #{objs[0]['id']} is close.", f, PLACES) == []
    assert "id" in kinds(violations("Object #99 is close.", f, PLACES))


def test_not_seen_reply_must_not_contain_a_position():
    b = MockBackend()                                   # nothing seen yet
    l = ToolLayer(b)
    l.call("where_is", {"object": {"class": "box", "color": "red"}})
    f = collect_facts(l.log, "where is the red box")
    assert violations("I have not seen a red box.", f, PLACES) == []
    v = violations("The red box is 2.0 metres ahead in room B.", f, PLACES)
    assert {"number", "direction", "place"} <= set(kinds(v))


def test_plurals_aliases_and_spelled_numbers():
    f = collect_facts([{"tool": "x", "args": {}, "result": {"ok": True, "error": None,
                        "data": {"objects": [{"class": "box"}, {"class": "box"}]}}}])
    assert violations("Two boxes.", f) == []
    assert "class" in kinds(violations("A crate and a drum.", f))
    assert "number" in kinds(violations("Five boxes.", f))


def test_regenerate_once_then_fallback_template():
    b, l, e, f = turn("where is the red box", ("where_is", {"object": {"class": "box", "color": "red"}}))
    o = e[0]["result"]["data"]["matches"][0]
    seen_fb = []

    def bad_then_good(fb):
        seen_fb.append(fb)
        return "It is 9.9 metres away." if len(seen_fb) == 1 else f"The red box is {o['distance_m']:.1f} metres {o['bearing_word']}."
    r = grounded_reply(bad_then_good, e, "where is the red box", PLACES)
    assert r.attempts == 2 and not r.fallback and seen_fb[0] is None and "9.9 metres" in seen_fb[1]
    assert len(r.rejected) == 1

    r = grounded_reply(lambda fb: "It is 9.9 metres away on the left.", e, "where is the red box", PLACES)
    assert r.fallback and r.attempts == 2 and r.violations == []
    assert f"{o['distance_m']:.1f} metres" in r.text and "9.9" not in r.text


def test_good_reply_costs_one_generation():
    b, l, e, f = turn("status", ("status", {}))
    pct = e[0]["result"]["data"]["battery_pct"]
    r = grounded_reply(lambda fb: f"Battery {pct:.0f} percent.", e, "status", PLACES, extra_facts=[])
    assert r.attempts == 1 and not r.fallback
