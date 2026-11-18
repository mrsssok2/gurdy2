"""Readiness gate + setup-failure detection for the final eval (no ROS, no Docker). Run: python3 -m unittest brain.tests.test_readiness"""
import json
import tempfile
import unittest
from pathlib import Path

from brain.eval.final_checks import evaluate
from brain.eval.final_report import first_pass_section, load, markdown, setup_failures, summarize
from brain.ros.readiness import first_reply_setup_failure, wait_ready


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class StubPort:
    """Becomes ready one piece at a time: `ready_at[name]` = poll number from which that thing is up."""

    def __init__(self, ready_at=None):
        self.polls = 0
        self.ready_at = {"nav": 1, "bt": 1, "gcm_node": 1, "lcm_node": 1, "map": 1, "pose": 1, **(ready_at or {})}

    def tick(self):
        self.polls += 1

    def up(self, k):
        return self.polls >= self.ready_at[k]

    def nav_ready(self):
        return self.up("nav")

    def lifecycle_state(self, node):
        k = {"bt_navigator": "bt", "global_costmap/global_costmap": "gcm_node", "local_costmap/local_costmap": "lcm_node"}[node]
        return "active" if self.up(k) else "inactive"

    def global_costmap_ready(self):
        return self.up("map")

    def pose(self):
        return (0.0, 0.0, 0.0) if self.up("pose") else None


class Res:
    def __init__(self, ok, reachable):
        self.ok, self.data = ok, {"reachable": reachable}


class StubBackend:
    def __init__(self, port, good_from=1):
        self.port, self.calls, self.good_from = port, 0, good_from

    def path_check(self, place=None, **kw):
        self.calls += 1
        return Res(True, self.calls >= self.good_from)


def run_gate(port, backend=None, timeout_s=30, poll_s=1.0):
    clk, lines = Clock(), []
    orig = port.tick

    def sleep(s):
        clk.sleep(s)
    # advance the stub's own poll counter once per poll
    import brain.ros.readiness as R
    real_check = R.check_all

    def counting(p, b=None, **kw):
        orig()
        return real_check(p, b, **kw)
    R.check_all = counting
    try:
        res = wait_ready(port, backend, timeout_s=timeout_s, poll_s=poll_s, clock=clk, sleep=sleep, log=lines.append)
    finally:
        R.check_all = real_check
    return res, lines


class Gate(unittest.TestCase):
    def test_ready_at_once(self):
        p = StubPort()
        res, lines = run_gate(p, StubBackend(p))
        self.assertTrue(res["ready"])
        self.assertEqual(res["polls"], 1)
        self.assertEqual(lines, [])

    def test_waits_for_each_piece_and_reports_what_is_missing(self):
        p = StubPort({"nav": 3, "map": 5, "pose": 4})
        res, lines = run_gate(p, StubBackend(p))
        self.assertTrue(res["ready"])
        self.assertEqual(res["polls"], 5)
        self.assertIn("nav_action_server", lines[0])
        self.assertTrue(any("global_costmap_has_map" in l for l in lines))
        self.assertEqual(set(res["checks"].values()), {True})

    def test_times_out_bounded_with_the_missing_item_named(self):
        p = StubPort({"map": 10 ** 9})                      # the map never reaches the global costmap
        res, _ = run_gate(p, StubBackend(p), timeout_s=10)
        self.assertFalse(res["ready"])
        self.assertLessEqual(res["wall_s"], 11)
        self.assertEqual([k for k, v in res["checks"].items() if not v], ["global_costmap_has_map"])

    def test_inactive_lifecycle_node_blocks(self):
        p = StubPort({"bt": 10 ** 9})
        res, _ = run_gate(p, StubBackend(p), timeout_s=5)
        self.assertFalse(res["ready"])
        self.assertIn("bt_navigator_active", [k for k, v in res["checks"].items() if not v])

    def test_plan_is_tried_only_when_everything_else_is_up_and_must_succeed(self):
        p = StubPort({"nav": 4})
        be = StubBackend(p, good_from=3)                    # planner says unreachable twice (costmap still filling), then works
        res, _ = run_gate(p, be)
        self.assertTrue(res["ready"])
        self.assertEqual(be.calls, 3)                       # no plan attempts during polls 1-3 (nav server missing)... then 3 tries
        self.assertEqual(res["polls"], 6)

    def test_plan_that_never_succeeds_means_not_ready(self):
        p = StubPort()
        res, _ = run_gate(p, StubBackend(p, good_from=10 ** 9), timeout_s=6)
        self.assertFalse(res["ready"])
        self.assertEqual([k for k, v in res["checks"].items() if not v], ["plan_to_room_B"])

    def test_a_raising_check_is_not_ready_not_a_crash(self):
        class Boom(StubPort):
            def lifecycle_state(self, node):
                raise RuntimeError("service call failed")
        p = Boom()
        res, lines = run_gate(p, StubBackend(p), timeout_s=3)
        self.assertFalse(res["ready"])
        self.assertTrue(any("RuntimeError" in l for l in lines))


def msgs(first_reply, user_wall=0.0, reply_wall=0.5):
    return [{"who": "user", "t": 0.0, "wall": user_wall, "text": "go to room B"}, {"who": "robot", "t": 0.1, "wall": reply_wall, "text": first_reply}]


class FirstReply(unittest.TestCase):
    def test_the_two_real_cases(self):
        self.assertIn("unknown space", first_reply_setup_failure(msgs("I cannot find a path there; it is blocked or in unknown space.")))
        self.assertIn("not running", first_reply_setup_failure(msgs("Navigation (Nav2 navigate_to_pose) is not running.", reply_wall=0.8)))

    def test_normal_and_late_and_other_replies_are_not_setup_failures(self):
        for text in ("Going to room B, 4.7 metres.", "I can't go there. I do not know a place or object called kitchen.", "That is inside or behind a no-go zone.",
                     "Walking forward 3 metres.", "I have not seen a box. Should I look for it?"):
            self.assertIsNone(first_reply_setup_failure(msgs(text)), text)
        self.assertIsNone(first_reply_setup_failure(msgs("Navigation (Nav2 navigate_to_pose) is not running.", reply_wall=5.0)))   # after 2 s: a real robot answer
        self.assertIsNone(first_reply_setup_failure([]))
        self.assertIsNone(first_reply_setup_failure([{"who": "user", "wall": 0.0, "text": "x"}]))

    def test_only_the_first_reply_counts(self):
        m = msgs("Going to room B, 4.7 metres.") + [{"who": "user", "wall": 20.0, "text": "go home"},
                                                     {"who": "robot", "wall": 20.5, "text": "I cannot find a path there; it is blocked or in unknown space."}]
        self.assertIsNone(first_reply_setup_failure(m))

    def test_status_is_an_infrastructure_failure(self):
        v = evaluate({"mission": "nav_room_b", "status": "setup_failed", "poses": [], "messages": [], "turns": []})
        self.assertEqual((v["success"], v["infra_failure"]), (False, True))


class Report(unittest.TestCase):
    def make_run(self, root: Path, rec_for: dict):
        (root / "missions").mkdir(parents=True)
        (root / "attempts").mkdir()
        (root / "meta.json").write_text(json.dumps({"started": "t", "commit": "abc", "model": "m"}))
        for mid, rec in rec_for.items():
            (root / "missions" / f"{mid}.json").write_text(json.dumps({"mission": mid, "poses": [], "turns": [], "stops": [], "estops": [], "falls": 0, **rec}))

    def test_first_pass_section_lists_only_the_setup_failures_and_shows_both_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            bad = {"status": "ok", "messages": msgs("Navigation (Nav2 navigate_to_pose) is not running.", reply_wall=0.8)}
            fine = {"status": "ok", "messages": msgs("Going to room B, 4.7 metres.")}
            self.make_run(run / "first_pass", {"find_then_home": bad, "nav_room_b": fine})
            self.make_run(run, {"find_then_home": fine, "nav_room_b": fine})
            rows, _ = load(run)
            first, _ = load(run / "first_pass")
            self.assertEqual(list(setup_failures(first)), ["find_then_home"])
            self.assertEqual(setup_failures(rows), {})
            md = "\n".join(first_pass_section(first, rows))
            self.assertIn("`find_then_home`", md)
            self.assertNotIn("`nav_room_b`", md)
            self.assertIn("first pass", md)
            self.assertIn("is not running", md)
            full = markdown(rows, {}, run, first_rows=first)
            self.assertIn("AFTER re-running", full)
            self.assertEqual(summarize(first)["missions_run"], 2)


if __name__ == "__main__":
    unittest.main()
