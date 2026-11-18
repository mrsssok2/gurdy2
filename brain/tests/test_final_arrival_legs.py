"""Per-leg arrival scoring (final eval scorer fix). Run: python3 -m unittest brain.tests.test_final_arrival_legs"""
import unittest

from brain.eval.final_checks import arrival_errors, evaluate


def claim(text, x, y):
    return {"who": "robot", "text": text, "pose": [x, y], "t": 0, "kind": "pump"}


class Legs(unittest.TestCase):
    GOALS = ["room B", "home"]

    def test_true_claims_on_both_legs_are_fine(self):
        rec = {"messages": [claim("I have arrived, 0.1 metres from the goal.", 4.4, -0.1), claim("I have arrived, 0.1 metres from the goal.", -0.1, 0.1)]}
        self.assertEqual(arrival_errors(rec, self.GOALS), [])

    def test_the_old_single_goal_scorer_would_have_failed_this_run(self):
        rec = {"messages": [claim("I have arrived.", 4.4, -0.1), claim("I have arrived.", 0.0, 0.0)]}
        self.assertEqual(len(arrival_errors(rec, "home")), 1)           # single goal: the room-B claim is judged against home
        self.assertEqual(arrival_errors(rec, self.GOALS), [])

    def test_a_false_claim_on_a_leg_is_still_caught(self):
        rec = {"messages": [claim("I'm in room B.", 1.8, 0.0)]}          # still in room A (not near home either)
        self.assertEqual(len(arrival_errors(rec, self.GOALS)), 1)
        rec = {"messages": [claim("I have arrived.", 4.4, 0.0), claim("I am back home.", 3.0, 0.0)]}   # second claim made in room B
        self.assertEqual(len(arrival_errors(rec, self.GOALS)), 1)

    def test_skipped_leg_is_not_a_false_claim_but_repeats_are(self):
        home_only = {"messages": [claim("I'm back home.", 0.1, 0.0)]}    # never reached room B (that is a different check), the claim itself is true
        self.assertEqual(arrival_errors(home_only, self.GOALS), [])
        twice = {"messages": [claim("I have arrived.", 0.1, 0.0), claim("I have arrived.", 0.1, 0.0)]}
        self.assertEqual(len(arrival_errors(twice, self.GOALS)), 1)      # one 'home' leg cannot be claimed twice

    def test_messages_without_pose_or_arrival_wording_are_ignored(self):
        rec = {"messages": [{"who": "robot", "text": "I have arrived.", "pose": None}, claim("Going home, 4.6 metres.", 4.0, 0.0), {"who": "user", "text": "I arrived", "pose": [9, 9]}]}
        self.assertEqual(arrival_errors(rec, self.GOALS), [])
        self.assertEqual(arrival_errors(rec, None), [])

    def test_the_two_missions_are_defined_with_leg_goals_and_the_recorded_run_passes(self):
        from brain.eval.final_suite import BY_ID
        self.assertEqual(BY_ID["nav_there_and_back"]["goal"], ["room B", "home"])
        self.assertEqual(BY_ID["go_b_report_home"]["goal"], ["room B", "home"])
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parents[2] / "brain/runs/final_eval/20261118_181226/missions/nav_there_and_back.json"
        if f.exists():                                                   # the recorded Gazebo run (gitignored): rescored under the fixed scorer
            v = evaluate(json.loads(f.read_text()))
            self.assertEqual(v["arrival_errors"], [])
            self.assertTrue(v["success"])


if __name__ == "__main__":
    unittest.main()
