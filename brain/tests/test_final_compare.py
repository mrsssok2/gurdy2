"""The three-column report. Run: python3 -m unittest brain.tests.test_final_compare"""
import json
import tempfile
import unittest
from pathlib import Path

from brain.eval import final_compare


def write_run(root: Path, results: dict):
    (root / "missions").mkdir(parents=True)
    (root / "attempts").mkdir()
    (root / "meta.json").write_text(json.dumps({"started": "t", "commit": "c", "model": "m"}))
    for mid, rec in results.items():
        (root / "missions" / f"{mid}.json").write_text(json.dumps({"mission": mid, "poses": [], "turns": [], "stops": [], "estops": [], "falls": 0, "messages": [], **rec}))


class Compare(unittest.TestCase):
    def test_three_columns_and_changed_missions(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_run(d / "a", {"nav_room_b": {"status": "infra_failed"}, "nav_doorway": {"status": "wall_timeout"}})
            write_run(d / "b", {"nav_room_b": {"status": "timeout"}, "nav_doorway": {"status": "wall_timeout"}})
            write_run(d / "c", {"nav_room_b": {"status": "sim_hang"}, "nav_doorway": {"status": "ok", "messages": []}})
            notes = d / "n.md"
            notes.write_text("## Notes\nhello\n")
            doc = d / "out.md"
            import sys
            argv = sys.argv
            sys.argv = ["x", "--first-pass", str(d / "a"), "--rerun", str(d / "b"), "--fresh", str(d / "c"), "--notes", str(notes), "--doc", str(doc)]
            try:
                final_compare.main()
            finally:
                sys.argv = argv
            md = doc.read_text()
            self.assertIn("| first pass | after setup re-runs | fresh run after the fixes |", md)
            self.assertIn("## Notes", md)
            self.assertIn("| `nav_room_b` | rooms | INFRA (infra_failed) | TIMEOUT | INFRA (sim_hang) |", md)
            self.assertIn("Missions whose result differs between the runs", md)
            self.assertIn("`nav_room_b`", md.split("Missions whose result differs")[1].splitlines()[0])
            self.assertIn("# Fresh run: full report", md)
            self.assertIn("WALL TIMEOUT", final_compare.result({"rec": {"status": "wall_timeout"}, "verdict": {"infra_failure": False, "success": False, "timed_out": True}}))


if __name__ == "__main__":
    unittest.main()
