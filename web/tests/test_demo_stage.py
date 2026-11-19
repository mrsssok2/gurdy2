"""The staged Space folder must be self-contained: run the demo agent from it with ONLY that folder on the path (catches a missing file
before a deploy; this is how `brain.eval.keyword_parser` was found missing)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "scripts"))


def have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


@unittest.skipUnless(have("langgraph") and have("yaml"), "needs langgraph + pyyaml (the brain image)")
class StagedDemoTests(unittest.TestCase):
    def test_staged_folder_runs_the_demo_without_the_repo(self):
        import demo_space
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "space"
            demo_space.stage(out)
            code = (
                "import json, sys; sys.path[:] = [p for p in sys.path if 'gurdy2' not in p]\n"
                "from web.api.demo import make_agent\n"
                "a = make_agent(); b = a.layer.backend\n"
                "r = [a.say(t).reply for t in ('enter room b', 'where are you?', 'find the yellow box', 'go home')]\n"
                "import web.api.server\n"
                "print(json.dumps({'replies': r, 'world': b.world.map_id, 'objects': sorted(o.color for o in b.world.objects), 'file': web.api.demo.__file__}))\n")
            env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{out}:{out}/gurdy_core", "PYTHONDONTWRITEBYTECODE": "1"}
            r = subprocess.run([sys.executable, "-c", code], cwd=out, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            res = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(res["file"].startswith(str(out)))                 # imported from the staged folder, not the repo
            self.assertEqual(res["world"], "flat_lab")
            self.assertEqual(res["objects"], ["blue", "green", "red", "yellow"])
            self.assertFalse(any("went wrong" in x or "did not understand" in x for x in res["replies"]), res["replies"])

    def test_nothing_ros_cuda_or_secret_is_staged(self):
        import demo_space
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "space"
            demo_space.stage(out)
            names = {p.name for p in out.rglob("*") if p.is_file()}
            self.assertFalse({"ros_extras.py", "ros_telemetry.py", "memory.db", ".env"} & names)
            self.assertEqual((out / "README.md").read_text().split("\n")[0], "---")
            self.assertIn("app_port: 7860", (out / "README.md").read_text())
            self.assertIn("sdk: docker", (out / "README.md").read_text())
            code_lines = [x for x in (out / "Dockerfile").read_text().lower().splitlines() if not x.lstrip().startswith("#")]
            self.assertTrue(any(x.startswith("from python:3.12-slim") for x in code_lines))
            for bad in ("cuda", "nvidia", "ros2", "ollama", "whisper", "torch"):
                self.assertFalse(any(bad in x for x in code_lines), bad)
            reqs = (out / "requirements.txt").read_text().lower()
            for bad in ("torch", "faster-whisper", "ctranslate2", "onnx", "opencv", "ultralytics"):
                self.assertNotIn(bad, reqs)


if __name__ == "__main__":
    unittest.main()
