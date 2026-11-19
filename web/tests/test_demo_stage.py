"""The Hugging Face (static Space) folder: same site as GitHub Pages + a Space README; nothing server-side, secret or heavy inside."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


class StagedSpaceTests(unittest.TestCase):
    def test_static_space_folder(self):
        import demo_space
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "space"
            demo_space.stage(out)
            readme = (out / "README.md").read_text()
            self.assertEqual(readme.split("\n")[0], "---")
            self.assertIn("sdk: static", readme)
            self.assertIn("app_file: index.html", readme)
            for f in ("index.html", "static_backend.js", "demo_worker.js", "demo_config.js", "gurdy_demo.zip", "app.js"):
                self.assertTrue((out / f).exists(), f)
            names = {p.name for p in out.rglob("*") if p.is_file()}
            self.assertFalse({"Dockerfile", "requirements.txt", "sw.js", ".nojekyll", "memory.db", ".env"} & names)
            self.assertTrue(all(p.stat().st_size < 1_000_000 for p in out.rglob("*") if p.is_file()))    # the site itself is small; Pyodide comes from the CDN
            self.assertIn("cdn.jsdelivr.net/pyodide/v", (out / "demo_config.js").read_text())

    def test_deploy_needs_the_token_in_the_environment(self):
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts/demo_space.py"), "deploy"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HF_TOKEN is not set", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
