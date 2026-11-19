"""Static (in-browser) demo: the Python session, the langgraph stand-in, the build output.

The key test runs the SAME fixed command list on (a) the repo with the real langgraph and (b) the unzipped build with ONLY its own files and the
langgraph stand-in on the path (what Pyodide sees), and requires identical replies, announcements and robot poses."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

COMMANDS = ["enter room b", "yes", "where are you?", "find the yellow box", "what do you see", "what is near you", "take me to the doorway",
            "go home", "look around", "stop", "go to room A", "find the red box", "turn left 90 degrees", "walk forward 1 meter", "patrol room A and room B",
            "what is your battery", "why did you stop", "pause", "resume", "cancel", "where is the green barrel", "go to the table", "sing me a song",
            "take me to room b and find the green barrel", "come back", "no", "enter room z", "x" * 201, "", "  where are you  "]

DRIVER = r"""
import json, sys
import langgraph.graph as langgraph
from web.static_demo.session import DemoSession
s = DemoSession()
be = s.host.agent.layer.backend
out = []
for t in json.loads(sys.argv[1]):
    r = s.say(t)
    row = {"cmd": t[:30], "reply": r.get("reply", {}).get("text") if "reply" in r else r["error"]}
    be.advance(15.0)                                   # deterministic virtual time instead of wall-clock ticks
    row["announce"] = s.host.step(0.0)
    row["pose"] = [round(be.x, 2), round(be.y, 2)]
    out.append(row)
print(json.dumps({"rows": out, "graph": langgraph.__file__, "mode": s.state()["mode"], "chat": len(s.chat)}))
"""


def have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def run_driver(cwd: Path, pythonpath: list[str]) -> dict:
    r = subprocess.run([sys.executable, "-c", DRIVER, json.dumps(COMMANDS)], cwd=cwd, capture_output=True, text=True, timeout=120,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(pythonpath), "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@unittest.skipUnless(have("yaml") and have("langgraph"), "needs pyyaml + langgraph (the brain image)")
class StaticDemoTests(unittest.TestCase):
    def test_session_basics(self):
        from web.static_demo.session import DemoSession
        s = DemoSession()
        h = s.hello()
        self.assertEqual((h["demo"], h["voice"], h["map"]["map_id"], sorted(h["map"]["rooms"])), (True, False, "flat_lab", ["room A", "room B"]))
        self.assertEqual(s.say("enter room b")["reply"]["text"], "That is outside the area I have mapped. Shall I go?")
        self.assertIn("Going to room B", s.say("yes")["reply"]["text"])
        s.host.agent.layer.backend.advance(40)
        t = s.tick()
        self.assertGreater(t["state"]["pose"]["x"], 1.0)
        self.assertEqual(t["state"]["demo"], True)
        self.assertTrue(any(m["role"] == "robot" for m in s.chat_since(0)))
        self.assertEqual([m["seq"] for m in s.chat_since(0)], sorted(m["seq"] for m in s.chat_since(0)))
        self.assertEqual(s.say("x" * 201)["error"][:6], "'text'")
        self.assertEqual(s.say("   ")["error"][:6], "'text'")
        self.assertIn("Emergency stop", s.estop()["item"]["text"])

    def test_sessions_are_independent(self):
        from web.static_demo.session import DemoSession
        a, b = DemoSession(), DemoSession()
        a.say("enter room b"); a.say("yes")
        a.host.agent.layer.backend.advance(60)
        self.assertGreater(a.state()["pose"]["x"], 1.0)
        self.assertEqual((b.state()["pose"]["x"], b.chat_since(0)), (0.0, []))

    def test_json_entry_point_used_by_the_worker(self):
        from web.static_demo import session
        session.call("reset")
        out = json.loads(session.call("say", json.dumps(["where are you?"])))
        self.assertIn("room A", out["reply"]["text"])
        self.assertEqual(json.loads(session.call("hello"))["type"], "hello")

    def test_built_site_runs_with_the_standin_and_gives_the_same_answers_as_the_real_langgraph(self):
        import build_static_demo
        with tempfile.TemporaryDirectory() as d:
            site = build_static_demo.build(Path(d) / "site")
            for f in ("index.html", "static_backend.js", "demo_worker.js", "demo_config.js", "gurdy_demo.zip", "app.js", "lib.js", "style.css", ".nojekyll"):
                self.assertTrue((site / f).exists(), f)
            self.assertIn('<script src="static_backend.js"></script>\n<script type="module" src="app.js">', (site / "index.html").read_text())
            self.assertFalse((site / "sw.js").exists())
            root = Path(d) / "unzipped"
            zipfile.ZipFile(site / "gurdy_demo.zip").extractall(root)
            names = {p.name for p in root.rglob("*.py")}
            self.assertFalse({"server.py", "ros_telemetry.py", "ollama_llm.py", "stt.py", "audio.py"} & names)       # no server, ROS, LLM client, audio code
            standin = run_driver(root, [str(root / "pylib"), str(root), str(root / "gurdy_core")])
            self.assertTrue(standin["graph"].startswith(str(root / "pylib")), standin["graph"])                      # really the stand-in
            real = run_driver(ROOT, [str(ROOT), str(ROOT / "gurdy_core")])
            self.assertIn("site-packages", real["graph"])                                                            # really the real langgraph
            self.assertEqual(len(real["rows"]), len(COMMANDS))
            for a, b in zip(real["rows"], standin["rows"]):
                self.assertEqual(a, b)
            self.assertEqual((real["mode"], real["chat"]), (standin["mode"], standin["chat"]))
            replies = [r["reply"] for r in real["rows"]]
            self.assertFalse(any("went wrong" in (x or "") for x in replies), replies)
            self.assertGreaterEqual(sum(1 for x in replies if x and "did not understand" not in x), 20)              # the planner understands most of the list

    def test_standin_graph_semantics(self):
        sys.path.insert(0, str(ROOT / "web/static_demo/pylib"))
        try:
            import importlib
            import types
            spec = importlib.util.spec_from_file_location("standin_graph", ROOT / "web/static_demo/pylib/langgraph/graph.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        finally:
            sys.path.pop(0)
        g = m.StateGraph(dict)
        g.add_node("a", lambda s: {"n": s["n"] + 1, "next": "b" if s["n"] < 2 else "end"})
        g.add_node("b", lambda s: {"trace": s.get("trace", []) + ["b"]})
        g.add_node("c", lambda s: None)
        g.set_entry_point("a")
        g.add_conditional_edges("a", lambda s: s["next"], {"b": "b", "end": "c"})
        g.add_edge("b", "a")
        g.add_edge("c", m.END)
        out = g.compile().invoke({"n": 0})
        self.assertEqual((out["n"], out["trace"]), (3, ["b", "b"]))
        loop = m.StateGraph(dict)
        loop.add_node("x", lambda s: {})
        loop.set_entry_point("x")
        loop.add_edge("x", "x")
        with self.assertRaises(RuntimeError):
            loop.compile().invoke({})


class VoiceImportIsLightTests(unittest.TestCase):
    def test_host_import_does_not_pull_in_numpy_or_audio(self):
        r = subprocess.run([sys.executable, "-c", "import sys; import brain.voice.host; print('numpy' in sys.modules, 'brain.voice.audio' in sys.modules)"],
                           cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{ROOT}:{ROOT}/gurdy_core"})
        self.assertEqual(r.stdout.strip(), "False False", r.stderr[-400:])

    def test_lazy_voice_exports_still_work(self):
        r = subprocess.run([sys.executable, "-c", "from brain.voice import VoiceConfig, FRAME_S; print(VoiceConfig().mode, FRAME_S)"],
                           cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{ROOT}:{ROOT}/gurdy_core"})
        self.assertEqual(r.stdout.split()[0], "vad", r.stderr[-400:])


if __name__ == "__main__":
    unittest.main()
