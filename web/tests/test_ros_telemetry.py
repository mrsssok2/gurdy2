"""Real-robot telemetry without ROS: RosTelemetry on a FakePort + the real RosBackend, a fake `extras` (plan / camera)."""
import json
import threading
import unittest
from pathlib import Path

from brain.ros.backend import RosBackend
from brain.ros.fake_port import FakePort
from web.api.ros_telemetry import RosTelemetry, grid_to_rects
from web.api.server import GurdyWebServer
from web.tests.test_api_server import TOKEN, request

PLACES = str(Path(__file__).resolve().parents[2] / "maps" / "flat_lab.places.yaml")


class Port(FakePort):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        W, H = 20, 10
        self.grid = [0] * (W * H)
        for i in range(W):                                    # a wall along the bottom row and the top row
            self.grid[i] = 100
            self.grid[(H - 1) * W + i] = 100
        for j in range(H):                                    # and a 1-cell wide vertical wall at column 10
            self.grid[j * W + 10] = 100

    def occupancy(self):
        return 20, 10, 0.5, -1.0, -2.0, self.grid


class Extras:
    def __init__(self):
        self.pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5)]
        self.cam = None

    def plan(self):
        return list(self.pts)

    def has_camera(self):
        return self.cam is not None

    def camera(self):
        return self.cam


class Host:
    def __init__(self, backend):
        self.agent = type("A", (), {"layer": type("L", (), {"backend": backend})()})()
        self.lock = threading.Lock()


class GridTests(unittest.TestCase):
    def test_runs_merge_into_few_rectangles(self):
        rects = grid_to_rects(4, 3, 1.0, 0.0, 0.0, [100, 100, 0, 0, 100, 100, 0, 0, 100, 100, 0, 0])
        self.assertEqual(rects, [{"x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 3.0}])       # three identical rows -> one 2 x 3 block
        self.assertEqual(grid_to_rects(2, 2, 0.5, 1.0, 2.0, [0, 0, 0, 0]), [])
        l_shape = grid_to_rects(2, 2, 1.0, 0.0, 0.0, [100, 0, 100, 100])
        covered = sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in l_shape)
        self.assertEqual(covered, 3.0)

    def test_unknown_cells_are_not_walls(self):
        self.assertEqual(grid_to_rects(2, 1, 1.0, 0.0, 0.0, [-1, 64]), [])


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.port = Port()
        self.port.battery, self.port.legs = 87.0, {"front_left": "ok", "rear_right": "degraded"}
        self.be = RosBackend(self.port, PLACES)
        self.host, self.extras = Host(self.be), Extras()
        self.tel = RosTelemetry(self.host, self.port, self.extras)

    def test_snapshot_reads_the_port(self):
        s = self.tel.snapshot()
        self.assertTrue(s["connected"])
        self.assertEqual((s["source"], s["battery_pct"], s["battery_simulated"]), ("ros", 87.0, True))
        self.assertEqual(s["legs"]["rear_right"], "degraded")
        self.assertEqual(s["pose"]["place"], "room A")
        self.assertEqual(s["mode"], "idle")
        self.assertEqual(s["path"], [])                       # idle: no planned path even though a stale plan exists

    def test_never_takes_the_agent_lock(self):
        with self.host.lock:                                  # an LLM turn is running
            s = self.tel.snapshot()
        self.assertTrue(s["connected"])
        self.assertFalse(s["stale"])

    def test_no_pose_means_not_connected_not_invented(self):
        self.port.pose = lambda: None
        s = self.tel.snapshot()
        self.assertFalse(s["connected"])
        self.assertIn("transform", s["error"])

    def test_objects_come_from_the_semantic_map(self):
        self.port.t = 100.0
        self.port.set_objects([{"id": 1, "class": "box", "color": "red", "x": 1.0, "y": 1.6, "last_seen": 99.0},
                               {"id": 2, "class": "box", "color": "blue", "x": 1.8, "y": -2.2, "last_seen": 10.0}])
        objs = {o["color"]: o for o in self.tel.snapshot()["objects"]}
        self.assertEqual(objs["red"]["place"], "room A")
        self.assertFalse(objs["red"]["stale"])
        self.assertTrue(objs["blue"]["stale"])

    def test_trail_and_planned_path_while_a_task_runs(self):
        self.tel.snapshot()
        self.port.x = 1.0
        self.assertEqual(len(self.tel.snapshot()["trail"]), 2)
        r = self.be.go_to(place="room B")
        self.assertTrue(r.ok, r)
        self.extras.pts = [(i * 0.1, 0.0) for i in range(200)]
        path = self.tel.snapshot()["path"]
        self.assertTrue(2 <= len(path) <= 42)
        self.assertEqual(path[-1], [19.9, 0.0])               # always ends at the real goal end

    def test_map_has_rooms_places_and_walls_from_the_grid(self):
        m = self.tel.map()
        self.assertEqual(set(m["rooms"]), {"room A", "room B"})
        self.assertEqual(m["places"]["doorway"], {"x": 2.5, "y": 0.5})
        self.assertGreaterEqual(len(m["walls"]), 3)
        self.assertEqual(m["bounds"], {"x0": -2.3, "y0": -3.3, "x1": 6.8, "y1": 3.3})
        self.assertIs(self.tel.map(), m)                      # cached once the real /map exists

    def test_map_without_a_grid_is_not_cached(self):
        self.port.occupancy = lambda: None
        m = self.tel.map()
        self.assertEqual(m["walls"], [])
        self.assertIsNot(self.tel.map(), m)


class CameraRouteTests(unittest.TestCase):
    def test_camera_route_unavailable_then_available(self):
        port = Port()
        be = RosBackend(port, PLACES)
        extras = Extras()
        srv = GurdyWebServer(Host(be), telemetry=RosTelemetry(Host(be), port, extras), token=TOKEN, static_dir=None, port=0).start()
        try:
            st, body = request(srv, "GET", "/api/camera")
            self.assertEqual((st, body), (200, {"available": False}))
            self.assertEqual(request(srv, "GET", "/api/camera", token="wrong")[0], 401)
            extras.cam = {"w": 320, "h": 240, "jpeg": b"\xff\xd8abc", "boxes": [{"name": "box", "score": 0.9, "box": [1, 2, 3, 4]}], "boxes_w": 320, "boxes_h": 240}
            st, body = request(srv, "GET", "/api/camera")
            self.assertTrue(body["available"])
            self.assertEqual(body["boxes"][0]["name"], "box")
            self.assertNotIn("jpeg", body)
            self.assertTrue(body["jpeg_b64"].startswith("/9hh"))
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()


class EstopLatchTests(unittest.TestCase):
    def test_web_estop_latches_the_real_robot_and_release_frees_it(self):
        from brain.agent.graph import Agent
        from brain.agent.llm import KeywordLLM
        from brain.tools import ToolLayer
        from brain.voice.host import AgentHost
        port = Port()
        be = RosBackend(port, PLACES)
        host = AgentHost(Agent(ToolLayer(be), KeywordLLM(be.known_places, smart=True)))
        srv = GurdyWebServer(host, telemetry=RosTelemetry(host, port, Extras()), token=TOKEN, static_dir=None, port=0).start()
        try:
            self.assertFalse(port.estop_active())
            self.assertEqual(request(srv, "POST", "/api/estop", {})[0], 200)
            self.assertTrue(port.estop_active())
            self.assertTrue(request(srv, "GET", "/api/state")[1]["estop"])
            self.assertEqual(request(srv, "POST", "/api/estop/release", {})[1], {"ok": True})
            self.assertFalse(port.estop_active())
            self.assertEqual(request(srv, "POST", "/api/estop/release", {}, token="bad")[0], 401)
        finally:
            srv.close()
