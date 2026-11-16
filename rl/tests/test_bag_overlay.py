"""scripts/bag_overlay.py geometry: a robot looking at a wall maps it where the wall is (synthetic, no ROS)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from bag_overlay import compose, interp_pose, occupied_cells, scans_to_world  # noqa: E402


def test_scan_of_a_wall_lands_on_the_wall_for_any_pose():
    ang = -np.pi + np.arange(360) * (2 * np.pi / 360)
    poses = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 1.0], [-0.3, 0.1, -2.5]])
    ranges = []
    for x, y, th in poses:                                  # wall x = 3: range along each ray (inf if not hit)
        d = np.cos(ang + th)
        r = np.where(d > 1e-3, (3.0 - x) / np.where(d > 1e-3, d, 1), np.inf)
        ranges.append(np.where(r < 6.0, r, np.inf))
    pts = scans_to_world(np.array(ranges), -np.pi, 2 * np.pi / 360, poses)
    assert len(pts) > 100 and np.allclose(pts[:, 0], 3.0, atol=1e-6)


def test_compose_and_interp():
    a = np.array([[1.0, 2.0, np.pi / 2]])
    b = np.array([[1.0, 0.0, 0.1]])
    assert np.allclose(compose(a, b), [[1.0, 3.0, np.pi / 2 + 0.1]])
    s = np.array([[0.0, 0, 0, 3.1], [1.0, 2, 0, -3.1]])      # yaw wraps: no jump through 0
    p = interp_pose(s, np.array([0.5, 2.0]))
    assert abs(abs(p[0, 2]) - np.pi) < 0.1 and np.isnan(p[1, 0])


def test_occupied_cells_need_several_hits():
    pts = np.array([[0.01, 0.01]] * 3 + [[1.0, 1.0]])
    assert len(occupied_cells(pts, min_hits=3)) == 1
