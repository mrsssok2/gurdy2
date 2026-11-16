"""Frontier library on synthetic grids (no ROS)."""
import numpy as np

from gurdy_core import frontier as fr

RES = 0.05
ORIGIN = (0.0, 0.0)


def room(h=100, w=100):
    """Everything unknown, a known free square 2 m x 2 m in the middle ringed by walls with one open side."""
    g = np.full((h, w), -1, dtype=np.int8)
    g[30:70, 30:70] = 0
    g[30, 30:70] = g[69, 30:70] = 100      # south / north walls
    g[30:70, 30] = 100                     # west wall
    return g                               # east side (x = 69) stays open to unknown


def test_frontier_only_on_open_side():
    g = room()
    cl = fr.clean_frontier_mask(g, 6)
    assert len(cl) == 1
    ix = cl[0][:, 0]
    assert ix.max() == 69 and ix.min() >= 69      # free cells next to the unknown east column


def test_fully_enclosed_has_no_frontier():
    g = room()
    g[30:70, 69] = 100
    assert fr.clean_frontier_mask(g, 1) == []
    assert fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0)) == []


def test_speckle_is_dropped():
    g = room()
    g[30:70, 69] = 100
    g[50, 50] = -1                       # one unknown pixel inside free space
    assert fr.clean_frontier_mask(g, 1)   # it is a frontier ...
    assert fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0), min_cells=6) == []     # ... but too small


def test_plan_picks_reachable_goal_near_frontier():
    g = room()
    fs = fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0))
    assert len(fs) == 1
    gx, gy = fs[0].goal_xy
    assert 3.0 < gx < 3.5 and 1.5 < gy < 3.5                  # near the open east edge (x ~ 3.5)
    assert fs[0].path_m > 1.0


def test_unreachable_frontier_is_skipped():
    g = room()
    g[30:70, 69] = 100                    # close the room
    g[80:90, 10:60] = 0                   # free strip elsewhere, not connected to the robot, with unknown above it
    assert fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0), min_cells=3) == []


def test_blacklist_removes_frontier():
    g = room()
    f = fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0))[0]
    assert fr.plan_frontiers(g, RES, ORIGIN, (2.0, 2.0), blacklist=[f.goal_xy]) == []


def test_nearer_and_larger_frontier_ranks_first():
    g = np.full((100, 200), -1, dtype=np.int8)
    g[40:60, 20:180] = 0                  # corridor, robot at its centre; both ends open (unknown)
    g[39, 20:180] = g[60, 20:180] = 100
    robot = (3.0, 2.5)                    # x index 60: left end (20) is 2 m away, right end (179) 5.95 m
    fs = fr.plan_frontiers(g, RES, ORIGIN, robot)
    assert len(fs) == 2 and fs[0].goal_xy[0] < robot[0] < fs[1].goal_xy[0]
    assert fs[0].path_m < fs[1].path_m


def test_robot_inside_inflation_band_still_plans():
    g = room()
    fs = fr.plan_frontiers(g, RES, ORIGIN, (1.8, 1.8))        # 0.3 m from the west/south walls (30 -> 1.5 m), inside the 0.4 m band
    assert len(fs) == 1


def test_map_stats_interior_fraction():
    g = np.full((20, 20), -1, dtype=np.int8)
    g[:, :10] = 0                          # left half known
    s = fr.map_stats(g, 0.1, (0.0, 0.0), bounds=(0.0, 2.0, 0.0, 2.0))
    assert abs(s["interior_fraction"] - 0.5) < 0.05
    assert abs(s["known_m2"] - 2.0) < 1e-6
