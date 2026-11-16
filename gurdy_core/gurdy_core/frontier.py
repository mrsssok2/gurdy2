"""Frontier detection and selection on an occupancy grid (pure numpy, no ROS).

A frontier cell is a known-free cell with an unknown 4-neighbour: the boundary between mapped free space and
space the robot has not seen. Exploration = repeatedly drive to the best reachable frontier until none is left.

    grid  int8 array [iy, ix], values like nav_msgs/OccupancyGrid: -1 unknown, 0 free, 100 occupied
    cell  (ix, iy);  world = origin + (cell + 0.5) * resolution

Pipeline (`plan_frontiers`):
  1. frontier mask -> 8-connected clusters, tiny clusters dropped (lidar speckle);
  2. reachability: Dijkstra over free cells that keep `clearance_m` from walls, started at the robot. A cluster is
     reachable when a reachable cell lies within `reach_m` of it; that cell (the nearest one) is the goal;
  3. blacklisted goals are skipped; score = path length - size_weight * min(size, size_cap).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

UNKNOWN, FREE_MAX, OCC_MIN = -1, 49, 50
SQRT2 = 2 ** 0.5
_NEIGH8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


@dataclass
class Frontier:
    cells: np.ndarray            # (n, 2) int, (ix, iy)
    size_m: float                # cluster length estimate = cells * resolution
    centroid_xy: tuple[float, float]
    goal_cell: tuple[int, int]   # reachable cell to drive to
    goal_xy: tuple[float, float]
    path_m: float                # path length robot -> goal over the clearance-inflated free space
    score: float = 0.0


def free_mask(grid: np.ndarray) -> np.ndarray:
    return (grid >= 0) & (grid <= FREE_MAX)


def frontier_mask(grid: np.ndarray) -> np.ndarray:
    """Free cells with at least one unknown 4-neighbour (map border does not count as unknown)."""
    unk = grid == UNKNOWN
    nb = np.zeros_like(unk)
    nb[1:, :] |= unk[:-1, :]
    nb[:-1, :] |= unk[1:, :]
    nb[:, 1:] |= unk[:, :-1]
    nb[:, :-1] |= unk[:, 1:]
    return free_mask(grid) & nb


def disk_offsets(radius_cells: float) -> list[tuple[int, int]]:
    r = int(np.ceil(radius_cells))
    return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= radius_cells ** 2]


def dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dx, dy in disk_offsets(radius_cells):
        ys, ye = max(0, dy), min(h, h + dy)
        xs, xe = max(0, dx), min(w, w + dx)
        out[ys:ye, xs:xe] |= mask[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def cluster_cells(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a boolean mask as arrays of (ix, iy)."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    clusters = []
    for iy, ix in zip(*np.nonzero(mask)):
        if seen[iy, ix]:
            continue
        stack, cells = [(ix, iy)], []
        seen[iy, ix] = True
        while stack:
            cx, cy = stack.pop()
            cells.append((cx, cy))
            for dx, dy, _ in _NEIGH8:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        clusters.append(np.array(cells, dtype=int))
    return clusters


def path_lengths(passable: np.ndarray, start: tuple[int, int], res: float) -> np.ndarray:
    """Dijkstra (8-connected) path length in metres over `passable` from `start`; inf where unreachable."""
    h, w = passable.shape
    dist = np.full((h, w), np.inf)
    sx, sy = start
    dist[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy, c in _NEIGH8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and passable[ny, nx]:
                nd = d + c
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, nx, ny))
    return dist * res


def world_to_cell(xy, res: float, origin) -> tuple[int, int]:
    return int(np.floor((xy[0] - origin[0]) / res)), int(np.floor((xy[1] - origin[1]) / res))


def cell_to_world(cell, res: float, origin) -> tuple[float, float]:
    return origin[0] + (cell[0] + 0.5) * res, origin[1] + (cell[1] + 0.5) * res


def clean_frontier_mask(grid: np.ndarray, min_cells: int) -> list[np.ndarray]:
    return [c for c in cluster_cells(frontier_mask(grid)) if len(c) >= min_cells]


def plan_frontiers(grid: np.ndarray, res: float, origin, robot_xy, *, min_cells: int = 6, clearance_m: float = 0.40,
                   reach_m: float = 0.4, blacklist=(), blacklist_radius_m: float = 0.6, size_weight: float = 0.5,
                   size_cap_m: float = 2.0) -> list[Frontier]:
    """All usable frontiers, best (lowest score) first. Empty list = nothing left to explore (or nothing reachable)."""
    rc = world_to_cell(robot_xy, res, origin)
    h, w = grid.shape
    if not (0 <= rc[0] < w and 0 <= rc[1] < h):
        return []
    clusters = clean_frontier_mask(grid, min_cells)
    if not clusters:
        return []
    occ = grid >= OCC_MIN
    blocked = dilate(occ, clearance_m / res)
    passable = free_mask(grid) & ~blocked
    # the robot may stand inside the inflated band (spawn near a wall): free a small disk around it
    for dx, dy in disk_offsets(clearance_m / res):
        x, y = rc[0] + dx, rc[1] + dy
        if 0 <= x < w and 0 <= y < h and free_mask(grid)[y, x]:
            passable[y, x] = True
    if not passable[rc[1], rc[0]]:
        return []
    dist = path_lengths(passable, rc, res)
    out = []
    r_cells = reach_m / res
    offs = disk_offsets(r_cells)
    for cells in clusters:
        cand_best, cand_d = None, np.inf
        for cx, cy in cells[:: max(1, len(cells) // 40)]:          # subsample big clusters
            for dx, dy in offs:
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h and dist[y, x] < cand_d:
                    cand_best, cand_d = (x, y), dist[y, x]
        if cand_best is None or not np.isfinite(cand_d):
            continue
        goal_xy = cell_to_world(cand_best, res, origin)
        if any(np.hypot(goal_xy[0] - bx, goal_xy[1] - by) < blacklist_radius_m for bx, by in blacklist):
            continue
        size_m = len(cells) * res
        cen = cells.mean(axis=0)
        f = Frontier(cells=cells, size_m=size_m, centroid_xy=cell_to_world(cen, res, origin), goal_cell=cand_best,
                     goal_xy=goal_xy, path_m=float(cand_d))
        f.score = f.path_m - size_weight * min(size_m, size_cap_m)
        out.append(f)
    return sorted(out, key=lambda f: f.score)


def map_stats(grid: np.ndarray, res: float, origin, bounds=None) -> dict:
    """Known area (free + occupied cells) in m^2; with bounds=(xmin, xmax, ymin, ymax) also inside that rectangle and
    as a fraction of it (the unmapped part of the rectangle counts as unknown)."""
    known = grid != UNKNOWN
    s = {"known_m2": float(known.sum() * res * res), "free_m2": float(free_mask(grid).sum() * res * res),
         "occupied_cells": int((grid >= OCC_MIN).sum())}
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        h, w = grid.shape
        xs = origin[0] + (np.arange(w) + 0.5) * res
        ys = origin[1] + (np.arange(h) + 0.5) * res
        inside = ((ys[:, None] >= ymin) & (ys[:, None] <= ymax)) & ((xs[None, :] >= xmin) & (xs[None, :] <= xmax))
        total = (xmax - xmin) * (ymax - ymin)
        s["interior_known_m2"] = float((known & inside).sum() * res * res)
        s["interior_fraction"] = min(1.0, s["interior_known_m2"] / total)
    return s
