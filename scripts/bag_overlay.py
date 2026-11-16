#!/usr/bin/env python3
"""Dead-reckoned scan maps from an exported bag (rl image: numpy scipy matplotlib pyyaml). Every scan is drawn into the world
with ONE pose source, no SLAM: this separates "the odometry/TF is wrong" from "slam_toolbox went wrong".
    python3 scripts/bag_overlay.py run.npz OUT_PREFIX
Pose sources: gt (/odom/ground_truth), ekf_tf (TF odom->base_link as slam_toolbox saw it), ekf (/odometry/filtered),
legs (/odom/legs), slam (map_odom TF composed with ekf_tf = slam_toolbox's corrected pose). Scores like map_quality.py.
Also prints the pose error of each source against gt at the scan times (xy, yaw) and the TF-vs-topic stamp lag."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = 0.05


def interp_pose(series, t):
    """(N,4) [t,x,y,yaw] -> poses at times t (linear, yaw unwrapped); NaN outside the series."""
    ts = series[:, 0]
    yaw = np.unwrap(series[:, 3])
    out = np.stack([np.interp(t, ts, series[:, 1]), np.interp(t, ts, series[:, 2]), np.interp(t, ts, yaw)], axis=1)
    out[(t < ts[0]) | (t > ts[-1])] = np.nan
    return out


def compose(a, b):
    """pose a (map->odom) o pose b (odom->base): both (N,3)."""
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    return np.stack([a[:, 0] + c * b[:, 0] - s * b[:, 1], a[:, 1] + s * b[:, 0] + c * b[:, 1], a[:, 2] + b[:, 2]], axis=1)


def scans_to_world(ranges, angle_min, inc, poses, rmin=0.2, rmax=6.0):
    """ranges (N,B), poses (N,3) -> (M,2) hit points in the world frame (lidar at the base_link origin)."""
    ang = angle_min + inc * np.arange(ranges.shape[1])
    pts = []
    for r, (x, y, th) in zip(ranges, poses):
        if np.isnan(x):
            continue
        ok = np.isfinite(r) & (r > rmin) & (r < rmax)
        a = ang[ok] + th
        pts.append(np.stack([x + r[ok] * np.cos(a), y + r[ok] * np.sin(a)], axis=1))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def occupied_cells(pts, min_hits=3):
    """Hit points -> occupied cell centres (cells with >= min_hits hits), like an occupancy grid."""
    ij = np.floor(pts / RES).astype(int)
    u, n = np.unique(ij, axis=0, return_counts=True)
    return (u[n >= min_hits] + 0.5) * RES


def main():
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from map_quality import world_cells
    d = np.load(sys.argv[1])
    t = d["scan_t"]
    src = {"gt": interp_pose(d["gt"], t)}
    for k in ("ekf_tf", "ekf", "legs"):
        if len(d[k]):
            src[k] = interp_pose(d[k], t)
    if len(d["map_odom"]) and "ekf_tf" in src:
        src["slam"] = compose(interp_pose(d["map_odom"], t), src["ekf_tf"])
    world = world_cells()
    tree_w = cKDTree(world)
    fig, axes = plt.subplots(1, len(src), figsize=(4.2 * len(src), 4), squeeze=False)
    for ax, (k, p) in zip(axes[0], src.items()):
        pts = scans_to_world(d["scan_ranges"], float(d["scan_angle_min"]), float(d["scan_angle_increment"]), p)
        cells = occupied_cells(pts)
        dist = tree_w.query(cells)[0] if len(cells) else np.array([np.nan])
        err = p[:, :2] - src["gt"][:, :2]
        yerr = np.degrees((p[:, 2] - src["gt"][:, 2] + np.pi) % (2 * np.pi) - np.pi)
        ok = ~np.isnan(err[:, 0])
        print(f"{k:7s} cells {len(cells):5d}  precision mean {np.nanmean(dist):.3f} m p95 {np.nanpercentile(dist, 95):.3f} m | "
              f"pose err vs gt: xy mean {np.nanmean(np.hypot(*err[ok].T)):.3f} max {np.nanmax(np.hypot(*err[ok].T)):.3f} m, "
              f"yaw mean {np.nanmean(np.abs(yerr[ok])):.1f} deg max {np.nanmax(np.abs(yerr[ok])):.1f} deg")
        ax.scatter(cells[:, 0], cells[:, 1], s=1, c="k"); ax.scatter(world[::4, 0], world[::4, 1], s=0.5, c="tab:red", alpha=0.4)
        ax.set_aspect("equal"); ax.set_title(k)
    if len(d["ekf_tf"]) and len(d["ekf"]):          # is the TF the same pose as the topic, and how old is it?
        lag = d["ekf_tf"][:, 0].max() - d["ekf"][:, 0].max()
        print(f"TF vs /odometry/filtered: last stamps differ by {lag:+.3f} s; TF samples {len(d['ekf_tf'])}, topic samples {len(d['ekf'])}")
    fig.tight_layout(); fig.savefig(sys.argv[2] + "_overlay.png", dpi=100)


if __name__ == "__main__":
    main()
