"""Depth -> 3D geometry for the detector: robust box depth, pixel back-projection, rigid transforms."""
from __future__ import annotations

import numpy as np


def box_depth(depth: np.ndarray, box, inner: float = 0.5, pct: float = 25.0, rng=(0.1, 5.0), min_valid: float = 0.1):
    """Depth of the object inside an xyxy box: low percentile of valid pixels in the central `inner` fraction
    (background bleeding through gaps/edges is farther, so a low percentile ignores it).
    Returns (depth_m, valid_fraction) or (None, valid_fraction) when too few valid pixels."""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner, (y2 - y1) * inner
    xa, xb = int(max(0, np.floor(cx - bw / 2))), int(min(w, np.ceil(cx + bw / 2)))
    ya, yb = int(max(0, np.floor(cy - bh / 2))), int(min(h, np.ceil(cy + bh / 2)))
    if xb <= xa or yb <= ya:
        return None, 0.0
    patch = depth[ya:yb, xa:xb].astype(np.float64).ravel()
    ok = np.isfinite(patch) & (patch >= rng[0]) & (patch <= rng[1])
    frac = float(ok.mean()) if patch.size else 0.0
    if frac < min_valid:
        return None, frac
    return float(np.percentile(patch[ok], pct)), frac


def K_from_camera_info(k9) -> tuple:
    return float(k9[0]), float(k9[4]), float(k9[2]), float(k9[5])  # fx, fy, cx, cy


def backproject(u: float, v: float, z: float, K) -> np.ndarray:
    """Pixel (u, v) at depth z (optical frame: x right, y down, z forward) -> 3D point."""
    fx, fy, cx, cy = K
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def surface_to_centre(p_surface: np.ndarray, radius: float) -> np.ndarray:
    """Move the front-surface point `radius` metres further along the viewing ray (object centre estimate)."""
    n = np.linalg.norm(p_surface)
    return p_surface * (1.0 + radius / n) if n > 1e-6 else p_surface


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def transform_point(p, t, q) -> np.ndarray:
    """p in the child frame -> parent frame, given the child's translation t and quaternion q (x, y, z, w)."""
    return quat_to_R(q) @ np.asarray(p, float) + np.asarray(t, float)
