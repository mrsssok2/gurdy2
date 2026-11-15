"""Terrain for the Gurdy MuJoCo env (step 2.8): heightfield generation + per-env curriculum.

One heightfield per env (size_m x size_m, `res` m cells, robot spawned at the centre), regenerated
on every reset for the episode's (type, level). The flat type uses the original floor plane (the
heightfield's collisions are switched off), so flat episodes are identical to steps 2.4-2.7.

Types and difficulty (level 0 .. levels-1 maps linearly from `easy` to `hard`):
  rough  : smooth random bumps, peak-to-peak amplitude [m] (bumps on a `rough_cell` m grid,
           bilinearly interpolated)
  slopes : planar incline, angle [deg], random uphill direction
  steps  : random blocks (side 0.25-0.40 m) at height 0 or h: every block edge is a step of h [m]
ROADMAP targets (rough 0.5-2 cm, slopes 5-12 deg, steps 1.5-3 cm) lie inside the default ranges.
The hardest levels go beyond them (rough 4 cm, slopes 15 deg, steps 5 cm): a baseline eval showed
rough <= 2.5 cm is easy for both the tripod and the flat policy (100 % success), and the tripod
starts failing on steps >= 2.5 cm, so the extra range is where RL can show an advantage.

Curriculum (per env, per type; legged_gym style): after each episode on a type
  fall                                   -> level - 1
  no fall, mean tracking error < promote -> level + 1   (top level: jump to a random level)
  no fall, error > demote                -> level - 1
Initial levels are uniform in [0, init_max_level].
"""
from __future__ import annotations

import numpy as np

TYPES = ("flat", "rough", "slopes", "steps")

DEFAULTS = {
    "enabled": False,
    "size_m": 10.0,
    "res": 0.02,
    "levels": 10,
    "init_max_level": 2,
    "type_probs": {"flat": 0.10, "rough": 0.30, "slopes": 0.25, "steps": 0.35},
    "easy": {"rough": 0.002, "slopes": 2.0, "steps": 0.005},
    "hard": {"rough": 0.04, "slopes": 15.0, "steps": 0.05},
    "rough_cell": 0.10,
    "step_block": [0.25, 0.40],
    "promote_error": 0.35,
    "demote_error": 0.60,
    "max_elevation": 2.6,        # m; total height range of the hfield (slope 15 deg over 10 m = 2.68 m peak-to-peak)
}


def merged(cfg: dict | None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    for k in ("type_probs", "easy", "hard"):
        c[k] = {**DEFAULTS[k], **(cfg or {}).get(k, {})}
    return c


def difficulty_value(cfg: dict, ttype: str, level: int) -> float:
    if ttype == "flat":
        return 0.0
    f = level / max(cfg["levels"] - 1, 1)
    return cfg["easy"][ttype] + f * (cfg["hard"][ttype] - cfg["easy"][ttype])


def _interp_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Linear interpolation matrix (n_out x n_in) from n_in to n_out evenly spaced samples."""
    x = np.linspace(0, n_in - 1, n_out)
    i0 = np.clip(np.floor(x).astype(int), 0, n_in - 2)
    w = x - i0
    m = np.zeros((n_out, n_in))
    m[np.arange(n_out), i0] = 1 - w
    m[np.arange(n_out), i0 + 1] = w
    return m


class TerrainGenerator:
    """Height arrays (n x n, row = y, col = x, metres, world frame, 0 at the spawn point)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = merged(cfg)
        c = self.cfg
        self.n = int(round(c["size_m"] / c["res"])) + 1
        self.half = c["size_m"] / 2
        self.axis = np.linspace(-self.half, self.half, self.n)
        nc = int(round(c["size_m"] / c["rough_cell"])) + 1
        self._rough_w = _interp_matrix(self.n, nc)
        self._nc = nc

    def generate(self, ttype: str, value: float, rng: np.random.Generator) -> np.ndarray:
        n, ax = self.n, self.axis
        if ttype == "flat" or value <= 0:
            return np.zeros((n, n))
        if ttype == "rough":
            coarse = rng.uniform(-value / 2, value / 2, (self._nc, self._nc))
            h = self._rough_w @ coarse @ self._rough_w.T
            return h - self.height_at(h, 0.0, 0.0)
        if ttype == "slopes":
            phi = rng.uniform(0, 2 * np.pi)
            x, y = np.meshgrid(ax, ax)                    # x varies along columns
            return np.tan(np.radians(value)) * (x * np.cos(phi) + y * np.sin(phi))
        if ttype == "steps":
            b = rng.uniform(*self.cfg["step_block"])
            off = rng.uniform(0, b, 2)
            ix = np.floor((ax + off[0]) / b).astype(int)
            iy = np.floor((ax + off[1]) / b).astype(int)
            ix -= ix.min()
            iy -= iy.min()
            blocks = rng.integers(0, 2, (iy.max() + 1, ix.max() + 1)) * value
            return blocks[np.ix_(iy, ix)]
        raise ValueError(f"unknown terrain type {ttype}")

    def height_at(self, h: np.ndarray, x: float, y: float) -> float:
        """Bilinear height at world (x, y) (clamped to the map)."""
        r = self.cfg["res"]
        fx = np.clip((x + self.half) / r, 0, self.n - 1.000001)
        fy = np.clip((y + self.half) / r, 0, self.n - 1.000001)
        i, j = int(fy), int(fx)
        wy, wx = fy - i, fx - j
        return float((1 - wy) * ((1 - wx) * h[i, j] + wx * h[i, j + 1])
                     + wy * ((1 - wx) * h[i + 1, j] + wx * h[i + 1, j + 1]))

    def max_near(self, h: np.ndarray, x: float, y: float, radius: float) -> float:
        r = self.cfg["res"]
        lo = lambda v: int(np.clip((v - radius + self.half) / r, 0, self.n - 1))   # noqa: E731
        hi = lambda v: int(np.clip((v + radius + self.half) / r, 0, self.n - 1)) + 1   # noqa: E731
        return float(h[lo(y):hi(y), lo(x):hi(x)].max())


class TerrainCurriculum:
    """Per-env level for each terrain type, updated from episode outcomes."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.levels = {t: int(rng.integers(0, cfg["init_max_level"] + 1)) for t in TYPES if t != "flat"}

    def sample_type(self, rng: np.random.Generator) -> str:
        p = np.array([self.cfg["type_probs"][t] for t in TYPES], float)
        return TYPES[int(rng.choice(len(TYPES), p=p / p.sum()))]

    def level(self, ttype: str) -> int:
        return 0 if ttype == "flat" else self.levels[ttype]

    def update(self, ttype: str, fell: bool, mean_error: float | None, rng: np.random.Generator) -> int:
        """Returns the level change (-1, 0, +1, or a random jump from the top level)."""
        if ttype == "flat":
            return 0
        top = self.cfg["levels"] - 1
        lv = self.levels[ttype]
        if fell or (mean_error is not None and mean_error > self.cfg["demote_error"]):
            new = max(lv - 1, 0)
        elif mean_error is not None and mean_error < self.cfg["promote_error"]:   # None: only zero commands
            new = int(rng.integers(0, top + 1)) if lv >= top else lv + 1
        else:
            new = lv
        self.levels[ttype] = new
        return new - lv
