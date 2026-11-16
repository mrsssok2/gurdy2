"""Procedural floor/wall textures (numpy + PIL): checker, tiles, stripes, noise, wood-like. Deterministic by seed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

KINDS = ("checker", "tiles", "stripes", "noise", "wood")


def make_texture(kind: str, seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.9, 3)
    alt = np.clip(base * rng.uniform(0.5, 1.4, 3), 0, 1)
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "checker":
        n = int(rng.integers(4, 16))
        m = ((xx * n // size) + (yy * n // size)) % 2
        img = np.where(m[..., None] == 0, base, alt)
    elif kind == "tiles":
        n = int(rng.integers(3, 8))
        grout = ((xx * n % size) < 4) | ((yy * n % size) < 4)
        img = np.where(grout[..., None], 0.2, base) * (1 + 0.05 * rng.standard_normal((n, n, 1)).repeat(size // n + 1, 0)
                                                        .repeat(size // n + 1, 1)[:size, :size])
    elif kind == "stripes":
        n = int(rng.integers(4, 20))
        img = np.where((((xx * n) // size) % 2 == 0)[..., None], base, alt)
    elif kind == "wood":
        ring = np.sin((yy + 12 * np.sin(xx / 23.0)) / rng.uniform(3, 8))
        img = (base * 0.7 + 0.15 * ring[..., None]) + alt * 0.1
    else:
        img = base + 0.12 * rng.standard_normal((size, size, 1))
    img = img + 0.03 * rng.standard_normal((size, size, 1))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def write_textures(out_dir: str, n: int = 12, seed: int = 0) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        name = f"tex_{i:02d}_{KINDS[i % len(KINDS)]}.png"
        Image.fromarray(make_texture(KINDS[i % len(KINDS)], seed * 1000 + i)).save(out / name)
        names.append(name)
    return names
