"""Image augmentation applied to rendered RGB before saving (labels unaffected): exposure, noise, motion blur."""
from __future__ import annotations

import numpy as np


def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Average `length` shifted copies along a direction (cheap line-kernel blur, no OpenCV)."""
    if length <= 1:
        return img
    acc = np.zeros_like(img, np.float32)
    dx, dy = np.cos(angle), np.sin(angle)
    for t in np.linspace(-0.5, 0.5, length):
        sx, sy = int(round(t * length * dx)), int(round(t * length * dy))
        acc += np.roll(np.roll(img.astype(np.float32), sy, 0), sx, 1)
    return (acc / length).astype(np.uint8)


def augment_rgb(img: np.ndarray, rng, cfg: dict | None = None) -> np.ndarray:
    c = {"noise_sigma": (0.0, 8.0), "blur_prob": 0.35, "blur_len": (3, 9), "gain": (0.75, 1.25), "bias": (-15, 15)}
    c.update(cfg or {})
    out = img.astype(np.float32) * rng.uniform(*c["gain"]) + rng.uniform(*c["bias"])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < c["blur_prob"]:
        out = motion_blur(out, int(rng.integers(c["blur_len"][0], c["blur_len"][1] + 1)), float(rng.uniform(0, np.pi)))
    sigma = rng.uniform(*c["noise_sigma"])
    if sigma > 0.5:
        out = np.clip(out.astype(np.float32) + rng.normal(0, sigma, out.shape), 0, 255).astype(np.uint8)
    return out
