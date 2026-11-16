"""Colour naming for a detected object crop (for 'laal dabba' = red box). English + Hinglish names."""
from __future__ import annotations

import colorsys

import numpy as np

HINGLISH = {"red": "laal", "orange": "narangi", "yellow": "peela", "green": "hara", "blue": "neela",
            "purple": "baingani", "pink": "gulabi", "brown": "bhura", "black": "kaala", "white": "safed", "gray": "dhusar"}


def color_name(rgb_crop: np.ndarray) -> str:
    """Name of the dominant colour of a crop (median RGB -> HSV rules)."""
    px = np.asarray(rgb_crop, np.float64).reshape(-1, 3)
    if not len(px):
        return "gray"
    r, g, b = (np.median(px, 0) / 255.0).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.18:
        return "white" if v > 0.75 else "gray"
    h *= 360
    if h < 15 or h >= 340:
        return "red"
    if h < 40:
        return "brown" if v < 0.6 else "orange"
    if h < 70:
        return "yellow"
    if h < 170:
        return "green"
    if h < 260:
        return "blue"
    return "purple" if h < 310 else "pink"


def resolve_color(text: str) -> str | None:
    words = text.lower().split()
    for en, hi in HINGLISH.items():
        if en in words or hi in words:
            return en
    return None
