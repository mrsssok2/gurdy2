#!/usr/bin/env python3
"""Generate the dashboard's app icons (pure standard library, no external assets): a top-down hexapod in teal on the dark background.
Writes web/app/icons/{icon-192,icon-512,icon-maskable-512,apple-touch-icon}.png. Re-run after changing the drawing; output is deterministic.
Method: signed-distance shapes (capsules + circle), one anti-aliased sample per pixel, PNG written with zlib."""
from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path

BG = (0x0E, 0x11, 0x16)
FG = (0x2D, 0xD4, 0xBF)
OUT = Path(__file__).resolve().parents[1] / "web" / "app" / "icons"


def hexapod_shapes():
    """Unit coordinates, y down, centred on 0, extent about +-1. Body capsule, head circle, six two-segment legs."""
    caps = [((0.0, -0.22), (0.0, 0.30), 0.27)]                                 # body
    for side in (-1, 1):
        for hip_y, knee_dy, foot_dy in ((-0.14, -0.20, -0.52), (0.04, 0.0, 0.0), (0.22, 0.20, 0.52)):
            hip = (side * 0.20, hip_y)
            knee = (side * 0.62, hip_y + knee_dy)
            foot = (side * 0.78, hip_y + foot_dy + (0.12 if foot_dy else 0.30))
            caps.append((hip, knee, 0.075))
            caps.append((knee, foot, 0.062))
    head = ((0.0, -0.50), 0.15)
    return caps, head


def sd_capsule(px, py, a, b, r):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)) - r


def render(size: int, scale: float, rounded: bool) -> bytes:
    """`scale`: unit -> fraction of the icon (0.9 = near full bleed; maskable icons keep the art inside the central safe zone)."""
    caps, (hc, hr) = hexapod_shapes()
    half, px_unit = size / 2, size / 2 * scale          # pixels per unit
    rows = []
    corner = size * 0.22
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            ux, uy = (x + 0.5 - half) / px_unit, (y + 0.5 - half) / px_unit
            d = math.hypot(ux - hc[0], uy - hc[1]) - hr
            for a, b, r in caps:
                if abs(ux) < 1.0 + r and abs(uy) < 1.0 + r:
                    d = min(d, sd_capsule(ux, uy, a, b, r))
            cov = max(0.0, min(1.0, 0.5 - d * px_unit))   # distance in px -> coverage
            rgb = tuple(round(BG[i] + (FG[i] - BG[i]) * cov) for i in range(3))
            alpha = 255
            if rounded:                                  # transparent corners for the plain ("any") icon
                cx, cy = abs(x + 0.5 - half), abs(y + 0.5 - half)
                qx, qy = max(cx - (half - corner), 0), max(cy - (half - corner), 0)
                alpha = round(255 * max(0.0, min(1.0, 0.5 - (math.hypot(qx, qy) - corner))))
            row += bytes((*rgb, alpha))
        rows.append(bytes(row))
    return b"".join(rows)


def png(size: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# name -> (size, art scale, rounded corners). Maskable: full-bleed background, art inside the 80 % safe zone (platforms crop to a circle / squircle).
ICONS = {"icon-192.png": (192, 0.86, True), "icon-512.png": (512, 0.86, True),
         "icon-maskable-512.png": (512, 0.62, False), "apple-touch-icon.png": (180, 0.74, False)}   # iOS applies its own rounding, wants no alpha


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (size, scale, rounded) in ICONS.items():
        (OUT / name).write_bytes(png(size, render(size, scale, rounded)))
        print("wrote", OUT / name)


if __name__ == "__main__":
    sys.exit(main())
