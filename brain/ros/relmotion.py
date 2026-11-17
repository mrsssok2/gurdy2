"""Closed-loop short walk / turn on pose feedback (the 'NEW walk/turn executor' of design 1.3), pure Python.

Commands are meant for `cmd_vel_smoothed`, so the min-speed lift (0.09 m/s / 0.23 rad/s) and the collision monitor
still apply. Speeds are chosen above those minimums: walking below them would be lifted anyway.
"""
from __future__ import annotations

import math

V_WALK, W_TURN, W_SLOW = 0.10, 0.30, 0.23     # m/s, rad/s (cruise, slow-down floor = the min-speed lift)
POS_TOL, ANG_TOL = 0.03, math.radians(3.0)
_DIR = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RelativeMotion:
    """kind 'walk': amount = metres along `direction` in the start frame; kind 'turn': amount = degrees."""

    def __init__(self, kind: str, direction: str, amount: float, pose0: tuple[float, float, float]):
        self.kind, self.direction, self.amount = kind, direction, amount
        self.x0, self.y0, self.yaw0 = pose0
        self.last_yaw = pose0[2]
        self.turned = 0.0                    # signed, counter-clockwise positive, unwrapped
        self.moved = 0.0
        self.done = False

    @property
    def estimate(self) -> dict:
        if self.kind == "walk":
            return {"distance_m": round(self.amount, 2), "duration_s": round(self.amount / V_WALK, 1)}
        return {"distance_m": 0.0, "duration_s": round(math.radians(self.amount) / W_TURN, 1)}

    def command(self, pose: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, yaw = pose
        self.turned += wrap(yaw - self.last_yaw)
        self.last_yaw = yaw
        if self.kind == "turn":
            sign = 1.0 if self.direction == "left" else -1.0
            remaining = math.radians(self.amount) - sign * self.turned
            if remaining <= ANG_TOL:
                self.done = True
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, sign * (W_TURN if remaining > 0.3 else W_SLOW)
        dx, dy = _DIR[self.direction]
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        wx, wy = dx * c - dy * s, dx * s + dy * c             # unit direction in the map frame
        self.moved = (x - self.x0) * wx + (y - self.y0) * wy
        if self.amount - self.moved <= POS_TOL:
            self.done = True
            return 0.0, 0.0, 0.0
        cy, sy = math.cos(yaw), math.sin(yaw)                 # same map direction, expressed in the current body frame
        vx, vy = V_WALK * (wx * cy + wy * sy), V_WALK * (-wx * sy + wy * cy)
        return vx, vy, max(-W_TURN, min(W_TURN, 1.0 * wrap(self.yaw0 - yaw)))   # hold the start heading

    def result(self) -> dict:
        if self.kind == "walk":
            return {"moved_m": round(max(self.moved, 0.0), 2)}
        return {"turned_deg": round(abs(math.degrees(self.turned)), 1)}
