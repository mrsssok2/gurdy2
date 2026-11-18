"""cmd_vel e-stop gate logic (no ROS): while the e-stop is latched every velocity command becomes zero.

The gate sits between everything that speaks /cmd_vel (Nav2, teleop, the agent) and the controller (`/cmd_vel_safe`), so a
latched /estop stops the robot even if a producer keeps sending. It is independent of the controller's own /estop handling.
"""
from __future__ import annotations

ZERO = (0.0, 0.0, 0.0)


class EstopGate:
    def __init__(self) -> None:
        self.estop = False
        self.blocked = 0          # non-zero commands replaced by zero since the last release
        self.passed = 0

    def set_estop(self, active: bool) -> int:
        """-> number of non-zero commands blocked during the stop (returned when it is released, else 0)."""
        n = 0
        if self.estop and not active:
            n, self.blocked = self.blocked, 0
        self.estop = bool(active)
        return n

    def filter(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        if self.estop:
            if vx or vy or wz:
                self.blocked += 1
            return ZERO
        self.passed += 1
        return (vx, vy, wz)
