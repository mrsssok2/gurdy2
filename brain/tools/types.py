"""Shared result / error / object types of the tool layer (no ROS, no third-party packages)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Fixed error code list (docs/phase5_design.md section 2). Never add ad-hoc strings.
ERROR_CODES = (
    "NOT_SEEN", "AMBIGUOUS", "UNKNOWN_PLACE", "NO_PATH", "BLOCKED", "IN_NO_GO_ZONE",
    "NEEDS_CONFIRMATION", "REFUSED", "BUSY", "TIMEOUT", "ESTOP", "UNAVAILABLE",
    "INVALID_ARGS",   # schema / validation failure raised by the tool layer itself (before any backend call)
    "UNKNOWN_TOOL",
)
STOPPED_REASONS = ("reached", "done", "blocked", "no_path", "cancelled", "timeout", "estop", "fell")
BEARING_WORDS = ("ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right")


@dataclass
class Result:
    """Every tool returns {ok, data, error}. `error` = {code, message, ...extra} (AMBIGUOUS carries candidates)."""
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}

    @property
    def code(self) -> str | None:
        return self.error["code"] if self.error else None


def ok(**data: Any) -> Result:
    return Result(True, data, None)


def fail(code: str, message: str, **extra: Any) -> Result:
    assert code in ERROR_CODES, f"unknown error code {code}"
    return Result(False, {}, {"code": code, "message": message, **extra})


def bearing_word(bearing_deg: float) -> str:
    """8-way word for a bearing relative to the robot heading (+ = left / counter-clockwise).

    Computed here, never by the LLM (design 1.5, rule 3). Sector width 45 deg, 'ahead' = +-22.5 deg.
    """
    b = (bearing_deg + 180.0) % 360.0 - 180.0
    idx = int(((b + 22.5) % 360.0) // 45.0)          # 0 ahead, 1 ahead-left, 2 left, ...
    return BEARING_WORDS[idx]
