"""Plain dataclasses for the incident graph (design section 3): run, event, incident (+ member events), cause, evidence.
Times in `stamp`/`start`/`end` are SIM/run time in seconds (the run's own clock); `created` fields are wall-clock epoch seconds."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("info", "warning", "critical")          # ordered, low -> high
INCIDENT_STATUSES = ("open", "closed")
ROLES = ("trigger", "member", "context")             # context = explains other events but is no failure (e.g. operator e-stop)
EVIDENCE_KINDS = ("metric", "log", "config_diff", "event")
CAUSE_SOURCES = ("rule", "agent", "label")           # label = ground-truth fault label


def severity_rank(severity: str) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITIES}")
    return SEVERITIES.index(severity)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ID = re.compile(r"\b[0-9a-f]{8,}\b")


def fingerprint(type_: str, source: str, text: str) -> str:
    """Stable hash of type + source + text with numbers / long hex ids replaced by placeholders, so 'leg 3 error 0.21 rad' and
    'leg 5 error 0.34 rad' share one fingerprint (log-template idea)."""
    norm = _NUM.sub("<n>", _ID.sub("<id>", " ".join(text.lower().split())))
    return hashlib.sha1(f"{type_}|{source}|{norm}".encode()).hexdigest()[:16]


@dataclass
class Run:
    run_id: str
    robot_yaml_hash: str = ""
    controller: str = ""
    fault_label: str | None = None          # ground truth from fault injection, never read by the clustering / rules
    config: dict[str, Any] = field(default_factory=dict)   # flattened robot.yaml / gains for config diffs
    created: float = 0.0


@dataclass
class Event:
    run_id: str
    stamp: float
    type: str
    severity: str
    source: str
    text: str = ""
    dedup_key: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    id: int | None = None

    def __post_init__(self) -> None:
        severity_rank(self.severity)
        if not self.fingerprint:
            self.fingerprint = fingerprint(self.type, self.source, self.text)


@dataclass
class Evidence:
    kind: str
    ref: str                                 # event id, metric name@window, log line key, config key
    summary: str
    id: int | None = None


@dataclass
class Cause:
    incident_id: int
    label: str
    confidence: float
    source: str
    evidence: list[Evidence] = field(default_factory=list)
    id: int | None = None


@dataclass
class Incident:
    run_id: str
    start: float
    severity: str
    title: str = ""
    end: float | None = None
    status: str = "open"
    id: int | None = None
    event_ids: list[int] = field(default_factory=list)
