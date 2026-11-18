"""Lazy exports (PEP 562): `import brain.agent.rooms` / `.stopwords` must not pull in langgraph (the GPU voice image has none)."""
from __future__ import annotations

import importlib

_EXPORTS = {"Agent": "graph", "TurnResult": "graph", "is_estop_phrase": "stopwords", "MissionExecutor": "executor",
            "SafetyChecker": "safety", "Verdict": "safety", "ConfirmationGate": "safety", "is_clear_yes": "safety",
            "collect_facts": "grounding", "grounded_reply": "grounding", "violations": "grounding",
            "ScriptedLLM": "llm", "KeywordLLM": "llm"}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
