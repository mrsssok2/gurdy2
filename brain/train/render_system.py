"""Re-render the system prompt of an SFT data dir (idempotent): system.txt = compact prompt (what the model is trained on and
what the Ollama client sends), system_json.txt = the old JSON-schema prompt, kept for reference.
    python3 -m brain.train.render_system brain/data/full
The examples (train/val.jsonl) never contain the system prompt, so nothing else changes."""
from __future__ import annotations

import sys
from pathlib import Path

from ..agent.ollama_llm import build_system


def render(d: Path) -> None:
    (d / "system_json.txt").write_text(build_system("json"))
    (d / "system.txt").write_text(build_system("compact"))


if __name__ == "__main__":
    render(Path(sys.argv[1]))
    print("rendered", sys.argv[1])
