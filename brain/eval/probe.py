"""Dev tool: send utterances (one conversation, in order) to the REAL agent + Ollama model on the mock world and print what happened: raw model output
of every LLM call, tool calls, safety verdicts, the final reply. For finding root causes of eval failures.
  OLLAMA_URL=... docker compose run --rm brain python3 -m brain.eval.probe --model gurdy-sft-400 "find the blue box" "where is the yellow box"
  --advance S   advance the mock world S seconds between utterances (default 5)   --seen red,blue   pre-seed the memory with objects seen (mock tracks)"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("utterances", nargs="+")
    ap.add_argument("--model", default="gurdy-sft-400")
    ap.add_argument("--advance", type=float, default=5.0)
    ap.add_argument("--keyword", action="store_true", help="use the keyword planner instead of the model")
    a = ap.parse_args()
    from ..agent.graph import Agent
    from ..mock import MockBackend
    from ..tools import ToolLayer
    b = MockBackend()
    if a.keyword:
        from ..agent.llm import KeywordLLM
        llm = KeywordLLM(b.known_places)
    else:
        from ..agent.ollama_llm import OllamaLLM
        llm = OllamaLLM(model=a.model, constrain=not a.model.startswith("gurdy-sft"))
        raw = llm._chat

        def logged(kind, user, fmt):
            out = raw(kind, user, fmt)
            print(f"      [{kind}] -> {out[:300]!r}")
            return out
        llm._chat = logged
    agent = Agent(ToolLayer(b), llm)
    for u in a.utterances:
        print(f"> {u}")
        r = agent.say(u)
        print(f"  calls={[(c.get('tool'), c.get('args')) for c in r.calls]} verdicts={[(v['tool'], v['outcome'], v.get('rule')) for v in r.verdicts]}")
        print(f"  reply: {r.reply}")
        b.advance(a.advance)
        for m in agent.pump():
            print(f"  pump: {m}")


if __name__ == "__main__":
    main()
