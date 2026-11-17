"""Stage A of the SFT data pipeline: dialogues -> clean examples (design 6.1 steps 1-3, 6, 7).

  python3 -m brain.data_gen.generate --out brain/data/smoke --train 400 --val 60 [--workers 8]

Each dialogue runs through the REAL agent (graph, safety checker, executor, tool layer, grounding check) in a random
mock world, with a `GoldLLM` that answers with the gold outputs of the dialogue. The prompts are built by the very same
`OllamaLLM` code that serves the model, so training and serving prompts cannot drift apart. One example per LLM call.

Outputs (OUT/): system.txt (the one stable system prompt, NOT repeated per example), train.jsonl, val.jsonl,
noise_candidates.json (unique utterances for stage B), stats.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from ..agent import Agent
from ..agent.graph import SYSTEM_PROMPT
from ..agent.ollama_llm import OllamaLLM, build_system
from ..agent.templates import summarize_results
from ..eval.harness import _advance, _settle, observe
from ..tools import ToolLayer
from .families import FAMILIES, Ctx, Dialogue
from .lexicon import norm
from .worlds import base_scenario, build, random_world, with_saved_places

TEST = Path(__file__).resolve().parent.parent / "eval" / "testset_v1.yaml"
VAL_SEED0 = 10_000_000


def test_utterances() -> set[str]:
    cases = yaml.safe_load(TEST.read_text())["cases"]
    return {norm(t["say"]) for c in cases for t in c["turns"]}


class GoldLLM(OllamaLLM):
    """Same prompt construction as the served model; answers with scripted gold. Records every call."""

    def __init__(self):
        super().__init__("gold", host="none")
        self.queue: list = []
        self.recs: list[dict] = []
        self._respond = ""
        self.agent: Agent | None = None

    def _chat(self, kind, user, fmt):
        if kind == "respond":
            out = self._respond
        else:
            if not self.queue:
                raise RuntimeError("gold queue empty: the agent asked the LLM more often than the dialogue scripted")
            o = self.queue.pop(0)
            o = o(self.agent) if callable(o) else o
            out = json.dumps(o, separators=(",", ":"))
        self.recs.append({"kind": kind, "user": user, "assistant": out})
        return out

    def replan(self, system, state, tools, failure, tool_results, feedback=None):
        return {}          # no scripted recovery: the agent reports the failure (re-plan examples come with the DPO stage)

    def respond(self, system, utterance, tool_results, feedback=None):
        self._respond = summarize_results(tool_results)
        return super().respond(system, utterance, tool_results, feedback)


def _pick_family(rng: random.Random):
    w = [f[0] for f in FAMILIES]
    return rng.choices(FAMILIES, w)[0]


def make_dialogue(seed: int, split: str):
    """-> (Dialogue | None, ctx, spec) with the world built for the final scenario."""
    rng = random.Random(seed)
    world_proto = random_world(rng)
    spec = base_scenario(rng, world_proto)
    _, group, fn = _pick_family(rng)
    need: dict = {}
    for attempt in range(2):                                   # pass 2 = rebuild with what the family asked for
        import copy
        world = copy.deepcopy(world_proto)
        b = build({**spec, **need}, world, seed)
        with_saved_places(b, random.Random(seed + 1))
        ctx = Ctx(rng=random.Random(f"{seed}-fam"), split=split, world=world, spec={**spec, **need}, b=b)
        d = fn(ctx)
        if d is None:
            return None, group, ctx
        if ctx.need == need:
            return d, group, ctx
        need = dict(ctx.need)
    return d, group, ctx


def run_dialogue(seed: int, split: str, tests: set[str]):
    """-> (examples, drop_reason | None, family)"""
    d, group, ctx = make_dialogue(seed, split)
    if d is None:
        return [], "family_not_applicable", group
    if any(norm(t.say) in tests for t in d.turns):
        return [], "equals_test_utterance", d.family
    gold = GoldLLM()
    agent = Agent(ToolLayer(ctx.b), gold)
    gold.agent = agent
    out: list[dict] = []
    for ti, turn in enumerate(d.turns):
        gold.queue = list(turn.outputs)
        gold.recs.clear()
        n0 = len(agent.layer.log)
        if turn.bypass_agent:
            try:
                gold.plan("", agent.layer.state_block(), [], turn.say, [])
            except Exception as e:                              # noqa: BLE001
                return [], f"bypass_error:{e}", d.family
            behaviour = "refuse"
        else:
            try:
                res = agent.say(turn.say, turn.stt)
            except Exception as e:                              # noqa: BLE001
                return [], f"agent_error:{type(e).__name__}:{str(e)[:60]}", d.family
            delta = agent.layer.log[n0:]
            behaviour = observe(res, delta, agent)
            if res.grounding.get("fallback") or res.grounding.get("violations"):
                return [], "grounding_violation_in_gold", d.family
            if any(not e["result"]["ok"] and e["result"]["error"]["code"] == "INVALID_ARGS" for e in delta):
                return [], "gold_calls_invalid", d.family
        if gold.queue:
            return [], "gold_not_consumed", d.family
        if not gold.recs:
            return [], "no_llm_call", d.family
        if behaviour not in turn.expect:
            return [], f"behaviour_{behaviour}_not_in_{'|'.join(turn.expect)}", d.family
        for r in gold.recs:
            out.append({"family": d.family, "group": group, "kind": r["kind"], "turn": ti, "n_turns": len(d.turns), "say": turn.say,
                        "behaviour": behaviour, "user": r["user"], "assistant": r["assistant"], "seed": seed, "noisy": False})
        if not turn.bypass_agent:
            if turn.wait_s is None:
                _settle(agent, ctx.b)
            elif turn.wait_s:
                _advance(agent, ctx.b, turn.wait_s)
    return out, None, d.family


def _work(args):
    seed, split, tests = args
    try:
        return seed, run_dialogue(seed, split, tests)
    except Exception as e:                                       # noqa: BLE001
        return seed, ([], f"crash:{type(e).__name__}:{str(e)[:80]}", "?")


def generate(split: str, n_examples: int, seed0: int, workers: int, tests: set[str], log=print):
    examples: list[dict] = []
    seen = set()
    drops: Counter = Counter()
    n_dialogues = 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        it = pool.imap(_work, ((seed0 + i, split, tests) for i in range(n_examples * 3)), chunksize=8)
        for seed, (exs, why, fam) in it:
            n_dialogues += 1
            if why:
                drops[why] += 1
                continue
            for e in exs:
                key = (e["user"], e["assistant"])
                if key in seen:
                    drops["duplicate_example"] += 1
                    continue
                seen.add(key)
                e["id"] = f"{split}-{len(examples):06d}"
                e["split"] = split
                examples.append(e)
            if len(examples) >= n_examples:
                pool.terminate()
                break
            if n_dialogues % 500 == 0:
                log(f"[{split}] dialogues {n_dialogues} examples {len(examples)} ({time.time() - t0:.0f}s)")
    return examples[:n_examples], drops, n_dialogues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system.txt").write_text(build_system())
    tests = test_utterances()
    stats: dict = {"test_utterances": len(tests), "splits": {}}
    cand: dict[str, list[str]] = {}
    for split, n, seed0 in (("train", a.train, a.seed), ("val", a.val, VAL_SEED0 + a.seed)):
        ex, drops, nd = generate(split, n, seed0, a.workers, tests)
        with open(out / f"{split}.jsonl", "w") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        stats["splits"][split] = {
            "examples": len(ex), "dialogues_tried": nd, "dropped": dict(drops),
            "by_group": dict(Counter(e["group"] for e in ex)), "by_family": dict(sorted(Counter(e["family"] for e in ex).items())),
            "by_kind": dict(Counter(e["kind"] for e in ex)), "by_behaviour": dict(Counter(e["behaviour"] for e in ex)),
            "multi_turn_examples": sum(1 for e in ex if e["n_turns"] > 1)}
        for e in ex:
            if e["kind"] == "plan" and e["split"] == split and e["family"] != "unsafe_text":
                cand.setdefault(e["say"], []).append(e["id"])
    (out / "noise_candidates.json").write_text(json.dumps(sorted(cand)))
    stats["noise_candidate_utterances"] = len(cand)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: (v if k != "splits" else {s: {"examples": x["examples"], "dialogues_tried": x["dialogues_tried"], "dropped": x["dropped"],
                                                      "by_group": x["by_group"]} for s, x in v.items()}) for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
