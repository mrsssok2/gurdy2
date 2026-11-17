import json

from brain.agent.ollama_llm import OllamaLLM, build_system, plan_schema
from brain.tools import TOOLS


def test_system_prompt_is_stable_and_has_all_tools():
    a, b = build_system(), build_system()
    assert a == b and all(f'"name":"{n}"' in a for n in TOOLS)
    assert "STATE" not in a          # nothing per-turn in the cached prefix


def test_schema_enum_and_done_only_with_results():
    s0, s1 = plan_schema(), plan_schema(True)
    assert len(s1["anyOf"]) == len(s0["anyOf"]) + 1
    assert set(s0["anyOf"][0]["properties"]["calls"]["items"]["properties"]["tool"]["enum"]) == set(TOOLS)


def test_same_system_message_for_every_task_kind():
    seen = []
    llm = OllamaLLM("m", host="http://x")
    llm._chat = lambda kind, user, fmt: (seen.append((kind, user, fmt)), '{"reply": "x"}' if kind != "respond" else "hi")[1]
    llm.plan("", "STATE a", [], "go", [])
    llm.respond("", "go", [{"tool": "status", "args": {}, "result": {}}])
    assert [k for k, _, _ in seen] == ["plan", "respond"] and seen[0][1].startswith("TASK: plan")


def test_bad_json_becomes_a_reply_and_is_counted():
    llm = OllamaLLM("m", host="http://x")
    llm._chat = lambda *a: "not json"
    assert "reply" in llm.plan("", "S", [], "x", []) and llm.parse_failures == 1
