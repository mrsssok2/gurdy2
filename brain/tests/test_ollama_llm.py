import json

from brain.agent.ollama_llm import OllamaLLM, build_system, plan_schema
from brain.tools import TOOLS


def test_system_prompt_is_stable_and_has_all_tools():
    a, b = build_system(), build_system()
    assert a == b and all(f"{n}(" in a for n in TOOLS)
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


def test_compact_is_default_json_selectable_and_both_stable():
    from brain.agent.ollama_llm import OllamaLLM
    assert build_system() == build_system("compact") != build_system("json")
    assert all(f'"name":"{n}"' in build_system("json") for n in TOOLS) and "CLASS=" in build_system()
    assert len(build_system()) < 0.6 * len(build_system("json"))
    assert OllamaLLM("m", host="http://x").system == build_system()
    assert OllamaLLM("m", host="http://x", prompt_style="json").system == build_system("json")


def test_compact_block_is_lossless_with_alias():
    from brain.train.compact_tools import check_roundtrip
    assert check_roundtrip() == len(TOOLS)


def test_render_system_writes_both(tmp_path):
    from brain.train.render_system import render
    render(tmp_path)
    assert (tmp_path / "system.txt").read_text() == build_system() and (tmp_path / "system_json.txt").read_text() == build_system("json")
