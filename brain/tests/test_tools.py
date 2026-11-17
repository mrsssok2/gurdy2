import json

import pytest

from brain.mock import MockBackend
from brain.tools import ERROR_CODES, TOOLS, RobotBackend, ToolLayer, bearing_word, missing_methods, validate


def test_24_tools_and_json_serialisable():
    assert len(TOOLS) == 24
    for s in TOOLS.values():
        json.dumps(s.as_llm_tool())
        assert s.parameters["type"] == "object" and s.description
        assert s.kind in ("motion", "query", "config", "dialogue")


def test_backend_conformance():
    b = MockBackend()
    assert missing_methods(b) == []
    assert isinstance(b, RobotBackend)


@pytest.mark.parametrize("deg,word", [(0, "ahead"), (22, "ahead"), (23, "ahead-left"), (90, "left"), (135, "behind-left"),
                                      (180, "behind"), (-180, "behind"), (-135, "behind-right"), (-90, "right"),
                                      (-45, "ahead-right"), (350, "ahead"), (-200, "behind"), (150, "behind-left")])
def test_bearing_word(deg, word):
    assert bearing_word(deg) == word


def test_schema_strict():
    p = TOOLS["walk"].parameters
    assert validate({"direction": "forward", "distance_m": 1.0}, p) == []
    assert validate({"direction": "up", "distance_m": 1.0}, p)               # enum
    assert validate({"direction": "forward", "distance_m": 5}, p)            # maximum
    assert validate({"direction": "forward"}, p)                             # required
    assert validate({"direction": "forward", "distance_m": 1, "speed": 2}, p)  # unknown arg
    assert validate({"direction": "forward", "distance_m": True}, p)         # bool is not a number
    assert validate({"direction": "forward", "distance_m": "1"}, p)          # no coercion


def test_exactly_one_target():
    p = TOOLS["go_to"].parameters
    assert validate({"place": "room B"}, p) == []
    assert validate({}, p)
    assert validate({"place": "room B", "point": {"x": 1, "y": 2}}, p)
    assert validate({"object": {"class": "sofa"}}, p)                        # unknown class
    assert validate({"object": {"class": "box", "color": "red"}, "standoff_m": 0.5}, p) == []


def test_layer_rejects_before_backend():
    b = MockBackend()
    L = ToolLayer(b)
    r = L.call("walk", {"direction": "forward", "distance_m": 99})
    assert not r.ok and r.code == "INVALID_ARGS" and b.task is None
    assert L.call("fly", {}).code == "UNKNOWN_TOOL"
    assert L.call("walk", {"direction": "forward", "distance_m": 1}).ok


def test_every_error_code_is_in_the_fixed_list():
    b = MockBackend()
    L = ToolLayer(b)
    for name, args in [("go_to", {"place": "mars"}), ("where_is", {"object": {"class": "box"}}), ("fly", {}),
                       ("walk", {"direction": "up", "distance_m": 1})]:
        assert L.call(name, args).code in ERROR_CODES


def test_ask_user_has_no_backend_call():
    L = ToolLayer(MockBackend())
    r = L.call("ask_user", {"question": "Which red box?", "options": ["left", "right"]})
    assert r.ok and r.data["awaiting_user"]
