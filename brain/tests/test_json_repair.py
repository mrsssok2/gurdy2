"""repair_json: structural slips in the model's plan JSON must not turn a correct plan into "I did not understand".
Run: python3 -m unittest brain.tests.test_json_repair  (stdlib only)"""
import json
import unittest

from brain.agent.ollama_llm import OllamaLLM, repair_json

PLAN = {"calls": [{"tool": "walk", "args": {"direction": "forward", "distance_m": 2}}, {"tool": "turn", "args": {"direction": "right", "angle_deg": 90}},
                  {"tool": "return_home", "args": {}}], "continue": False}


def load(s):
    out = repair_json(s)
    return None if out is None else json.loads(out)


class Repair(unittest.TestCase):
    def test_valid_json_is_unchanged(self):
        self.assertEqual(load(json.dumps(PLAN)), PLAN)

    def test_stray_closing_brace_in_a_middle_call(self):
        s = '{"calls":[{"tool":"walk","args":{"direction":"forward","distance_m":2}}},{"tool":"turn","args":{"direction":"right","angle_deg":90}},{"tool":"return_home","args":{}}],"continue":false}'
        self.assertEqual(load(s), PLAN)

    def test_stray_closing_bracket_and_double_brace_at_the_end(self):
        self.assertEqual(load('{"reply":"ok"}}'), {"reply": "ok"})
        self.assertEqual(load('{"calls":[{"tool":"stop","args":{}}]],"continue":false}'), {"calls": [{"tool": "stop", "args": {}}], "continue": False})

    def test_truncated_output_gets_its_closers(self):
        s = '{"calls":[{"tool":"go_to","args":{"place":"room A"}},{"tool":"return_home","args":{}}'
        self.assertEqual(load(s), {"calls": [{"tool": "go_to", "args": {"place": "room A"}}, {"tool": "return_home", "args": {}}]})

    def test_text_around_the_object_and_markdown_fences(self):
        self.assertEqual(load('Here is the plan: {"reply":"hello"} Thanks!'), {"reply": "hello"})
        self.assertEqual(load('```json\n{"done": true}\n```'), {"done": True})

    def test_braces_and_quotes_inside_strings_are_left_alone(self):
        s = '{"reply":"use {curly} and \\"quotes\\" and ]brackets["}'
        self.assertEqual(load(s), {"reply": 'use {curly} and "quotes" and ]brackets['})
        self.assertEqual(load('{"reply":"a } b"}}'), {"reply": "a } b"})

    def test_hopeless_text_is_none(self):
        for s in ("", "no json here", '{"calls": [1, 2,,]}', '{"a": }', "}}}"):
            self.assertIsNone(repair_json(s), s)


class Parse(unittest.TestCase):
    """The same repair through the planner's _parse (different wording than the eval mission)."""

    def llm(self):
        return OllamaLLM(model="x")

    def test_slip_in_a_new_plan_still_plans(self):
        llm = self.llm()
        text = '{"calls":[{"tool":"patrol","args":{"places":["room A","room B"],"loops":2}}},{"tool":"status","args":{}}],"continue":false}'
        out = llm._parse(text)
        self.assertEqual([c["tool"] for c in out["calls"]], ["patrol", "status"])
        self.assertEqual((llm.repaired, llm.parse_failures), (1, 0))

    def test_unrepairable_output_still_says_did_not_understand(self):
        llm = self.llm()
        self.assertEqual(llm._parse("total nonsense"), {"reply": "Sorry, I did not understand that."})
        self.assertEqual(llm.parse_failures, 1)

    def test_valid_output_is_not_counted_as_repaired(self):
        llm = self.llm()
        llm._parse('{"reply":"hi"}')
        self.assertEqual(llm.repaired, 0)


if __name__ == "__main__":
    unittest.main()
