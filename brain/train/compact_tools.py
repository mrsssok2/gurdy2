"""Lossless compact notation for the tool block (5.5 measurement; the eval prompt is NOT changed by this file).

  walk(direction: forward|backward|left|right, distance_m: num[0.1,3]) -- Walk a short distance ...
  patrol(places: list<str>[1,12], loops?: int[1,10]) -- ...
`?` = optional property, `str>=1` = minLength, `num[lo,hi]` / `int[lo,hi]` = minimum/maximum (empty side = none),
`list<T>[min,max]` = items/minItems/maxItems, `{...; exactly_one_of(a,b)}` = object constraints.
`decode()` parses it back to the JSON schema, so "no information lost" is checked by equality, not by eye.
    python3 -m brain.train.compact_tools            # token savings with the Qwen3 tokenizer
"""
from __future__ import annotations

import json
import re

CONSTRAINTS = ("exactly_one_of", "at_most_one_of")


def _num(x) -> str:
    return json.dumps(x)


def _bounds(s: dict, lo: str, hi: str) -> str:
    if lo not in s and hi not in s:
        return ""
    return "[" + (_num(s[lo]) if lo in s else "") + "," + (_num(s[hi]) if hi in s else "") + "]"


def enc_type(s: dict) -> str:
    t = s["type"]
    known = {"type", "enum", "minLength", "minimum", "maximum", "items", "minItems", "maxItems", "properties", "required", *CONSTRAINTS}
    extra = set(s) - known
    assert not extra, f"unsupported keys {extra}"
    if "enum" in s:
        assert t == "string" and all(re.fullmatch(r"[\w.]+", v) for v in s["enum"]) and set(s) == {"type", "enum"}, s
        return "|".join(s["enum"])
    if t == "string":
        assert set(s) <= {"type", "minLength"}, s
        return "str" + (f">={s['minLength']}" if "minLength" in s else "")
    if t in ("number", "integer"):
        return ("num" if t == "number" else "int") + _bounds(s, "minimum", "maximum")
    if t == "boolean":
        return "bool"
    if t == "array":
        return f"list<{enc_type(s['items'])}>" + _bounds(s, "minItems", "maxItems")
    if t == "object":
        if "properties" not in s:
            assert set(s) == {"type"}, s
            return "obj"  # free-form object
        return enc_obj(s)
    raise ValueError(t)


def enc_obj(s: dict) -> str:
    req = set(s.get("required", []))
    parts = [f"{k}{'' if k in req else '?'}: {enc_type(v)}" for k, v in s.get("properties", {}).items()]
    assert req <= set(s.get("properties", {})), s
    out = ", ".join(parts)
    for c in CONSTRAINTS:
        if c in s:
            out += f"; {c}({','.join(s[c])})"
    return "{" + out + "}"


def encode_tool(name: str, description: str, params: dict) -> str:
    inner = enc_obj(params)[1:-1]
    return f"{name}({inner}) -- {description}"


# ---------------------------------------------------------------- decoder (proof of losslessness)
class _P:
    def __init__(self, text: str):
        self.s, self.i = text, 0

    def peek(self, lit: str) -> bool:
        return self.s.startswith(lit, self.i)

    def kw(self, word: str) -> bool:
        """A type keyword, not the start of an enum value such as `object|mission`."""
        return self.peek(word) and not re.match(r"[\w|.]", self.s[self.i + len(word):self.i + len(word) + 1])

    def eat(self, lit: str) -> None:
        assert self.peek(lit), (lit, self.s[self.i:self.i + 30])
        self.i += len(lit)

    def rx(self, pattern: str) -> str:
        m = re.compile(pattern).match(self.s, self.i)
        assert m, (pattern, self.s[self.i:self.i + 30])
        self.i = m.end()
        return m.group(0)

    def bounds(self, lo: str, hi: str, d: dict) -> None:
        if not self.peek("["):
            return
        self.eat("[")
        a = self.rx(r"-?[\d.eE+-]*")
        self.eat(",")
        b = self.rx(r"-?[\d.eE+-]*")
        self.eat("]")
        if a:
            d[lo] = json.loads(a)
        if b:
            d[hi] = json.loads(b)

    def typ(self) -> dict:
        if self.peek("{"):
            return self.obj()
        if self.peek("list<"):
            self.eat("list<")
            d = {"type": "array", "items": self.typ()}
            self.eat(">")
            self.bounds("minItems", "maxItems", d)
            return d
        if self.kw("obj"):
            self.eat("obj")
            return {"type": "object"}
        if self.kw("bool"):
            self.eat("bool")
            return {"type": "boolean"}
        if self.kw("str"):
            self.eat("str")
            d = {"type": "string"}
            if self.peek(">="):
                self.eat(">=")
                d["minLength"] = int(self.rx(r"\d+"))
            return d
        if self.kw("num") or self.kw("int"):
            d = {"type": "number" if self.kw("num") else "integer"}
            self.i += 3
            self.bounds("minimum", "maximum", d)
            return d
        return {"type": "string", "enum": self.rx(r"[\w.]+(?:\|[\w.]+)*").split("|")}

    def obj(self) -> dict:
        self.eat("{")
        props, req, extra = {}, [], {}
        while not self.peek("}"):
            if self.peek(";"):
                self.eat("; ")
                c = self.rx(r"\w+")
                self.eat("(")
                extra[c] = self.rx(r"[^)]*").split(",")
                self.eat(")")
                continue
            if props:
                self.eat(", ")
            k = self.rx(r"\w+")
            if not self.peek("?"):
                req.append(k)
            else:
                self.eat("?")
            self.eat(": ")
            props[k] = self.typ()
        self.eat("}")
        return {"type": "object", "properties": props, "required": req, **extra}


def decode_tool(line: str) -> tuple[str, str, dict]:
    name, rest = line.split("(", 1)
    body, desc = rest.rsplit(") -- ", 1)
    return name, desc, _P("{" + body + "}").obj()


def build_compact_block() -> str:
    from ..tools import TOOLS
    return "\n".join(encode_tool(s.name, s.description, s.parameters) for s in TOOLS.values())


def check_roundtrip() -> int:
    from ..tools import TOOLS
    n = 0
    for s in TOOLS.values():
        name, desc, params = decode_tool(encode_tool(s.name, s.description, s.parameters))
        assert (name, desc, params) == (s.name, s.description, s.parameters), s.name
        n += 1
    return n


if __name__ == "__main__":
    from ..tools import TOOLS
    from pathlib import Path
    marker = "Tools (JSON schema of each):\n"  # the data's system.txt is the exact prompt the model is trained on
    full = Path(__file__).resolve().parents[1].joinpath("data/full/system.txt").read_text()
    SYSTEM_RULES, cur_block = full[:full.index(marker) + len(marker)], full[full.index(marker) + len(marker):]
    n = check_roundtrip()
    cmp_block = build_compact_block()
    from ..tools.specs import _CLASSES
    enum_s = "|".join(_CLASSES)
    aliased = f"CLASS={enum_s}\n" + cmp_block.replace(enum_s, "CLASS")  # plain text substitution, trivially reversible
    nodesc = "\n".join(encode_tool(s.name, "", s.parameters) for s in TOOLS.values())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/hf/Qwen3-1.7B/tokenizer.json")
    T = lambda x: len(tok.encode(x).ids)
    rules = T(SYSTEM_RULES)
    res = {"tools": n, "roundtrip_lossless": True, "rules_tokens": rules,
           "tool_block_tokens": {"current_json": T(cur_block), "compact_lossless": T(cmp_block), "compact_class_alias": T(aliased),
                          "compact_without_descriptions_upper_bound": T(nodesc)},
           "chars": {"current_json": len(cur_block), "compact_lossless": len(cmp_block)}}
    cur, cmp_ = res["tool_block_tokens"]["current_json"], res["tool_block_tokens"]["compact_lossless"]
    res["saving_pct_with_alias"] = round(100 * (cur - res["tool_block_tokens"]["compact_class_alias"]) / cur, 1)
    res["saving_tokens"] = cur - cmp_
    res["saving_pct_of_tool_block"] = round(100 * (cur - cmp_) / cur, 1)
    res["system_tokens"] = {"current": rules + cur, "compact": rules + cmp_}
    res["saving_pct_of_system"] = round(100 * (cur - cmp_) / (rules + cur), 1)
    print(json.dumps(res, indent=1))
