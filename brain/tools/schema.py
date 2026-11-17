"""Tiny strict validator for the JSON-schema subset the tools use (no pydantic: keeps the runtime image small).

Supported keywords: type (string/integer/number/boolean/array/object), enum, minimum, maximum, minItems,
maxItems, items, properties, required, additionalProperties(false implied for objects). Extra keywords
`exactly_one_of` / `at_most_one_of` (lists of property names) are checked on objects.
The schema dicts are plain JSON schema, so they go straight into an Ollama / OpenAI-style `tools` list.
"""
from __future__ import annotations

from typing import Any

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate(value: Any, schema: dict, path: str = "args") -> list[str]:
    """Return a list of human-readable problems (empty = valid). Strict: no type coercion."""
    errs: list[str] = []
    t = schema.get("type")
    if t and not _TYPES[t](value):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if t == "string" and schema.get("minLength", 0) > len(value):
        errs.append(f"{path}: empty string")
    if t == "array":
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: needs at least {schema['minItems']} items")
        if len(value) > schema.get("maxItems", 10**9):
            errs.append(f"{path}: at most {schema['maxItems']} items")
        for i, v in enumerate(value):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing required '{k}'")
        for k, v in value.items():
            if k not in props:
                errs.append(f"{path}: unknown argument '{k}'")
            else:
                errs += validate(v, props[k], f"{path}.{k}")
        given = lambda names: [n for n in names if value.get(n) is not None]  # noqa: E731
        if "exactly_one_of" in schema and len(given(schema["exactly_one_of"])) != 1:
            errs.append(f"{path}: give exactly one of {schema['exactly_one_of']}")
        if "at_most_one_of" in schema and len(given(schema["at_most_one_of"])) > 1:
            errs.append(f"{path}: give at most one of {schema['at_most_one_of']}")
    return errs
