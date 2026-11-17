"""Template phrasing built directly from tool results (every number / name comes from the result, so it is
grounded by construction). Used for motion acknowledgements, event announcements and the grounding fallback."""
from __future__ import annotations

from typing import Any


def _cls(c: str) -> str:
    return c.replace("_", " ")


def obj_name(o: dict) -> str:
    return f"{o.get('color') or ''} {_cls(o['class'])}".strip()


def a_obj(o: dict) -> str:
    n = obj_name(o)
    return ("an " if n[:1] in "aeiou" else "a ") + n


def obj_where(o: dict) -> str:
    return f"{obj_name(o)} in {o['place']}, {o['distance_m']:.1f} metres {o['bearing_word']}"


def _target_name(a: dict) -> str:
    if a.get("place"):
        return a["place"]
    if a.get("object"):
        return obj_name(a["object"])
    if a.get("object_id"):
        return f"object {a['object_id']}"
    if a.get("point"):
        return "that point"
    return "the target"


def ack(tool: str, args: dict, data: dict) -> str:
    """Spoken acknowledgement of a started motion tool, e.g. 'Going to room B, 4.2 metres.'"""
    est = data.get("estimate", {})
    dist = f", {est['distance_m']:.1f} metres" if est.get("distance_m") else ""
    if tool == "walk":
        return f"Walking {args['direction']} {args['distance_m']:g} metres."
    if tool == "turn":
        return f"Turning {args['direction']} {args['angle_deg']:g} degrees."
    if tool == "go_to":
        return f"Going to {_target_name(args)}{dist}."
    if tool == "return_home":
        return f"Going home{dist}."
    if tool == "find":
        return f"Looking for the {obj_name(args['object'])}."
    if tool == "explore":
        return "Exploring."
    if tool == "patrol":
        return "Starting the patrol."
    if tool == "inspect":
        return f"Inspecting {_target_name(args)}{dist}."
    if tool == "stop":
        return "Stopped."
    return "Okay."


def summarize(tool: str, args: dict, res: dict) -> str:
    """One sentence for a tool result (ok or error)."""
    if not res["ok"]:
        return res["error"]["message"]
    d = res["data"]
    if tool == "where_is":
        return "; ".join(f"The {obj_where(o)}" for o in d["matches"]) + "."
    if tool == "whats_nearby":
        parts = [f"{a_obj(o)}, {o['distance_m']:.1f} metres {o['bearing_word']}" for o in d["objects"]]
        obs = [f"obstacle {o['bearing_word']}, {o['distance_m']:.1f} metres" for o in d["obstacles"]]
        items = parts + obs
        return f"In {d['place']}: " + ("; ".join(items) if items else "nothing close by") + "."
    if tool == "where_am_i":
        return f"I am in {d['place']}, {d['home_distance_m']:.1f} metres from home."
    if tool == "relation":
        return f"The first is {d['a_from_b']}, {d['distance_m']:.1f} metres apart."
    if tool == "path_check":
        if not d["reachable"]:
            return f"I cannot get there; blocked by {d['blocked_by']}."
        return f"It is reachable, {d['path_length_m']:.1f} metres, about {d['eta_s']:.0f} seconds."
    if tool == "status":
        legs = [n.replace("_", " ") for n, s in d["legs"].items() if s != "ok"]
        leg_txt = "; leg problem: " + ", ".join(legs) if legs else ""
        return f"Battery {d['battery_pct']:.0f} percent (simulated), mode {d['mode']}{leg_txt}."
    if tool == "why":
        return " ".join(e["text"] for e in d["events"]) or "I have nothing logged about that."
    if tool == "recall":
        recs = d["records"]
        if not recs:
            return "I have no record of that."
        return "; ".join(f"{obj_name(r)} in {r['place']}" if "class" in r else str(r.get("name", r.get("tool"))) for r in recs) + "."
    if tool == "describe_view":
        return d["text"]
    if tool == "report":
        s = d["summary"]
        return f"{s['tool']} {s['state']}: {s['duration_s']:.0f} seconds, {s['distance_m']:.1f} metres, {len(s['objects'])} new objects."
    if tool == "no_go_zone":
        return f"{len(d['zones'])} no-go zones." if d["zones"] else "There are no no-go zones."
    if tool == "save_place":
        return f"Saved this place as {d['place']['name']}."
    if tool == "ask_user":
        return d["question"]
    return ack(tool, args, d)


def summarize_results(entries: list[dict]) -> str:
    """Template reply for all tool results of a turn (grounding fallback)."""
    return " ".join(summarize(e["tool"], e["args"], e["result"]) for e in entries) or "Okay."


def describe_event(ev: dict) -> str | None:
    """Announcement for a backend / executor event, or None if it should stay silent."""
    d, t = ev.get("data", {}), ev["type"]
    if t == "task_finished":
        return _finished(d)
    if t == "task_failed":
        return _finished(d, failed=True)
    if t in ("mission_failed", "mission_finished", "confirmation_needed"):
        return None if t == "mission_finished" else ev["text"]
    if t in ("leg_failure", "path_blocked", "object_missing", "battery_low", "estop"):
        return ev["text"]
    if t == "object_new":
        return ev["text"]
    return None


def _finished(d: dict, failed: bool = False) -> str:
    tool, reason = d.get("tool", "task"), d.get("stopped_reason", "done")
    if tool in ("go_to", "return_home") and not failed:
        return f"I have arrived, {d['final_distance_m']:.1f} metres from the goal."
    if tool == "find":
        o = d.get("object")
        return f"Found the {obj_where(o)}." if d.get("found") and o else f"I did not find it after {d.get('searched_s', 0):.0f} seconds."
    if failed:
        return f"{tool.replace('_', ' ')} stopped: {reason}."
    return f"{tool.replace('_', ' ')} done."
