"""Readiness gate for the final eval (and any scripted run): do NOT send the first command before Nav2 can really plan and drive.

Why (2026-11-18, final eval): two of 27 missions failed at t = 0 with "I cannot find a path there; it is blocked or in unknown space" and
"Navigation (Nav2 navigate_to_pose) is not running" although the same command worked in other missions. The start-up check only looked from
outside (action servers + TF); the runner's own node had not discovered the action server yet / the global costmap had not got the map.

`wait_ready` polls, in WALL time and bounded, until ALL of: the navigate_to_pose server is visible to THIS node, bt_navigator and both costmaps are lifecycle-active, the global
costmap has received the map, the robot is localized (TF map->base_link) and a real plan to room B succeeds (the exact call the agent's first
`go_to` makes). `first_reply_setup_failure` is the second line of defence: if the agent's very first reply comes within 2 s and says navigation is
not running / unknown space, the attempt is an INFRASTRUCTURE failure (retried), not a robot failure. Pure Python, tested with FakePort."""
from __future__ import annotations

import re
import time
from typing import Callable

LIFECYCLE_NODES = ("bt_navigator", "global_costmap/global_costmap", "local_costmap/local_costmap")
PLAN_PLACE = "room B"

SETUP_RE = re.compile(r"navigation\b.*\bnot running|nav2\b.*\bnot running|unknown space|(cannot|can't|could not|couldn't) find a path|no path", re.I)


class SetupFailed(Exception):
    """Nav2 / localization was not ready: an infrastructure failure of the attempt, never a robot failure."""


def check_all(port, backend=None, plan_place: str = PLAN_PLACE, do_plan: bool = True) -> dict[str, bool]:
    c: dict[str, bool] = {"nav_action_server": bool(port.nav_ready())}
    for n in LIFECYCLE_NODES:
        c[n.split("/")[0] + "_active"] = port.lifecycle_state(n) == "active"
    c["global_costmap_has_map"] = bool(port.global_costmap_ready())
    c["localized"] = port.pose() is not None
    if do_plan and backend is not None and all(c.values()):            # a plan takes up to ~8 s when failing: only once everything else is up
        r = backend.path_check(place=plan_place)
        c["plan_to_" + plan_place.replace(" ", "_")] = bool(r.ok and r.data.get("reachable"))
    return c


def wait_ready(port, backend=None, timeout_s: float = 90.0, poll_s: float = 1.0, clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> dict:
    """-> {"ready": bool, "wall_s": float, "polls": int, "checks": last result}. Never raises; the caller decides."""
    t0, polls, last, shown = clock(), 0, {}, None
    while True:
        polls += 1
        try:
            last = check_all(port, backend)
        except Exception as e:                                           # noqa: BLE001  a flaky service call is "not ready", not a crash
            last = {"error": False}
            log(f"[ready] check raised {type(e).__name__}: {e}")
        if last and all(last.values()):
            return {"ready": True, "wall_s": round(clock() - t0, 1), "polls": polls, "checks": last}
        waiting = sorted(k for k, v in last.items() if not v)
        if waiting != shown:                                             # log only when the set of missing things changes
            log(f"[ready] {clock() - t0:5.1f}s waiting for: {', '.join(waiting)}")
            shown = waiting
        if clock() - t0 >= timeout_s:
            return {"ready": False, "wall_s": round(clock() - t0, 1), "polls": polls, "checks": last}
        sleep(poll_s)


def first_reply_setup_failure(messages: list[dict], window_s: float = 2.0) -> str | None:
    """Text of the agent's first reply when it arrived within `window_s` WALL seconds of the first user message and says Nav2 / the map was not ready."""
    user = next((m for m in messages if m.get("who") == "user"), None)
    if user is None:
        return None
    reply = next((m for m in messages if m.get("who") == "robot" and m is not user and messages.index(m) > messages.index(user)), None)
    if reply is None or reply.get("wall") is None or user.get("wall") is None:
        return None
    if reply["wall"] - user["wall"] <= window_s and SETUP_RE.search(reply.get("text", "")):
        return reply["text"]
    return None
