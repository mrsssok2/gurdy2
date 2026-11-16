#!/usr/bin/env python3
"""Block until Nav2 is really ready: the bt_navigator, planner and controller action servers are all available
(navigate_to_pose, compute_path_to_pose, follow_path) and the map->base_link transform exists.
Exits 0 when ready, 1 on timeout (prints what is missing). Use it before sending the first goal:

    python3 scripts/wait_nav_ready.py [--timeout 300] [--no-tf]

The action servers only appear once their lifecycle node is ACTIVE, so this replaces polling `ros2 lifecycle get`
(which returns before the servers are discoverable and caused goals sent right after start-up to be rejected).
"""
import argparse
import sys
import time

import rclpy
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

SERVERS = [(NavigateToPose, "navigate_to_pose"), (ComputePathToPose, "compute_path_to_pose"), (FollowPath, "follow_path")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300.0, help="wall seconds")
    ap.add_argument("--no-tf", action="store_true", help="do not require map->base_link")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds all conditions must hold continuously")
    a = ap.parse_args()
    rclpy.init()
    n = Node("wait_nav_ready")
    clients = [(name, ActionClient(n, t, name)) for t, name in SERVERS]
    buf = Buffer()
    TransformListener(buf, n)
    t0, ok_since, missing = time.monotonic(), None, []
    while time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
        missing = [name for name, c in clients if not c.server_is_ready()]
        if not a.no_tf:
            try:
                buf.lookup_transform("map", "base_link", rclpy.time.Time())
            except TransformException:
                missing.append("tf map->base_link")
        if missing:
            ok_since = None
        else:
            ok_since = ok_since or time.monotonic()
            if time.monotonic() - ok_since >= a.settle:
                print(f"nav ready after {time.monotonic() - t0:.0f} s wall")
                return 0
    print(f"TIMEOUT after {a.timeout:.0f} s, missing: {', '.join(missing)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
