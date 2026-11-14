#!/usr/bin/env python3
"""Real-time factor = sim-time advance / wall-time advance, from /clock, over fixed windows.
(Averaging Gazebo's instantaneous /stats values proved unreliable.) Also reports a pure MuJoCo
stepping benchmark so host-speed changes (power mode, thermals) are visible next to the RTF.

    python3 scripts/measure_rtf.py [--warmup 10] [--windows 3] [--window 10]
"""
import argparse
import time

import rclpy
from rosgraph_msgs.msg import Clock


def mujoco_benchmark(steps=20000):
    try:
        import mujoco
        from sim_mujoco.mjcf import build_mjcf
    except ImportError:
        return None
    m = mujoco.MjModel.from_xml_string(build_mjcf())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    t = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return steps * m.opt.timestep / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--window", type=float, default=10.0)
    a = ap.parse_args()
    bench = mujoco_benchmark()
    rclpy.init()
    node = rclpy.create_node("measure_rtf")
    s = {}
    node.create_subscription(Clock, "/clock", lambda m: s.__setitem__("t", m.clock.sec + m.clock.nanosec * 1e-9), 10)
    w0 = time.time()
    while "t" not in s or time.time() - w0 < a.warmup:
        rclpy.spin_once(node, timeout_sec=0.05)
    rtfs = []
    for _ in range(a.windows):
        s0, w0 = s["t"], time.time()
        while time.time() - w0 < a.window:
            rclpy.spin_once(node, timeout_sec=0.05)
        rtfs.append((s["t"] - s0) / (time.time() - w0))
    print(f"RTF windows ({a.window:.0f} s): " + ", ".join(f"{r:.2f}" for r in rtfs)
          + f" | mean {sum(rtfs)/len(rtfs):.2f}"
          + (f" | host benchmark: MuJoCo stepping RTF {bench:.0f}" if bench else ""))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
