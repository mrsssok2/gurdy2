"""Node-level smoke tests without Gazebo (need rclpy): messages in, messages out."""
import json

import pytest

rclpy = pytest.importorskip("rclpy")
from builtin_interfaces.msg import Time  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_battery_node_publishes_simulated_percentage(ros):
    from gurdy_safety.battery_sim_node import BatterySim
    from rosgraph_msgs.msg import Clock
    n = BatterySim()
    out = []
    n.pub = type("P", (), {"publish": lambda _s, m: out.append(m)})()
    for t in (0, 100):
        n._on_clock(Clock(clock=Time(sec=t)))
        n.speed = 0.1
        n.tick()
    b = out[-1]
    assert b.location == "SIMULATED" and b.header.frame_id == "SIMULATED" and 0.96 < b.percentage < 0.98     # 3 % in 100 s moving
    n.destroy_node()


def test_leg_health_node_flags_a_leg_that_does_not_follow(ros):
    import math
    from gurdy_safety.leg_health_node import LegHealthNode
    n = LegHealthNode()
    pub = []
    n.pub = type("P", (), {"publish": lambda _s, m: pub.append(json.loads(m.data))})()
    names = list(n.index)
    for k in range(600):
        t = k * 0.01 + 1.0
        ref = [0.5 * math.sin(2 * math.pi * t + i) for i in range(18)]
        q = list(ref)
        for j in (12, 13, 14):                               # leg index 4 = "rm" frozen at 0.1
            q[j] = 0.1
        n._cmd(Float64MultiArray(data=ref))
        js = JointState(name=names, position=q)
        js.header.stamp = Time(sec=int(t), nanosec=int((t % 1) * 1e9))
        n._js(js)
    assert pub and pub[-1]["legs"]["middle_right"] == "failed" and pub[-1]["legs"]["front_left"] == "ok"
    n.destroy_node()
