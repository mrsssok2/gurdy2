"""GaitNode e-stop logic without Gazebo (needs rclpy): same contract as policy_node."""
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros):
    from gurdy_gait.gait_node import GaitNode
    n = GaitNode()
    n.published = []
    n.pub = type("P", (), {"publish": lambda _s, m: n.published.append(np.array(m.data))})()
    yield n
    n.destroy_node()


def run(n, t0, secs, cmd, hz=100):
    t = t0
    while t < t0 + secs:
        if cmd is not None:
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = cmd
            n.on_cmd(m)
        js = JointState()
        js.header.stamp = Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
        n.on_joint_states(js)
        t += 1.0 / hz
    return t


def test_walks_then_estop_goes_to_stand_ignores_cmd_and_releases(node):
    t = run(node, 0.0, 3.0, (0.08, 0.0, 0.0))
    assert np.ptp(np.array(node.published), axis=0).max() > 0.05          # it walks
    node.on_estop(Bool(data=True))
    n0 = len(node.published)
    t = run(node, t, 3.0, (0.08, 0.0, 0.0))                               # /cmd_vel keeps coming: ignored
    assert np.max(np.abs(node.published[-1] - node.q_stance)) < 1e-9
    steps = np.abs(np.diff(np.array(node.published[n0 - 1:]), axis=0))
    assert steps.max() <= node.get_parameter("stand_speed").value * node.gait.dt + 1e-9   # rate-limited, no jump
    node.on_estop(Bool(data=False))
    t = run(node, t, 0.5, None)                                           # released, no command yet: stays standing
    assert np.max(np.abs(node.published[-1] - node.q_stance)) < 1e-9
    run(node, t, 3.0, (0.08, 0.0, 0.0))
    assert np.ptp(np.array(node.published[-100:]), axis=0).max() > 0.05   # walks again after a fresh /cmd_vel
