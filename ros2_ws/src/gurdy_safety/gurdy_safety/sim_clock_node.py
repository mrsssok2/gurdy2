"""/clock (about 400 Hz in Gazebo) -> /clock_slow (rosgraph_msgs/Clock, 10 Hz wall timer).

Python nodes that only need "what time is it, roughly" (explorer, battery, alert monitor) subscribe to /clock_slow instead of
/clock: in the 5.6 live test each of them burned 40-90 % of a core on the 400 Hz callbacks. Resolution 0.1 s wall.
Only this node pays for the fast topic.
"""
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class SimClock(Node):
    def __init__(self):
        super().__init__("sim_clock_node")
        self.last = None
        self.pub = self.create_publisher(Clock, "/clock_slow", 10)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "last", m), 1)
        self.create_timer(0.1, self.tick)

    def tick(self):
        if self.last is not None:
            self.pub.publish(self.last)


def main():
    rclpy.init()
    n = SimClock()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
