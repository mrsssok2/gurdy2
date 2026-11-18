import time
from brain.ros.rclpy_port import RclpyPort
from brain.ros import RosBackend
p = RclpyPort(map_id="flat_lab")
while p.pose() is None:
    time.sleep(1)
b = RosBackend(p, "/tmp/p56/places_clean.yaml")
def wait(tmax=200):
    t0 = p.now()
    while p.now() - t0 < tmax:
        time.sleep(0.3); b.tick()
        if b.task is None: break
    print(b.events[-1]["text"], flush=True)
for call in [lambda: b.walk("forward", 1.5), lambda: b.walk("backward", 1.0), lambda: b.walk("left", 0.8), lambda: b.walk("right", 0.8),
             lambda: b.turn("left", 120), lambda: b.turn("right", 120), lambda: b.go_to(place="room a"), lambda: b.go_to(place="room b"), lambda: b.return_home()]:
    r = call(); print(r, flush=True); wait()
