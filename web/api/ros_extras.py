"""rclpy side of the real-robot telemetry: the Nav2 plan, the RGB camera frame and the detector's 2D boxes. Only importable where ROS 2 is
sourced (the sim image). Subscriptions are added to the agent port's node, so no second node / executor is needed. Images are only
converted and JPEG-encoded when the page asks (<= 2 per second), never per frame."""
from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
from nav_msgs.msg import Path
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

BOXES_MAX_AGE_S = 1.5          # older detector boxes are not drawn on a newer picture
JPEG_QUALITY = 70


class RosExtras:
    def __init__(self, node, plan_topic: str = "/plan", image_topic: str = "/camera/image", boxes_topic: str = "/detector/boxes"):
        self._lock = threading.Lock()
        self._plan: list[tuple[float, float]] = []
        self._img = None
        self._boxes = None
        self._boxes_wall = 0.0
        self._cache: tuple[tuple, dict] | None = None
        node.create_subscription(Path, plan_topic, self._on_plan, 1)
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        node.create_subscription(String, boxes_topic, self._on_boxes, 1)

    def _on_plan(self, m):
        pts = [(p.pose.position.x, p.pose.position.y) for p in m.poses]
        with self._lock:
            self._plan = pts

    def _on_image(self, m):
        with self._lock:
            self._img = m

    def _on_boxes(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with self._lock:
            self._boxes, self._boxes_wall = d, time.monotonic()

    def plan(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._plan)

    def has_camera(self) -> bool:
        return self._img is not None

    def camera(self):
        with self._lock:
            m, boxes, bw = self._img, self._boxes, self._boxes_wall
        if m is None:
            return None
        key = (m.header.stamp.sec, m.header.stamp.nanosec, id(boxes))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        arr = np.frombuffer(bytes(m.data), np.uint8)
        ch = 3 if m.encoding in ("rgb8", "bgr8") else 1
        if m.encoding not in ("rgb8", "bgr8", "mono8"):
            return None
        img = arr.reshape(m.height, m.step)[:, : m.width * ch].reshape(m.height, m.width, ch) if ch == 3 else arr.reshape(m.height, m.step)[:, : m.width]
        if m.encoding == "rgb8":
            img = img[:, :, ::-1]                       # OpenCV wants BGR
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        fresh = boxes is not None and time.monotonic() - bw <= BOXES_MAX_AGE_S
        out = {"w": int(m.width), "h": int(m.height), "jpeg": buf.tobytes(),
               "boxes": boxes["boxes"] if fresh else [], "boxes_w": boxes["w"] if fresh else m.width, "boxes_h": boxes["h"] if fresh else m.height}
        self._cache = (key, out)
        return out
