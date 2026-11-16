"""detector_node: RGB-D image -> detections -> 3D in the map frame -> semantic map (markers + JSON).

    /camera/image (rgb8) + /camera/depth_image (32FC1) + /camera/camera_info
        -> ONNX detector (gurdy_core.vision.detector)
        -> per detection: robust box depth -> back-project -> object centre (+ class radius prior)
        -> TF camera_optical_frame -> target_frame (map) at the IMAGE stamp
        -> SemanticMap (de-duplication, confirmation after min_hits sightings)
    out:  /detector/detections      vision_msgs/Detection3DArray  (this frame's detections, target frame)
          /semantic_map/markers     visualization_msgs/MarkerArray (confirmed objects: cube + label)
          /semantic_map/objects     std_msgs/String JSON (transient_local; for find_object / go_to)
          /detector/debug_image     sensor_msgs/Image (only with debug_image:=true)

Time: the node does NOT use /clock; it takes sim time from the image stamps (same convention as policy_node), so TF
lookups use the image stamp with `tf_timeout` and need the sim clock on the TF side only.
Detections touching the image border are drawn but not used for the map (truncated boxes bias the centre).
"""
import json
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from gurdy_core.vision.classes import COLORS, SIZE_PRIOR, radius_prior
from gurdy_core.vision.colors import color_name
from gurdy_core.vision.detector import OnnxDetector
from gurdy_core.vision.geometry import K_from_camera_info, backproject, box_depth, surface_to_centre, transform_point
from gurdy_core.vision.semantic_map import SemanticMap

DEFAULT_MODEL = "det_v1.onnx"


def image_to_numpy(m: Image) -> np.ndarray:
    """sensor_msgs/Image (rgb8 or 32FC1) -> numpy without cv_bridge."""
    if m.encoding in ("rgb8", "bgr8"):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
        return a[..., ::-1] if m.encoding == "bgr8" else a
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.step // 4)[:, :m.width]
    if m.encoding == "16UC1":
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width].astype(np.float32) / 1000.0
    raise ValueError(f"unsupported image encoding {m.encoding}")


def _stamp(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        P = self.declare_parameter
        P("model", ""); P("providers", "auto"); P("conf", 0.40); P("iou", 0.5)
        P("target_frame", "map"); P("max_range", 4.0); P("min_range", 0.25); P("max_rate_hz", 5.0)
        P("min_hits", 3); P("ttl", 10.0); P("edge_margin", 3.0); P("tf_timeout", 0.2)
        P("debug_image", False); P("save_path", "")
        model = self.get_parameter("model").value or DEFAULT_MODEL
        if not os.path.isabs(model) and not os.path.exists(model):
            model = os.path.join(os.environ.get("GURDY_REPO", os.path.expanduser("~/gurdy2")), "models", "detector", model)
        g = lambda k: self.get_parameter(k).value
        self.det = OnnxDetector(model, g("providers"), conf=g("conf"), iou=g("iou"))
        self.get_logger().info(f"detector {os.path.basename(model)} on {self.det.provider}, input {self.det.hw}")
        self.map = SemanticMap(min_hits=g("min_hits"), ttl=g("ttl"))
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.K = None
        self.last_t = -1e9
        self.n = {"frames": 0, "tf_fail": 0, "depth_fail": 0, "used": 0}
        self.create_subscription(CameraInfo, "camera/camera_info", self.on_info, qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([Subscriber(self, Image, "camera/image", qos_profile=qos_profile_sensor_data),
                                                 Subscriber(self, Image, "camera/depth_image", qos_profile=qos_profile_sensor_data)],
                                                queue_size=5, slop=0.06)
        self.sync.registerCallback(self.on_images)
        self.pub_det = self.create_publisher(Detection3DArray, "detector/detections", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "semantic_map/markers", 1)
        self.pub_json = self.create_publisher(String, "semantic_map/objects", QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.pub_dbg = self.create_publisher(Image, "detector/debug_image", 1) if g("debug_image") else None
        self.create_timer(5.0, self.log_stats)

    def on_info(self, m: CameraInfo):
        self.K = K_from_camera_info(m.k)

    def on_images(self, rgb_m: Image, depth_m: Image):
        if self.K is None:
            return
        t = _stamp(rgb_m.header)
        if t - self.last_t < 1.0 / self.get_parameter("max_rate_hz").value:
            return
        self.last_t = t
        self.n["frames"] += 1
        rgb, depth = image_to_numpy(rgb_m), image_to_numpy(depth_m)
        dets = self.det(np.ascontiguousarray(rgb))
        frame = self.get_parameter("target_frame").value
        try:
            tr = self.tf.lookup_transform(frame, rgb_m.header.frame_id, rgb_m.header.stamp,
                                          Duration(seconds=self.get_parameter("tf_timeout").value))
        except Exception as e:                                   # noqa: BLE001  (extrapolation / missing frame)
            self.n["tf_fail"] += 1
            if self.n["tf_fail"] in (1, 20, 200):
                self.get_logger().warn(f"TF {frame} <- {rgb_m.header.frame_id} unavailable: {e}")
            return
        t_xyz = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
        q = [tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w]
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = rgb_m.header.stamp, frame
        H, W = rgb.shape[:2]
        em, rmin, rmax = self.get_parameter("edge_margin").value, self.get_parameter("min_range").value, self.get_parameter("max_range").value
        for d in dets:
            x1, y1, x2, y2 = d.box
            z, _ = box_depth(depth, d.box)
            if z is None or not rmin <= z <= rmax:
                self.n["depth_fail"] += 1
                continue
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            p_opt = surface_to_centre(backproject(u, v, z, self.K), radius_prior(d.name))
            p = transform_point(p_opt, t_xyz, q)
            det = Detection3D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = d.name, d.score
            hyp.pose.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            det.results.append(hyp)
            det.bbox.center.position = hyp.pose.pose.position
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = SIZE_PRIOR[d.name]
            out.detections.append(det)
            touches = x1 < em or y1 < em or x2 > W - em or y2 > H - em
            if not touches:
                crop = rgb[int(y1 + 0.25 * (y2 - y1)):int(y2 - 0.25 * (y2 - y1)) + 1, int(x1 + 0.25 * (x2 - x1)):int(x2 - 0.25 * (x2 - x1)) + 1]
                self.map.update(d.name, p, d.score, t, range_m=float(np.linalg.norm(p_opt)), color=color_name(crop))
                self.n["used"] += 1
        self.pub_det.publish(out)
        self.publish_map(rgb_m.header.stamp, frame)

    def publish_map(self, stamp, frame):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for tr in self.map.confirmed():
            w, d, h = SIZE_PRIOR[tr.cls]
            c = COLORS[tr.cls]
            m = Marker()
            m.header.stamp, m.header.frame_id, m.ns, m.id, m.type = stamp, frame, "object", tr.id, Marker.CUBE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(tr.pos[0]), float(tr.pos[1]), h / 2
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = w, d, h
            m.color.r, m.color.g, m.color.b, m.color.a = c[0], c[1], c[2], 0.55
            arr.markers.append(m)
            t = Marker()
            t.header, t.ns, t.id, t.type = m.header, "label", tr.id, Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = float(tr.pos[0]), float(tr.pos[1]), h + 0.15
            t.scale.z = 0.12
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{tr.color + ' ' if tr.color else ''}{tr.cls} #{tr.id} ({tr.hits})"
            arr.markers.append(t)
        self.pub_markers.publish(arr)
        self.pub_json.publish(String(data=self.map.to_json()))

    def log_stats(self):
        self.get_logger().info(f"frames {self.n['frames']}, map updates {self.n['used']}, depth rejects {self.n['depth_fail']}, "
                               f"tf failures {self.n['tf_fail']}, confirmed objects {len(self.map.confirmed())}")

    def save(self):
        path = self.get_parameter("save_path").value
        if path:
            with open(path, "w") as fh:
                fh.write(self.map.to_json())
            self.get_logger().info(f"semantic map saved to {path}")


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.save()
    node.destroy_node()
    rclpy.try_shutdown()
