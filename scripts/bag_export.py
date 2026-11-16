#!/usr/bin/env python3
"""rosbag2 -> npz for offline map analysis (sim image, needs rosbag2_py; no sim running).
    python3 scripts/bag_export.py BAG_DIR OUT.npz
Saves: scan_t, scan_ranges[N,360], scan_angle_min/increment; and pose series (t, x, y, yaw):
  gt (/odom/ground_truth), ekf_tf (TF odom->base_link), map_odom (TF map->odom, slam correction), legs (/odom/legs pose), ekf (/odometry/filtered)."""
import math
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=(sys.argv[1] if __import__("os").path.exists(sys.argv[1]+"/metadata.yaml") else sys.argv[1]+"/bag_0.mcap"), storage_id=("" if __import__("os").path.exists(sys.argv[1]+"/metadata.yaml") else "mcap")), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
S = {k: [] for k in ("gt", "legs", "ekf", "ekf_tf", "map_odom")}
scans, meta = [], {}
while r.has_next():
    topic, data, _ = r.read_next()
    msg = deserialize_message(data, get_message(types[topic]))
    if topic == "/scan":
        scans.append((st(msg.header), np.array(msg.ranges, dtype=np.float32)))
        meta = {"scan_angle_min": msg.angle_min, "scan_angle_increment": msg.angle_increment}
    elif topic in ("/odom/ground_truth", "/odom/legs", "/odometry/filtered"):
        k = {"/odom/ground_truth": "gt", "/odom/legs": "legs", "/odometry/filtered": "ekf"}[topic]
        p = msg.pose.pose
        S[k].append((st(msg.header), p.position.x, p.position.y, yaw(p.orientation)))
    elif topic == "/tf":
        for t in msg.transforms:
            key = ("ekf_tf" if (t.header.frame_id, t.child_frame_id) == ("odom", "base_link") else
                   "map_odom" if (t.header.frame_id, t.child_frame_id) == ("map", "odom") else None)
            if key:
                tr = t.transform.translation
                S[key].append((st(t.header), tr.x, tr.y, yaw(t.transform.rotation)))
out = {k: np.array(v) for k, v in S.items()}
out["scan_t"] = np.array([s[0] for s in scans]); out["scan_ranges"] = np.array([s[1] for s in scans])
out.update({k: np.float64(v) for k, v in meta.items()})
np.savez(sys.argv[2], **out)
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()})
