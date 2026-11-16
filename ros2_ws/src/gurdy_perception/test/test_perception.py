"""detector_node helpers + a full synthetic pipeline check: detections -> 3D -> semantic map, no ROS graph needed."""
import numpy as np
import pytest

from gurdy_core.vision.classes import radius_prior
from gurdy_core.vision.geometry import K_from_camera_info, backproject, box_depth, surface_to_centre, transform_point
from gurdy_core.vision.semantic_map import SemanticMap


def test_image_to_numpy_rgb_and_depth():
    pytest.importorskip("rclpy")
    from sensor_msgs.msg import Image

    from gurdy_perception.detector_node import image_to_numpy
    m = Image()
    m.height, m.width, m.encoding, m.step = 2, 3, "rgb8", 9
    m.data = bytes(range(18))
    assert image_to_numpy(m).shape == (2, 3, 3) and image_to_numpy(m)[1, 2, 2] == 17
    d = Image()
    d.height, d.width, d.encoding, d.step = 2, 2, "32FC1", 8
    d.data = np.array([[1, 2], [3, 4]], np.float32).tobytes()
    assert image_to_numpy(d)[1, 0] == 3.0
    d.encoding = "foo"
    with pytest.raises(ValueError):
        image_to_numpy(d)


def test_pipeline_object_in_map_frame():
    """Barrel 2 m ahead of a robot standing at (1, 1) facing +y (yaw 90 deg): lands at (1, 3.xx) in the map."""
    K = K_from_camera_info([277.0, 0, 160.0, 0, 277.0, 120.0, 0, 0, 1])
    depth = np.full((240, 320), 5.0, np.float32)
    depth[60:180, 120:200] = 2.0 - radius_prior("barrel")      # barrel front surface (centre 2 m away)
    z, _ = box_depth(depth, (120, 60, 200, 180))
    p_opt = surface_to_centre(backproject(160, 120, z, K), radius_prior("barrel"))
    # optical frame of a camera looking along robot +x is (x right, y down, z fwd) = robot (-y, -z, +x): robot yaw 90 deg
    q_cam = np.array([-0.5, 0.5, -0.5, 0.5])                                # optical -> base (rpy -pi/2, 0, -pi/2)
    from gurdy_core.vision.geometry import quat_to_R
    R_base = quat_to_R(q_cam)
    p_base = R_base @ p_opt
    assert p_base[0] == pytest.approx(2.0, abs=1e-6) and abs(p_base[1]) < 1e-6
    q_yaw90 = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    p_map = transform_point(p_base, [1, 1, 0.15], q_yaw90)
    assert p_map[:2] == pytest.approx([1.0, 3.0], abs=0.01)
    m = SemanticMap(min_hits=3)
    for k in range(5):
        m.update("barrel", p_map + np.random.default_rng(k).normal(0, 0.02, 3), 0.9, k, range_m=2.0)
    assert len(m.confirmed()) == 1
