"""Phase 3 checks that need no ROS runtime: footprint, Nav2 params vs robot.yaml, world layout,
launch files parse. Run: colcon test --packages-select gurdy_navigation (or pytest in this dir)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "gurdy_core"))
sys.path.insert(0, str(PKG))

from gurdy_core.rl_command import ANGULAR_DEADBAND, LINEAR_DEADBAND  # noqa: E402
from gurdy_core.robot import load_robot  # noqa: E402
from gurdy_navigation import footprint as fp  # noqa: E402

ROBOT_YAML = REPO / "ros2_ws/src/gurdy_description/config/robot.yaml"


@pytest.fixture(scope="module")
def robot():
    return load_robot(ROBOT_YAML)


@pytest.fixture(scope="module")
def nav2():
    return yaml.safe_load((PKG / "config/nav2_params.yaml").read_text())


def _poly_from(text):
    return yaml.safe_load(text)


def test_footprint_contains_feet_and_body(robot):
    poly = np.array(fp.footprint_polygon(robot))
    hull = fp._hull(poly)
    for p in list(fp.foot_points_body(robot)) + [(0.15, 0.09), (-0.15, -0.09)]:
        # point inside convex CCW polygon: cross products all >= 0
        c = [np.cross(hull[(i + 1) % len(hull)] - hull[i], p - hull[i]) for i in range(len(hull))]
        assert min(c) >= -1e-9, p
    assert 0.25 < fp.max_extent(poly.tolist()) < 0.45


def test_committed_params_match_robot_yaml(robot, nav2):
    patched = fp.patch_nav2_params(nav2, robot)
    assert patched == nav2, "config/nav2_params.yaml is out of date vs robot.yaml (footprint / limits)"


def test_speed_limits_and_deadband(robot, nav2):
    lim = robot.command_limits
    f = nav2["controller_server"]["ros__parameters"]["FollowPath"]
    assert f["vx_max"] <= lim["vx"] and f["vy_max"] <= lim["vy"] and f["wz_max"] <= lim["wz"]
    assert f["motion_model"] == "Omni"
    sm = nav2["velocity_smoother"]["ros__parameters"]
    db = sm["deadband_velocity"]
    assert db[0] > LINEAR_DEADBAND and db[1] > LINEAR_DEADBAND and db[2] > ANGULAR_DEADBAND
    assert db[0] < 0.5 * lim["vx"] and db[2] < 0.5 * lim["wz"]
    assert sm["max_accel"][0] <= 0.5 and sm["max_accel"][2] <= 2.0     # gait node accel limits


def test_collision_monitor_sees_the_lidar_plane(nav2):
    m = nav2["collision_monitor"]["ros__parameters"]["scan"]
    assert m["min_height"] < 0.05 < m["max_height"]        # lidar is 0.05 m above base_link


def test_lidar_config_in_robot_yaml(robot):
    ld = robot.sensors["lidar"]
    assert ld["rays"] >= 180 and ld["rate_hz"] == 10 and ld["range"][1] >= 6.0


def _gen():
    spec = importlib.util.spec_from_file_location("gen_world", REPO / "scripts/gen_world_flat_lab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_world_is_up_to_date_and_navigable(robot):
    g = _gen()
    assert g.OUT.read_text() == g.build(), "run scripts/gen_world_flat_lab.py"
    poly = fp.footprint_polygon(robot)
    width = max(abs(y) for _, y in poly) * 2          # footprint width across the doorway
    gap = g.DOOR_Y[1] - g.DOOR_Y[0] - g.WALL_T * 0     # wall thickness does not narrow a straight gap
    assert gap >= width + 0.3, f"doorway {gap:.2f} m too tight for footprint width {width:.2f} m"
    # spawn (origin) clear of walls and objects by > footprint radius
    r = fp.max_extent(poly)
    for name, kind, xy, size, _ in g.OBJECTS:
        half = max(size[0], size[1]) / 2 if kind == "box" else size[0]
        assert np.hypot(*xy) - half > r + 0.2, name
    # all obstacles taller than the lidar plane (~0.16 m above ground)
    for name, kind, xy, size, _ in g.OBJECTS:
        assert (size[2] if kind == "box" else size[1]) > 0.25, name


def test_odom_transform_helper():
    pytest.importorskip("rclpy")
    from nav_msgs.msg import Odometry
    from gurdy_navigation.odom_tf_node import to_transform
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y = 1.0, 2.0
    m.pose.pose.orientation.w = 1.0
    t = to_transform(m, "odom", "base_link")
    assert (t.header.frame_id, t.child_frame_id) == ("odom", "base_link")
    assert t.transform.translation.x == 1.0 and t.transform.translation.y == 2.0


def test_launch_files_parse():
    pytest.importorskip("launch")
    pytest.importorskip("ament_index_python")
    import launch
    for path in [PKG / "launch/nav.launch.py", REPO / "ros2_ws/src/gurdy_bringup/launch/sim.launch.py"]:
        spec = importlib.util.spec_from_file_location(path.stem + "_t", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert isinstance(mod.generate_launch_description(), launch.LaunchDescription)


# ---------------------------------------------------------------- leg odometry + EKF (Phase 3 step 2)
def test_ekf_config_is_consistent():
    ekf = yaml.safe_load((PKG / "config/ekf.yaml").read_text())["ekf_filter_node"]["ros__parameters"]
    assert ekf["two_d_mode"] and ekf["publish_tf"] and ekf["world_frame"] == "odom" == ekf["odom_frame"]
    assert len(ekf["process_noise_covariance"]) == 225
    assert ekf["odom0"] == "/odom/legs" and ekf["imu0"] == "/imu/ekf"
    names = "x y z roll pitch yaw vx vy vz vroll vpitch vyaw ax ay az".split()
    on = lambda cfg: {n for n, f in zip(names, cfg) if f}                     # noqa: E731
    assert on(ekf["odom0_config"]) == {"vx", "vy", "vyaw"}
    assert on(ekf["imu0_config"]) == {"yaw", "vyaw"}                          # no accelerations (gait vibration)


def test_nav2_odom_topic_patch(nav2, robot):
    p = fp.patch_nav2_params(nav2, robot, odom_topic="/odometry/filtered")
    assert p["bt_navigator"]["ros__parameters"]["odom_topic"] == "/odometry/filtered"
    assert p["velocity_smoother"]["ros__parameters"]["odom_topic"] == "/odometry/filtered"
    assert fp.patch_nav2_params(nav2, robot) == fp.patch_nav2_params(nav2, robot, odom_topic=None)


def test_leg_odometry_message_helpers():
    pytest.importorskip("rclpy")
    from sensor_msgs.msg import Imu
    from gurdy_core.leg_odometry import LegOdomEstimate
    from gurdy_navigation.leg_odometry_node import BIG, fill_imu_covariance, odometry_message
    est = LegOdomEstimate(True, np.array([0.1, -0.02]), 0.3, np.diag([1e-4, 2e-4, 3e-4]), ("lf", "rm", "lr"))
    m = odometry_message(Imu().header.stamp, "odom", "base_link", 1.0, 2.0, 0.5, est)
    assert (m.header.frame_id, m.child_frame_id) == ("odom", "base_link")
    assert m.twist.twist.linear.x == pytest.approx(0.1) and m.twist.twist.angular.z == pytest.approx(0.3)
    assert (m.twist.covariance[0], m.twist.covariance[7], m.twist.covariance[35]) == pytest.approx((1e-4, 2e-4, 3e-4))
    assert m.pose.pose.orientation.z == pytest.approx(np.sin(0.25))
    bad = odometry_message(Imu().header.stamp, "odom", "base_link", 0, 0, 0, LegOdomEstimate(False))
    assert bad.twist.covariance[0] == bad.twist.covariance[35] == BIG          # no stance feet -> "do not believe"
    imu = fill_imu_covariance(Imu(), 0.01, 2.5e-5, 4e-4)
    assert imu.angular_velocity_covariance[8] == 2.5e-5 and imu.orientation_covariance[0] == 0.01
    keep = Imu()
    keep.angular_velocity_covariance[0] = 1e-3
    assert fill_imu_covariance(keep, 0.01, 2.5e-5, 4e-4).angular_velocity_covariance[0] == 1e-3


def test_odometry_launch_actions():
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    spec = importlib.util.spec_from_file_location("nav_launch_t", PKG / "launch/nav.launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.odometry_actions("ground_truth", "", "/x")) == 1
    assert len(mod.odometry_actions("ekf", "", "/x")) == 2
    with pytest.raises(ValueError):
        mod.odometry_actions("magic", "", "/x")
    assert mod.ODOM_TOPIC["ekf"] == "/odometry/filtered"
