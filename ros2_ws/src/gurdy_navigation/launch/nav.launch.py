"""Mapping / navigation stack (no simulator): odometry TF + scan source + SLAM or localization + Nav2.

Normally started through `gurdy_bringup sim.launch.py nav:=slam|localization` (adds Gazebo, the gait or
policy node). Can also be used alone against a running simulator / robot:

    ros2 launch gurdy_navigation nav.launch.py mode:=slam                    # map while driving (teleop or Nav2 goals)
    ros2 launch gurdy_navigation nav.launch.py mode:=localization map:=flat_lab   # saved map (maps/flat_lab.yaml) + AMCL

    mode            slam | localization | none (odom TF + scan only)
    navigation      true: also start Nav2 (planner, MPPI controller, costmaps, collision monitor); /cmd_vel out
    odom_source     ground_truth (Gazebo pose, perfect: upper bound) | ekf (leg odometry + IMU through robot_localization)
    scan_source     lidar (/scan straight from the Gazebo bridge) | depth (depthimage_to_laserscan from the RGB-D camera)
    map             localization only: path to map yaml, or a bare name -> <repo>/maps/<name>.yaml
    rviz            RViz with the navigation layout

Footprint and velocity limits in Nav2 come from robot.yaml at launch time (footprint.patch_nav2_params).
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from gurdy_core.robot import load_robot
from gurdy_navigation.footprint import patch_nav2_params

REPO_MAPS = os.path.join(os.path.expanduser("~"), "gurdy2", "maps")


def resolve_map(name: str) -> str:
    if not name:
        name = "flat_lab"
    if name.endswith(".yaml") or os.sep in name:
        return name
    return os.path.join(REPO_MAPS, name + ".yaml")


ODOM_TOPIC = {"ground_truth": "/odom", "ekf": "/odometry/filtered"}


def nav2_params_file(robot_yaml: str, odom_topic: str | None = None) -> str:
    share = get_package_share_directory("gurdy_navigation")
    with open(os.path.join(share, "config", "nav2_params.yaml")) as f:
        params = yaml.safe_load(f)
    robot = load_robot(robot_yaml or os.path.join(
        get_package_share_directory("gurdy_description"), "config", "robot.yaml"))
    fd, path = tempfile.mkstemp(prefix="gurdy_nav2_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(patch_nav2_params(params, robot, odom_topic=odom_topic), f)
    return path


def odometry_actions(odom_source: str, robot_yaml: str, share: str) -> list:
    """ground_truth: Gazebo pose -> /odom + TF (odom_tf_node).
    ekf: leg_odometry_node (/odom/legs, /imu/ekf) -> robot_localization ekf_node -> /odometry/filtered + TF."""
    if odom_source == "ground_truth":
        return [Node(package="gurdy_navigation", executable="odom_tf_node", output="screen")]
    if odom_source != "ekf":
        raise ValueError(f"odom_source must be ground_truth or ekf, not {odom_source!r}")
    return [
        Node(package="gurdy_navigation", executable="leg_odometry_node", output="screen",
             parameters=[{"robot_yaml": robot_yaml}]),
        Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node", output="screen",
             parameters=[os.path.join(share, "config", "ekf.yaml"), {"use_sim_time": True}]),
    ]


def _setup(context):
    cfg = {k: context.launch_configurations.get(k, "") for k in
           ("mode", "navigation", "scan_source", "map", "robot_yaml", "rviz", "odom_source")}
    share = get_package_share_directory("gurdy_navigation")
    nav2_share = get_package_share_directory("nav2_bringup")
    actions = odometry_actions(cfg["odom_source"], cfg["robot_yaml"], share)

    if cfg["scan_source"] == "depth":
        actions.append(Node(
            package="depthimage_to_laserscan", executable="depthimage_to_laserscan_node", output="screen",
            parameters=[os.path.join(share, "config", "depth_scan.yaml"), {"use_sim_time": True}],
            remappings=[("depth", "/camera/depth_image"), ("depth_camera_info", "/camera/camera_info"),
                        ("scan", "/scan")]))

    nav2_params = nav2_params_file(cfg["robot_yaml"], ODOM_TOPIC[cfg["odom_source"]])
    if cfg["mode"] == "slam":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")),
            launch_arguments={"slam_params_file": os.path.join(share, "config", "slam_toolbox.yaml"),
                              "use_sim_time": "true"}.items()))
    elif cfg["mode"] == "localization":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_share, "launch", "localization_launch.py")),
            launch_arguments={"map": resolve_map(cfg["map"]), "params_file": nav2_params,
                              "use_sim_time": "true"}.items()))
    if cfg["navigation"].lower() == "true" and cfg["mode"] in ("slam", "localization"):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_share, "launch", "navigation_launch.py")),
            launch_arguments={"params_file": nav2_params, "use_sim_time": "true"}.items()))
    if cfg["rviz"].lower() == "true":
        actions.append(Node(package="rviz2", executable="rviz2", output="screen",
                            arguments=["-d", os.path.join(share, "config", "nav.rviz")],
                            parameters=[{"use_sim_time": True}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="slam", description="slam | localization | none"),
        DeclareLaunchArgument("navigation", default_value="true", description="start Nav2"),
        DeclareLaunchArgument("scan_source", default_value="lidar", description="lidar | depth"),
        DeclareLaunchArgument("odom_source", default_value="ground_truth", description="ground_truth | ekf"),
        DeclareLaunchArgument("map", default_value="", description="localization: map yaml or name in maps/"),
        DeclareLaunchArgument("robot_yaml", default_value=""),
        DeclareLaunchArgument("rviz", default_value="false"),
        OpaqueFunction(function=_setup),
    ])
