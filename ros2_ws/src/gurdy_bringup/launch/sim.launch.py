"""Gurdy in Gazebo, walking: simulator + controllers + tripod gait node (+ optional RViz).

    ros2 launch gurdy_bringup sim.launch.py                      # Gazebo GUI, sensors on
    ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=false rviz:=true
    ros2 launch gurdy_bringup sim.launch.py controller:=policy      # RL policy (ONNX) instead of the tripod gait
Mapping / navigation (Phase 3; adds the lidar, flat_lab world, odom TF, slam_toolbox or AMCL, Nav2):
    ros2 launch gurdy_bringup sim.launch.py nav:=slam                       # map while driving (teleop or Nav2 goals)
    ros2 launch gurdy_bringup sim.launch.py nav:=localization map:=flat_lab # saved map (maps/flat_lab.yaml) + AMCL
    (nav:=slam|localization|none, scan_source:=lidar|depth, navigation:=false for SLAM only, controller:=tripod|policy)
Drive it (second terminal: docker compose exec sim bash):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _nav_setup(context):
    """nav:=slam|localization: choose world/lidar defaults, add the navigation launch."""
    nav = context.launch_configurations["nav"]
    if nav == "none":
        return []
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("gurdy_navigation"), "launch", "nav.launch.py")),
        launch_arguments={"mode": nav, "navigation": LaunchConfiguration("navigation"),
                          "scan_source": LaunchConfiguration("scan_source"), "map": LaunchConfiguration("map"),
                          "robot_yaml": LaunchConfiguration("robot_yaml"), "rviz": LaunchConfiguration("rviz")}.items())]


def _gazebo_setup(context):
    """The nav profile switches on the lidar (scan_source:=lidar) and, unless given, the flat_lab world."""
    nav = context.launch_configurations["nav"] != "none"
    lc = context.launch_configurations
    lidar = "true" if (nav and lc["scan_source"] == "lidar") else lc["lidar"]
    world = lc["world"] or ("flat_lab" if nav else "empty")
    sensors = "true" if nav else lc["sensors"]          # depth source and IMU need sensors
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("gurdy_gazebo"), "launch", "sim.launch.py")),
        launch_arguments={"gui": lc["gui"], "sensors": sensors, "lidar": lidar, "world": world,
                          "robot_yaml": lc["robot_yaml"], "js_rate": lc["js_rate"]}.items())]


def generate_launch_description():
    args = ["gui", "sensors", "world", "robot_yaml", "js_rate"]
    defaults = {"gui": "true", "sensors": "true", "world": "", "robot_yaml": "", "js_rate": "100"}   # world "" = auto
    sim = OpaqueFunction(function=_gazebo_setup)
    controller = LaunchConfiguration("controller")     # tripod (gait_node) | policy (policy_node, needs sensors:=true)
    gait = Node(package="gurdy_gait", executable="gait_node", output="screen",
                condition=IfCondition(PythonExpression(["'", controller, "' == 'tripod'"])),
                parameters=[{"robot_yaml": LaunchConfiguration("robot_yaml")}])   # timed by /joint_states stamps
    policy = Node(package="gurdy_policy", executable="policy_node", output="screen",
                  condition=IfCondition(PythonExpression(["'", controller, "' == 'policy'"])),
                  parameters=[{"robot_yaml": LaunchConfiguration("robot_yaml"), "model": LaunchConfiguration("model")}])
    rviz = Node(package="rviz2", executable="rviz2", condition=IfCondition(PythonExpression(
                    ["'", LaunchConfiguration("rviz"), "' == 'true' and '", LaunchConfiguration("nav"), "' == 'none'"])),
                arguments=["-d", os.path.join(get_package_share_directory("gurdy_description"), "rviz", "display.rviz")],
                parameters=[{"use_sim_time": True}])
    return LaunchDescription(
        [DeclareLaunchArgument(a, default_value=defaults[a]) for a in args]
        + [DeclareLaunchArgument("rviz", default_value="false"),
           DeclareLaunchArgument("controller", default_value="tripod"),
           DeclareLaunchArgument("nav", default_value="none", description="none | slam | localization"),
           DeclareLaunchArgument("navigation", default_value="true", description="start Nav2 (with nav:=slam|localization)"),
           DeclareLaunchArgument("scan_source", default_value="lidar", description="lidar | depth"),
           DeclareLaunchArgument("lidar", default_value="false", description="force the lidar on without nav"),
           DeclareLaunchArgument("map", default_value="", description="localization: map yaml or name in maps/"),
           OpaqueFunction(function=_nav_setup),
           DeclareLaunchArgument("model", default_value=""), sim, gait, policy, rviz])
