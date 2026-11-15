"""Policy node alone (sim already running): ros2 launch gurdy_policy policy.launch.py [mode:=tripod] [model:=...]"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {"mode": "policy", "model": "", "robot_yaml": ""}
    return LaunchDescription([DeclareLaunchArgument(a, default_value=d) for a, d in args.items()] + [
        Node(package="gurdy_policy", executable="policy_node", output="screen",
             parameters=[{a: LaunchConfiguration(a) for a in args}])])
