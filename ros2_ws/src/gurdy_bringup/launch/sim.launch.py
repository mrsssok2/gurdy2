"""Gurdy in Gazebo, walking: simulator + controllers + tripod gait node (+ optional RViz).

    ros2 launch gurdy_bringup sim.launch.py                      # Gazebo GUI, sensors on
    ros2 launch gurdy_bringup sim.launch.py gui:=false sensors:=false rviz:=true
Drive it (second terminal: docker compose exec sim bash):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.1 -p turn:=0.3
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = ["gui", "sensors", "world", "robot_yaml", "js_rate"]
    defaults = {"gui": "true", "sensors": "true", "world": "empty", "robot_yaml": "", "js_rate": "100"}
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("gurdy_gazebo"), "launch", "sim.launch.py")),
        launch_arguments={a: LaunchConfiguration(a) for a in args}.items())
    gait = Node(package="gurdy_gait", executable="gait_node", output="screen",
                parameters=[{"robot_yaml": LaunchConfiguration("robot_yaml")}])   # timed by /joint_states stamps
    rviz = Node(package="rviz2", executable="rviz2", condition=IfCondition(LaunchConfiguration("rviz")),
                arguments=["-d", os.path.join(get_package_share_directory("gurdy_description"), "rviz", "display.rviz")],
                parameters=[{"use_sim_time": True}])
    return LaunchDescription(
        [DeclareLaunchArgument(a, default_value=defaults[a]) for a in args]
        + [DeclareLaunchArgument("rviz", default_value="false"), sim, gait, rviz])
