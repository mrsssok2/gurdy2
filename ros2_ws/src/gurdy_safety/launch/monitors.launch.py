"""Battery (SIMULATED), leg health and alert monitor. Run next to the sim:  ros2 launch gurdy_safety monitors.launch.py"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    odom = LaunchConfiguration("odom_topic")
    return LaunchDescription([
        DeclareLaunchArgument("odom_topic", default_value="/odometry/filtered", description="/odom for ground_truth odometry"),
        DeclareLaunchArgument("battery_level", default_value="100.0"),
        Node(package="gurdy_safety", executable="sim_clock_node", output="screen"),     # /clock -> /clock_slow (10 Hz)
        Node(package="gurdy_safety", executable="battery_sim_node", output="screen",
             parameters=[{"odom_topic": odom, "level": LaunchConfiguration("battery_level"), "clock_topic": "/clock_slow"}]),
        Node(package="gurdy_safety", executable="leg_health_node", output="screen"),
        Node(package="gurdy_safety", executable="alert_monitor_node", output="screen", parameters=[{"odom_topic": odom, "clock_topic": "/clock_slow"}]),
    ])
