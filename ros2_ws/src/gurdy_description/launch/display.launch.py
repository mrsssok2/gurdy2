"""Show the robot model in RViz with joint sliders.

    ros2 launch gurdy_description display.launch.py            # sliders start in nominal stance
    ros2 launch gurdy_description display.launch.py gui:=false # no slider window
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _stance_zeros():
    """Start the sliders in the nominal stance (from gurdy_core IK) if gurdy_core is importable."""
    try:
        from gurdy_core.kinematics import nominal_joint_angles
        from gurdy_core.robot import load_robot
        return nominal_joint_angles(load_robot())
    except Exception as e:  # keep the viewer usable even without gurdy_core
        print(f"[display.launch] gurdy_core not available ({e}); sliders start at 0")
        return {}


def generate_launch_description():
    pkg = FindPackageShare("gurdy_description")
    gui = LaunchConfiguration("gui")
    description = ParameterValue(
        Command(["xacro ", PathJoinSubstitution([pkg, "urdf", "gurdy.urdf.xacro"])]), value_type=str)
    zeros = _stance_zeros()

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="joint slider window"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": description}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             condition=IfCondition(gui), parameters=[{"zeros": zeros}] if zeros else []),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             condition=UnlessCondition(gui), parameters=[{"zeros": zeros}] if zeros else []),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", PathJoinSubstitution([pkg, "rviz", "display.rviz"])]),
    ])
