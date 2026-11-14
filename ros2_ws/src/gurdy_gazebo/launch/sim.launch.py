"""Spawn Gurdy in Gazebo Harmonic with ros2_control (effort PD at robot.yaml's pd_rate_hz).

    ros2 launch gurdy_gazebo sim.launch.py                    # GUI, robot standing on the ground
    ros2 launch gurdy_gazebo sim.launch.py gui:=false         # headless server (tests, less RAM)
    ros2 launch gurdy_gazebo sim.launch.py fixed_base:=true   # body bolted in the air (step tests)
    ros2 launch gurdy_gazebo sim.launch.py robot_yaml:=/path/variant.yaml   # modified robot

Everything numeric comes from robot.yaml via gurdy_core: controller gains/rate, initial stance,
spawn height. The controller parameter file is generated at launch time (not hand-written).
"""
import os
import tempfile

import xacro
import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from gurdy_core.kinematics import leg_ik, nominal_foot
from gurdy_core.robot import load_robot


def controllers_params(robot) -> dict:
    """ros2_control parameters derived from robot.yaml (single source of truth)."""
    c = robot.control
    return {
        "controller_manager": {"ros__parameters": {
            "update_rate": int(c["pd_rate_hz"]),
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            "joint_pd_controller": {"type": "gurdy_control/JointPDController"},
        }},
        "joint_pd_controller": {"ros__parameters": {
            "joints": robot.joint_names,
            "kp": float(c["kp"]),
            "kd": float(c["kd"]),
            "effort_limit": float(robot.joint_common["effort"]),
        }},
    }


def _setup(context):
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"
    fixed = LaunchConfiguration("fixed_base").perform(context).lower() == "true"
    world = LaunchConfiguration("world").perform(context)

    desc_share = get_package_share_directory("gurdy_description")
    robot_yaml = LaunchConfiguration("robot_yaml").perform(context) or \
        os.path.join(desc_share, "config", "robot.yaml")
    robot = load_robot(robot_yaml)

    fd, ctrl_yaml = tempfile.mkstemp(prefix="gurdy_controllers_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(controllers_params(robot), f)

    q0 = leg_ik(robot, nominal_foot(robot))
    base_z = 0.3 if fixed else robot.stance["body_height"] + 0.005
    urdf = xacro.process_file(
        os.path.join(desc_share, "urdf", "gurdy.urdf.xacro"),
        mappings={"sim_gazebo": "true", "controllers_yaml": ctrl_yaml, "robot_yaml": robot_yaml,
                  "initial_q": " ".join(f"{v:.6f}" for v in q0),
                  "fixed_base": str(fixed).lower(), "base_z": str(base_z)},
    ).toxml()

    if not world.endswith(".sdf"):
        world = os.path.join(get_package_share_directory("gurdy_gazebo"), "worlds", world + ".sdf")
    gz_args = f"-r -v 2 {world}" + ("" if gui else " -s")

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items())
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": urdf, "use_sim_time": True}], output="screen")
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "gurdy",
                            "-z", "0.0" if fixed else str(base_z)])
    clock = Node(package="ros_gz_bridge", executable="parameter_bridge",
                 arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], output="screen")
    jsb = Node(package="controller_manager", executable="spawner",
               arguments=["joint_state_broadcaster", "--controller-manager-timeout", "60"])
    pd = Node(package="controller_manager", executable="spawner",
              arguments=["joint_pd_controller", "--controller-manager-timeout", "60"])
    return [
        gz, rsp, clock, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[pd])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("fixed_base", default_value="false"),
        DeclareLaunchArgument("world", default_value="empty"),
        DeclareLaunchArgument("robot_yaml", default_value="",
                              description="alternative robot.yaml (experiments / fault injection)"),
        OpaqueFunction(function=_setup),
    ])
