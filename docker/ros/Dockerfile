# Gurdy 2.0 — simulation / ROS image
# ROS 2 Jazzy + Gazebo Harmonic (via ros_gz) + ros2_control + RViz + MuJoCo (Python)
# Ubuntu 24.04 (noble). Runs as the image's built-in `ubuntu` user (UID 1000 = WSL user),
# so files written into the bind-mounted repo are owned by you, not root.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# ROS packages. ros-jazzy-ros-gz pulls in Gazebo Harmonic (the Gazebo paired with Jazzy).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-xacro \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-rviz2 \
      ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-rqt-graph \
      liburdfdom-tools \
      python3-colcon-common-extensions \
      python3-pip python3-pytest python3-yaml python3-numpy python3-scipy \
      python3-absl python3-opengl \
      mesa-utils libgl1-mesa-dri libegl1 libglfw3 \
      bash-completion less nano procps \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo from PyPI. numpy etc. come from apt above, so pip does not replace system numpy
# that ROS Python packages depend on. (Ubuntu 24.04 needs the PEP 668 override.)
RUN pip3 install --no-cache-dir --break-system-packages mujoco==3.3.0

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ubuntu
WORKDIR /home/ubuntu/gurdy2
RUN echo 'export GZ_PARTITION=${GZ_PARTITION:-$(hostname)}  # same as entrypoint.sh' >> ~/.bashrc \
 && echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc \
 && echo '[ -f ~/gurdy2/ros2_ws/install/setup.bash ] && source ~/gurdy2/ros2_ws/install/setup.bash' >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
