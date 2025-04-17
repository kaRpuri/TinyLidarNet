# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Disable interactive dialogues during build
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    python3-pip \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Enable the "universe" repository and update package list
RUN add-apt-repository universe && apt-get update

# Add the ROS Noetic repository and its key
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Update and install ROS Noetic (ros-base), add rospy and Eigen3 dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-rospy \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install additional ROS packages (tf and eigen_conversions)
RUN apt-get update && apt-get install -y \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-msgs \
    ros-noetic-tf2-py \
    ros-noetic-tf2-ros \
    ros-noetic-tf-conversions \
    ros-noetic-eigen-conversions && \
    rm -rf /var/lib/apt/lists/*

# Install additional system and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-rosdep \
    python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/*

# Initialize and update rosdep
RUN rosdep init && rosdep update

# Install additional Python packages via pip: scipy, pymavlink, tqdm, pyserial
RUN pip3 install scipy tqdm pymavlink pyserial

# Automatically source the ROS setup file for new bash sessions
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# (Optional) Set environment variable for eigen_conversions if needed
ENV eigen_conversions_DIR=/opt/ros/noetic/share/eigen_conversions/cmake

# Export PYTHONPATH
ENV PYTHONPATH="/workspace/drone-controller/"

# Default command: start bash
CMD ["/bin/bash"]
