#!/usr/bin/env python3
"""Bài 08 bring-up: TurtleBot 4 in an open area (open_field.sdf), a fixed overview camera that
shows the Gazebo scene, RViz2, robot_behavior_node and emotion_recognition_node (webcam + DeepFace).

The simulation itself is turtlebot4_bai04/sim.launch.py (Gazebo headless on macOS, bridges,
Create 3 nodes); no SLAM, map or Nav2 here.

    ros2 launch turtlebot4_bai08 bai08.launch.py                   # everything
    ros2 launch turtlebot4_bai08 bai08.launch.py emotion:=false    # robot only (test with ros2 topic pub)
    ros2 launch turtlebot4_bai08 bai08.launch.py video:=faces.mp4  # a video file instead of the webcam
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Overview camera pose (x y z roll pitch yaw): high behind the start point, looking forward and down.
CAMERA_POSE = ["-2.4", "0", "2.4", "0", "0.75", "0"]


def generate_launch_description():
    pkg = get_package_share_directory("turtlebot4_bai08")
    pkg_bai04 = get_package_share_directory("turtlebot4_bai04")

    args = [
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("emotion", default_value="true", choices=["true", "false"],
                              description="Start emotion_recognition_node (webcam)"),
        DeclareLaunchArgument("camera", default_value="0", description="Webcam index"),
        DeclareLaunchArgument("video", default_value="", description="Video file instead of the webcam"),
        DeclareLaunchArgument("detector", default_value="ssd",
                              description="DeepFace face detector: ssd, opencv, yolov8n, ..."),
        DeclareLaunchArgument("min_score", default_value="40.0",
                              description="% an expression needs (lower: Sad/Angry are accepted more easily)"),
    ]

    # Gazebo looks for open_field.sdf here.
    world_path = [AppendEnvironmentVariable(name, os.path.join(pkg, "worlds"))
                  for name in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH")]

    # The include sets rviz:=false for the whole launch (no scoping in Humble): keep ours first.
    keep_rviz = SetLaunchConfiguration("bai08_rviz", LaunchConfiguration("rviz"))
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_bai04, "launch", "sim.launch.py")),
        launch_arguments={"world": "open_field", "rviz": "false"}.items(),
    )

    # Gazebo's Sensors system comes with the robot (Create 3 description): add the camera after it.
    overview_camera = TimerAction(period=12.0, actions=[
        Node(
            package="ros_gz_sim", executable="create", output="log",
            arguments=["-file", os.path.join(pkg, "worlds", "overview_camera.sdf"), "-name", "overview_camera",
                       "-x", CAMERA_POSE[0], "-y", CAMERA_POSE[1], "-z", CAMERA_POSE[2],
                       "-R", CAMERA_POSE[3], "-P", CAMERA_POSE[4], "-Y", CAMERA_POSE[5]],
        ),
        Node(
            package="ros_gz_bridge", executable="parameter_bridge", name="overview_camera_bridge", output="log",
            arguments=["/overview_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image"],
        ),
    ])

    # On macOS the RViz2 OpenGL renderer can segfault while the simulation starts: respawn it.
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(pkg, "rviz", "bai08.rviz")],
        parameters=[{"use_sim_time": True}],
        respawn=True, respawn_delay=2.0,
        condition=IfCondition(LaunchConfiguration("bai08_rviz")),
    )

    behavior = Node(
        package="turtlebot4_bai08", executable="robot_behavior_node", name="robot_behavior_node", output="screen",
        parameters=[{"use_sim_time": True}],
    )
    emotion = Node(
        package="turtlebot4_bai08", executable="emotion_recognition_node", name="emotion_recognition_node",
        output="screen",
        parameters=[{"camera": ParameterValue(LaunchConfiguration("camera"), value_type=int),
                     "video": ParameterValue(LaunchConfiguration("video"), value_type=str),
                     "detector": LaunchConfiguration("detector"),
                     "min_score": ParameterValue(LaunchConfiguration("min_score"), value_type=float)}],
        condition=IfCondition(LaunchConfiguration("emotion")),
    )

    return LaunchDescription(args + world_path + [keep_rviz, sim, overview_camera, rviz, behavior,
                                                  TimerAction(period=15.0, actions=[emotion])])
