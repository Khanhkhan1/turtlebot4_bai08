from glob import glob

from setuptools import setup

package_name = "turtlebot4_bai08"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Khanhkhan1",
    maintainer_email="khanh20022810@gmail.com",
    description="Bài 08: TurtleBot 4 driven by facial expressions (DeepFace)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "emotion_recognition_node = turtlebot4_bai08.emotion_recognition_node:main",
            "robot_behavior_node = turtlebot4_bai08.robot_behavior_node:main",
        ],
    },
)
