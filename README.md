# turtlebot4_bai08 — Bài 08: facial expressions drive TurtleBot 4

TurtleBot 4 Lite in Ignition Gazebo (ROS 2 Humble) driven by the facial expression seen by the
laptop webcam: OpenCV reads the webcam, DeepFace finds the face and scores 7 expressions, and the
recognized expression becomes a simple trajectory on `/cmd_vel`. Simulation from
[bài 04](https://github.com/Khanhkhan1/turtlebot4_bai04); no map or Nav2.

**Video demo:** https://drive.google.com/file/d/1-AInoUOM27dgqxrS_jsgo4yKJa0Jn9bo/view?usp=sharing

```
webcam --OpenCV--> emotion_recognition_node (DeepFace: SSD face detector + emotion CNN,
       smoothing + debounce) --/detected_emotion (std_msgs/String, 10 Hz)--> robot_behavior_node
       --/cmd_vel--> TurtleBot 4 (Gazebo)        /odom, /scan --> robot_behavior_node
robot_behavior_node --/robot_behavior/state--> webcam window
Gazebo overview camera --/overview_camera/image--> RViz2 ("Gazebo view")
```

## Expressions

| Expression | Robot |
|---|---|
| Happy (Vui vẻ) | curve left (0.15 m/s, radius 0.5 m) while the expression is held |
| Sad (Buồn) | curve right, the same |
| Neutral (Bình thường) | stop |
| Angry (Tức giận) | back up 0.15 m |
| Surprise (Ngạc nhiên) | spin one turn in place |
| Fear (Sợ hãi) | move forward 0.2 m |

Back up, spin and forward are measured on `/odom` and run once per expression (show another
expression to repeat). No face, disgust or a top score under 40 % count as `None` → stop.
The 7 scores are averaged over a few frames and an expression must be held 0.5 s (Neutral/None
0.3 s) before it is published. The robot stops when `/detected_emotion` is silent for 1 s.

## Build

```bash
cd ~/tb4_ws/src && git clone https://github.com/Khanhkhan1/turtlebot4_bai08.git
vcs import . < turtlebot4_bai08/dependencies.repos   # turtlebot4_bai04, turtlebot4, simulator, create3_sim
pip install torch gdown
pip install --no-deps deepface      # PyTorch engine; --no-deps keeps the OpenCV build that has windows
cd ~/tb4_ws && colcon build && source install/setup.bash
```

DeepFace downloads its weights (emotion model 6 MB, SSD detector 10 MB) to `~/.deepface` on the first run.

## Run

```bash
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1          # in every terminal

ros2 launch turtlebot4_bai08 bai08.launch.py           # Gazebo, RViz2, both nodes (camera:=1 for another webcam)
                                                       # (macOS: allow camera access for the terminal)
ros2 launch turtlebot4_bai08 bai08.launch.py min_score:=30   # accept weaker expressions (default 40 %)
ros2 launch turtlebot4_bai08 bai08.launch.py video:=faces.mp4     # a video file instead of the webcam

# the robot without a webcam
ros2 launch turtlebot4_bai08 bai08.launch.py emotion:=false
ros2 topic pub -r 10 /detected_emotion std_msgs/msg/String "{data: Happy}"

cd ~/tb4_ws/src/turtlebot4_bai08 && python3 -m pytest test     # filter tests
```
