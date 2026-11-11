"""TurtleBot 4 motions for the recognized facial expressions (/detected_emotion), on /cmd_vel at 20 Hz.

    Happy     curve left: speed m/s on a radius m circle, while the expression is held
    Sad       curve right: the same, turning right
    Neutral   stop (also None: no face, unsure or disgust)
    Angry     back up backup_distance m, once
    Surprise  spin one turn in place, once
    Fear      move forward forward_distance m, once

Angry, Surprise and Fear are short moves measured on /odom that run to the end once started; then the
robot follows the current expression, and the same short move runs again only after the expression
has changed (holding a surprised face gives one turn, not many).
The robot stops when no /detected_emotion message arrives for signal_timeout s, and when the LiDAR
sees an obstacle closer than obstacle_distance m in its direction of motion.
The state is published on /robot_behavior/state (shown in the webcam window).

    ros2 run turtlebot4_bai08 robot_behavior_node
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

CURVES = {"Happy": (1.0, "curve left"), "Sad": (-1.0, "curve right")}   # turn direction
SHORT_MOVES = ("Angry", "Surprise", "Fear")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ShortMove:
    """Back up / forward a distance, or spin an angle, closed-loop on odometry."""

    def __init__(self, name, v, w, target, x, y, yaw, nominal_time):
        self.name, self.v, self.w, self.target = name, v, w, target
        self.x0, self.y0, self.prev_yaw = x, y, yaw
        self.turned = 0.0
        self.time, self.max_time = 0.0, 3.0 * nominal_time + 2.0

    def progress(self, x, y, yaw):
        if self.w:
            self.turned += abs(wrap(yaw - self.prev_yaw))
            self.prev_yaw = yaw
            return self.turned
        return math.hypot(x - self.x0, y - self.y0)

    def command(self, done):
        """(v, w): full speed, slower over the last part so the move ends close to the target."""
        left = self.target - done
        if self.w:
            return 0.0, math.copysign(max(0.25, min(abs(self.w), 2.0 * left)), self.w)
        return math.copysign(max(0.03, min(abs(self.v), 1.5 * left)), self.v), 0.0

    def text(self, done):
        if self.w:
            return f"{self.name}: spin {math.degrees(done):.0f}/{math.degrees(self.target):.0f} deg"
        word = "back up" if self.v < 0 else "forward"
        return f"{self.name}: {word} {done:.2f}/{self.target:.2f} m"


class RobotBehaviorNode(Node):
    def __init__(self):
        super().__init__("robot_behavior_node")
        self.declare_parameter("speed", 0.15)              # m/s, Happy / Sad curves
        self.declare_parameter("radius", 0.5)              # m, curve radius
        self.declare_parameter("backup_distance", 0.15)    # m, Angry
        self.declare_parameter("backup_speed", 0.1)
        self.declare_parameter("forward_distance", 0.2)    # m, Fear
        self.declare_parameter("forward_speed", 0.1)
        self.declare_parameter("spin_turns", 1.0)          # Surprise
        self.declare_parameter("spin_speed", 1.2)          # rad/s
        self.declare_parameter("signal_timeout", 1.0)      # s without /detected_emotion -> stop
        self.declare_parameter("obstacle_distance", 0.3)   # m, 0 = no LiDAR check
        self.declare_parameter("scan_yaw", math.pi / 2)    # rplidar_link yaw in base_link (TurtleBot 4)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.speed, self.radius = float(p("speed")), float(p("radius"))
        self.backup = (float(p("backup_distance")), float(p("backup_speed")))
        self.forward = (float(p("forward_distance")), float(p("forward_speed")))
        self.spin = (2 * math.pi * float(p("spin_turns")), float(p("spin_speed")))
        self.timeout = float(p("signal_timeout"))
        self.obstacle = float(p("obstacle_distance"))
        self.scan_yaw = float(p("scan_yaw"))

        self.emotion, self.last_msg = "None", None
        self.move = None              # ShortMove running
        self.move_done = None         # short move that ended while its expression is still shown
        self.pose = None              # x, y, yaw from /odom
        self.front = self.rear = math.inf
        self.state = ""
        self.zero_left = 0

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "robot_behavior/state", 10)
        self.create_subscription(String, "detected_emotion", self._on_emotion, 10)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        if self.obstacle > 0:
            self.create_subscription(LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.dt = 0.05
        self.create_timer(self.dt, self._tick)
        self.create_timer(0.5, self._publish_state)

        # Create 3 refuses to drive backwards more than 0.15 m unless the safety override allows it.
        self.param_client = self.create_client(SetParameters, "motion_control/set_parameters")
        self.override_timer = self.create_timer(2.0, self._allow_backup)
        self.get_logger().info("Waiting for expressions on /detected_emotion")

    # ---------------------------------------------------------------- inputs
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_emotion(self, msg):
        self.last_msg = self._now()
        if msg.data != self.emotion:
            self.get_logger().info(f"Expression: {msg.data}")
        self.emotion = msg.data
        if self.move_done is not None and self.emotion != self.move_done:
            self.move_done = None     # expression changed: the short move may run again

    def _on_odom(self, odom):
        q = odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (odom.pose.pose.position.x, odom.pose.pose.position.y, yaw)

    def _on_scan(self, scan):
        front = rear = math.inf
        for i, r in enumerate(scan.ranges):
            if not scan.range_min < r < scan.range_max:
                continue
            a = wrap(scan.angle_min + i * scan.angle_increment + self.scan_yaw)   # angle in base_link
            if abs(a) < math.radians(30):
                front = min(front, r)
            elif abs(a) > math.radians(150):
                rear = min(rear, r)
        self.front, self.rear = front, rear

    def _allow_backup(self):
        if not self.param_client.service_is_ready():
            return
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value="backup_only")
        self.param_client.call_async(SetParameters.Request(
            parameters=[Parameter(name="safety_override", value=value)]))
        self.get_logger().info("motion_control: safety_override = backup_only")
        self.override_timer.cancel()

    # ---------------------------------------------------------------- motion
    def _start(self, name):
        x, y, yaw = self.pose
        if name == "Angry":
            d, v = self.backup
            self.move = ShortMove(name, -v, 0.0, d, x, y, yaw, d / v)
        elif name == "Fear":
            d, v = self.forward
            self.move = ShortMove(name, v, 0.0, d, x, y, yaw, d / v)
        else:
            a, w = self.spin
            self.move = ShortMove(name, 0.0, w, a, x, y, yaw, a / w)
        self.get_logger().info(f"Start: {self.move.text(0.0)}")

    def _end_move(self, reason):
        name = self.move.name
        self.get_logger().info(f"{name}: {reason}")
        self.move = None
        if reason == "done" and self.emotion == name:
            self.move_done = name

    def _tick(self):
        now = self._now()
        if self.last_msg is None or now - self.last_msg > self.timeout:
            if self.move:
                self._end_move("stopped, no /detected_emotion")
            return self._stop("no /detected_emotion: stopped" if self.last_msg else "waiting for /detected_emotion")
        if self.pose is None:
            return self._stop("waiting for /odom")

        if self.move is None and self.emotion in SHORT_MOVES and self.emotion != self.move_done:
            self._start(self.emotion)

        if self.move:
            done = self.move.progress(*self.pose)
            self.move.time += self.dt
            if done >= self.move.target - (0.03 if self.move.w else 0.005):
                self._end_move("done")
            elif self.move.time > self.move.max_time:
                self._end_move("gave up (stuck?)")
            else:
                v, w = self.move.command(done)
                if self._blocked(v):
                    self._end_move("stopped, obstacle")
                    return self._stop(f"{self.emotion}: obstacle, stopped")
                return self._drive(v, w, self.move.text(done))

        if self.emotion in CURVES:
            sign, word = CURVES[self.emotion]
            if self._blocked(self.speed):
                return self._stop(f"{self.emotion}: obstacle, stopped")
            return self._drive(self.speed, sign * self.speed / self.radius,
                               f"{self.emotion}: {word} (r {self.radius:.1f} m)")
        if self.emotion == self.move_done:
            return self._stop(f"{self.emotion}: done, change expression to repeat")
        if self.emotion == "Neutral":
            return self._stop("Neutral: stopped")
        return self._stop("no face: stopped" if self.emotion == "None" else f"{self.emotion}: stopped")

    def _blocked(self, v):
        return (v > 0 and self.front < self.obstacle) or (v < 0 and self.rear < self.obstacle)

    def _drive(self, v, w, state):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)
        self.zero_left = 5
        self._set_state(state)

    def _stop(self, state):
        if self.zero_left > 0:     # a few zero commands so the robot surely stops
            self.zero_left -= 1
            self.cmd_pub.publish(Twist())
        self._set_state(state)

    def _set_state(self, state):
        if state.split(" ")[0:2] != self.state.split(" ")[0:2]:   # log kind changes, not progress
            self.get_logger().info(f"Robot: {state}")
        self.state = state

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))


def main():
    rclpy.init()
    node = RobotBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
