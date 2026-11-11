"""Facial expression recognition from the laptop webcam with DeepFace + OpenCV.

Webcam (OpenCV) -> DeepFace.analyze(actions=["emotion"]): face detector (SSD by default) + the
DeepFace emotion CNN (7 scores) -> largest face -> scores smoothed over a few frames -> top
expression if it scores at least min_score -> debounce (held for hold_time) -> /detected_emotion
(std_msgs/String, published at publish_rate even when nothing changes, so the robot can tell a
lost signal from "no face"):
    None | Happy | Sad | Neutral | Angry | Surprise | Fear      (disgust has no action: None)
The OpenCV window shows the face box, the 7 scores, the published expression, its action and the
robot state (/robot_behavior/state).

    ros2 run turtlebot4_bai08 emotion_recognition_node [--ros-args -p camera:=0 -p detector:=opencv]
"""
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from turtlebot4_bai08.emotions import ACTIONS, DEEPFACE_LABELS, NAME, NONE, EmotionSmoother, HoldFilter, classify

WINDOW = "Emotion recognition - webcam"
PANEL_W = 270                       # score panel right of the camera image
GREEN, YELLOW, WHITE, GRAY, ORANGE = (0, 220, 0), (0, 230, 255), (255, 255, 255), (150, 150, 150), (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__("emotion_recognition_node")
        self.declare_parameter("camera", 0)              # webcam index
        self.declare_parameter("video", "")              # or a video file instead of the webcam (tests)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("mirror", True)           # selfie view
        self.declare_parameter("detector", "ssd")        # DeepFace detector: ssd, opencv, yolov8n, ...
        self.declare_parameter("backend", "pytorch")     # DeepFace engine: pytorch, tensorflow, onnx
        self.declare_parameter("min_face_confidence", 0.5)
        self.declare_parameter("min_score", 40.0)        # % (smoothed); a lower top score counts as None
        self.declare_parameter("smoothing", 0.35)        # EMA weight of the newest frame
        self.declare_parameter("hold_time", 0.5)         # s an expression must be held before it is published
        self.declare_parameter("stop_hold_time", 0.3)    # Neutral / None: stop quickly
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("show_window", True)
        self.declare_parameter("display_scale", 0.85)    # window size; RViz2 opens right of it

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.mirror = bool(p("mirror"))
        self.detector = p("detector")
        self.min_face_conf = float(p("min_face_confidence"))
        self.min_score = float(p("min_score"))
        self.period = 1.0 / float(p("publish_rate"))
        self.show = bool(p("show_window"))
        self.scale = float(p("display_scale"))
        self.video = p("video")
        stop_hold = float(p("stop_hold_time"))
        self.smoother = EmotionSmoother(alpha=float(p("smoothing")))
        self.filter = HoldFilter(float(p("hold_time")), {"Neutral": stop_hold, NONE: stop_hold})

        if p("backend"):
            os.environ["DEEPFACE_BACKEND_ENGINE"] = p("backend")
        from deepface import DeepFace        # after the engine is chosen; slow import (torch)
        self.deepface = DeepFace
        self.get_logger().info(f"Loading DeepFace (emotion model, {self.detector} detector; "
                               "weights are downloaded to ~/.deepface the first time)")
        self._analyze(np.zeros((480, 640, 3), np.uint8))   # builds both models now, not on the first face

        self.source = os.path.expanduser(self.video) if self.video else int(p("camera"))
        self.size = (int(p("width")), int(p("height")))
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera/video {self.source!r} "
                               "(macOS: allow camera access for the terminal app)")
        self.frame_time = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.video else 0.0

        self.pub = self.create_publisher(String, "detected_emotion", 10)
        self.robot_state = "(robot_behavior_node not running)"
        self.create_subscription(String, "robot_behavior/state", self._on_state, 10)

        self.stable = NONE
        self.last_pub = 0.0
        self.fps, self.t_fps, self.n_fps = 0.0, time.monotonic(), 0
        if self.show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)   # no Qt toolbar
            cv2.moveWindow(WINDOW, 0, 30)
        self.get_logger().info(f"Webcam {self.source!r} -> DeepFace -> /detected_emotion "
                               f"(min score {self.min_score:.0f}%, hold {p('hold_time')} s)")

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not self.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        return cap

    def _on_state(self, msg):
        self.robot_state = msg.data

    # ---------------------------------------------------------------- recognition
    def _analyze(self, frame):
        """Largest detected face: (region dict, {deepface label: percent}) or None."""
        try:
            faces = self.deepface.analyze(frame, actions=["emotion"], detector_backend=self.detector,
                                          enforce_detection=False, silent=True)
        except ValueError as e:      # e.g. a face too small to crop
            self.get_logger().warning(f"DeepFace: {e}", throttle_duration_sec=2.0)
            return None
        h, w = frame.shape[:2]
        best = None
        for f in faces:
            r = f["region"]
            # without a face DeepFace returns the whole image with confidence 0
            if (f.get("face_confidence") or 0.0) < self.min_face_conf or (r["w"] >= w - 1 and r["h"] >= h - 1):
                continue
            if best is None or r["w"] * r["h"] > best[0]["w"] * best[0]["h"]:
                best = (r, {k: float(v) for k, v in f["emotion"].items()})
        return best

    def run(self):
        while rclpy.ok():
            t_frame = time.monotonic()
            ok, frame = self.cap.read()
            if not ok:
                if self.video:
                    self.get_logger().info("End of video")
                    return
                # No more /detected_emotion: robot_behavior_node stops on the missing signal.
                self.get_logger().error("Camera read failed, reopening", throttle_duration_sec=2.0)
                self.cap.release()
                time.sleep(1.0)
                self.cap = self._open()
                continue
            self._process(frame)
            if not rclpy.ok():    # Ctrl-C / SIGTERM shut the context down
                return
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.show:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return
            if self.frame_time:   # play a video file at its own speed
                time.sleep(max(0.0, self.frame_time - (time.monotonic() - t_frame)))

    def _process(self, frame):
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if frame.shape[1] != self.size[0]:
            frame = cv2.resize(frame, (self.size[0], int(frame.shape[0] * self.size[0] / frame.shape[1])))
        face = self._analyze(frame)
        now = time.monotonic()

        smoothed = self.smoother.update(face[1] if face else None, now)
        label, score = classify(smoothed, self.min_score, current=self.stable)
        stable = self.filter.update(label, now)
        if stable != self.stable:
            action = ACTIONS[stable][1] if stable in ACTIONS else "stop"
            self.get_logger().info(f"Expression: {stable} ({score:.0f}%) -> {action}")
            self.stable = stable
        if now - self.last_pub >= self.period:
            self.pub.publish(String(data=stable))
            self.last_pub = now
        if face:   # --log-level emotion_recognition_node:=debug
            top = max(face[1].items(), key=lambda kv: kv[1])
            self.get_logger().debug(f"frame: {top[0]} {top[1]:.0f}%  smoothed: {label} {score:.0f}%")

        self.n_fps += 1
        if now - self.t_fps >= 1.0:
            self.fps, self.t_fps, self.n_fps = self.n_fps / (now - self.t_fps), now, 0
        if self.show:
            view = self._draw(frame, face, smoothed, label, score, now)
            if self.scale != 1.0:
                view = cv2.resize(view, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, view)

    # ---------------------------------------------------------------- window
    def _draw(self, frame, face, smoothed, label, score, now):
        h, w = frame.shape[:2]
        if face:
            r = face[0]
            x0, y0, x1, y1 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]
            color = GREEN if label != NONE else GRAY
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            s = f"{label} {score:.0f}%" if label != NONE else f"? {score:.0f}%"
            (tw, th), _ = cv2.getTextSize(s, FONT, 0.8, 2)
            y_text = y0 - 10 if y0 - 10 > 110 else min(h - 30, y1 + 30)
            x_text = max(5, min(x0, w - tw - 5))
            cv2.rectangle(frame, (x_text - 4, y_text - th - 6), (x_text + tw + 4, y_text + 7), (0, 0, 0), -1)
            text(frame, s, (x_text, y_text), 0.8, color, 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        model = f"DeepFace ({self.detector}): " + (f"{label} {score:.0f}%" if face else "no face")
        action = ACTIONS[self.stable][1] if self.stable in ACTIONS else "stop"
        lines = [(model, WHITE),
                 (f"/detected_emotion: {self.stable} -> {action}", GREEN if self.stable != NONE else GRAY),
                 (f"Robot: {self.robot_state}", ORANGE)]
        for i, (s, color) in enumerate(lines):
            text(frame, s, (10, 25 + 28 * i), 0.62, color, 2)
        text(frame, f"{self.fps:.0f} fps", (w - 75, 25), 0.55, (200, 200, 200))

        held = self.filter.progress(now)
        if held is not None:   # a new expression being held
            cv2.rectangle(frame, (0, h - 10), (int(w * held), h), GREEN, -1)
            text(frame, f"hold {self.filter.candidate}...", (10, h - 18), 0.6, GREEN, 2)

        return np.hstack([frame, self._panel(h, smoothed)])

    def _panel(self, h, smoothed):
        """7 smoothed scores as bars, then the expression -> action table (active row highlighted)."""
        panel = np.full((h, PANEL_W, 3), 30, np.uint8)
        text(panel, "Scores (smoothed)", (10, 24), 0.6, WHITE, 1)
        top = max(smoothed, key=smoothed.get) if smoothed else None
        for i, k in enumerate(DEEPFACE_LABELS):
            y = 40 + 30 * i
            v = smoothed[k] if smoothed else 0.0
            used = NAME[k] in ACTIONS
            color = GREEN if k == top and used else (YELLOW if used else GRAY)
            text(panel, NAME[k], (10, y + 15), 0.5, color if used else GRAY)
            cv2.rectangle(panel, (90, y + 3), (90 + 125, y + 19), (70, 70, 70), 1)
            cv2.rectangle(panel, (90, y + 3), (90 + int(1.25 * v), y + 19), color, -1)
            text(panel, f"{v:3.0f}", (222, y + 16), 0.5, color)
        y = 40 + 30 * len(DEEPFACE_LABELS) + 18
        text(panel, "Expression -> robot", (10, y), 0.6, WHITE, 1)
        for i, (name, (_, action)) in enumerate(ACTIONS.items()):
            yy = y + 26 + 25 * i
            active = name == self.stable
            if active:
                cv2.rectangle(panel, (4, yy - 18), (PANEL_W - 4, yy + 7), (0, 90, 0), -1)
            color, thick = (WHITE, 2) if active else ((200, 200, 200), 1)
            text(panel, name, (10, yy), 0.5, color, thick)
            text(panel, action, (100, yy), 0.5, color, thick)
        return panel

    def close(self):
        self.cap.release()
        if self.show:
            cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = EmotionRecognitionNode()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
