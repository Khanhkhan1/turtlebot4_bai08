"""Smoothing, classification and debounce between DeepFace and /detected_emotion (pytest, ROS env for rclpy)."""
import math

from turtlebot4_bai08.emotions import ACTIONS, NONE, EmotionSmoother, HoldFilter, classify
from turtlebot4_bai08.robot_behavior_node import ShortMove


def scores(**kw):
    s = dict.fromkeys(("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"), 0.0)
    s.update(kw)
    return s


def test_six_expressions():
    assert set(ACTIONS) == {"Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear"}


def test_classify():
    assert classify(scores(happy=90, neutral=10), 40) == ("Happy", 90)
    assert classify(scores(surprise=55, fear=45), 40) == ("Surprise", 55)
    assert classify(scores(sad=35, neutral=33, fear=32), 40) == (NONE, 35)       # unsure
    assert classify(scores(disgust=80, angry=20), 40) == (NONE, 80)               # no action
    assert classify(None, 40) == (NONE, 0.0)                                      # no face
    # hysteresis: the expression already published keeps going down to 30 %
    assert classify(scores(neutral=36, sad=34, fear=30), 40, current="Neutral") == ("Neutral", 36)
    assert classify(scores(neutral=36, sad=34, fear=30), 40, current="Happy") == (NONE, 36)


def test_smoother_ignores_one_frame():
    sm = EmotionSmoother(alpha=0.35)
    for i in range(10):
        s = sm.update(scores(happy=95, neutral=5), i * 0.05)
    s = sm.update(scores(sad=90, neutral=10), 0.5)          # one wrong frame
    assert classify(s, 40)[0] == "Happy"
    for i in range(5):                                      # the face really changed
        s = sm.update(scores(sad=90, neutral=10), 0.55 + i * 0.05)
    assert classify(s, 40)[0] == "Sad"


def test_smoother_face_lost():
    sm = EmotionSmoother(lost_time=0.5)
    sm.update(scores(happy=90), 0.0)
    assert sm.update(None, 0.3) is not None                 # short dropout: keep the last scores
    assert sm.update(None, 0.6) is None                     # face gone


def test_hold_filter():
    f = HoldFilter(0.5, {"Neutral": 0.3, NONE: 0.3}, grace=0.2)
    t = 0.0
    while t < 0.45:
        assert f.update("Happy", t) == NONE
        t += 0.05
    assert f.update("Happy", 0.55) == "Happy"
    assert f.update("Sad", 0.6) == "Happy"                  # flicker
    assert f.update("Happy", 0.65) == "Happy"
    for t in (0.7, 0.8, 0.9, 1.0, 1.1):
        out = f.update("Neutral", t)
    assert out == "Neutral"                                 # Neutral needs only 0.3 s


def test_short_moves():
    spin = ShortMove("Surprise", 0.0, 1.2, 2 * math.pi, 0.0, 0.0, 0.0, 5.2)
    yaw, done = 0.0, 0.0
    for _ in range(100):                                    # yaw wraps around +-pi on the way
        yaw = math.atan2(math.sin(yaw + 0.06), math.cos(yaw + 0.06))
        done = spin.progress(0.0, 0.0, yaw)
    assert abs(done - 6.0) < 1e-6
    assert spin.command(done)[1] > 0
    back = ShortMove("Angry", -0.1, 0.0, 0.15, 1.0, 2.0, 0.0, 1.5)
    assert abs(back.progress(1.0, 1.9, 0.0) - 0.1) < 1e-9
    v, w = back.command(0.1)
    assert v < 0 and w == 0.0
