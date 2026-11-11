"""Expression -> robot action table and the filters between DeepFace and /detected_emotion (no ROS).

DeepFace gives 7 scores (percent) per face: angry, disgust, fear, happy, sad, surprise, neutral.
The scores are smoothed over a few frames (EmotionSmoother), the top expression is kept if it is
one of the 6 used here and scores at least min_score (classify), and it becomes the published
expression once it has been held for hold_time (HoldFilter).
"""

# DeepFace labels, in its order
DEEPFACE_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

NONE = "None"   # no face, unsure, or an expression that has no action (disgust)

# Published name -> (Vietnamese, robot action), as in the assignment table
ACTIONS = {
    "Happy": ("Vui vẻ", "curve left"),
    "Sad": ("Buồn", "curve right"),
    "Neutral": ("Bình thường", "stop"),
    "Angry": ("Tức giận", "back up a little"),
    "Surprise": ("Ngạc nhiên", "spin one turn"),
    "Fear": ("Sợ hãi", "forward a little"),
}
NAME = {label: label.capitalize() for label in DEEPFACE_LABELS}   # "happy" -> "Happy"


class EmotionSmoother:
    """Exponential moving average of the 7 scores. A frame without a face does not change the
    average; after `lost_time` s without a face the average is cleared (output None)."""

    def __init__(self, alpha=0.35, lost_time=0.5):
        self.alpha, self.lost_time = alpha, lost_time
        self.scores, self.last_face = None, None

    def update(self, scores, now):
        """scores: {deepface label: percent} for the face, or None. Returns the smoothed scores or None."""
        if scores is None:
            if self.last_face is None or now - self.last_face > self.lost_time:
                self.scores = None
            return self.scores
        self.last_face = now
        if self.scores is None:
            self.scores = {k: float(scores.get(k, 0.0)) for k in DEEPFACE_LABELS}
        else:
            a = self.alpha
            self.scores = {k: (1 - a) * self.scores[k] + a * float(scores.get(k, 0.0)) for k in DEEPFACE_LABELS}
        return self.scores


def classify(scores, min_score, current=None, margin=10.0):
    """Top expression of the (smoothed) scores: (published name, percent). NONE when there is no
    face, the top score is below min_score, or the top expression is disgust (no action).
    The expression published now (`current`) only needs min_score - margin, so a score close to
    the threshold does not make the output flicker."""
    if not scores:
        return NONE, 0.0
    label, score = max(scores.items(), key=lambda kv: kv[1])
    name = NAME[label]
    threshold = min_score - margin if name == current else min_score
    if score < threshold or name not in ACTIONS:
        return NONE, score
    return name, score


class HoldFilter:
    """Debounce: a label becomes the output once it has been seen for its hold time in a row.
    Another label for less than `grace` s (a flicker) does not restart the count."""

    def __init__(self, default_hold, holds=None, grace=0.2):
        self.default_hold, self.holds, self.grace = default_hold, holds or {}, grace
        self.output = self.candidate = NONE
        self.since = self.last_seen = 0.0
        self.change_at = None

    def hold_for(self, label):
        return self.holds.get(label, self.default_hold)

    def update(self, label, now):
        if label == self.candidate:
            self.last_seen, self.change_at = now, None
        else:
            if self.change_at is None:
                self.change_at = now
            if now - self.last_seen > self.grace:   # not a flicker: the face changed
                self.candidate, self.since, self.last_seen, self.change_at = label, self.change_at, now, None
        if self.candidate != self.output and now - self.since >= self.hold_for(self.candidate):
            self.output = self.candidate
        return self.output

    def progress(self, now):
        """0..1 while a new label is being held, None otherwise (drawn as a bar)."""
        if self.candidate == self.output:
            return None
        return min(1.0, (now - self.since) / max(1e-6, self.hold_for(self.candidate)))
