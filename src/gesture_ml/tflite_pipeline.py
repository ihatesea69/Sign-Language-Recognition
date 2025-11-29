"""
Utilities to run the legacy MediaPipe + TFLite pipeline inside the new app.
"""

from __future__ import annotations

import csv
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from utils.config import Config

from .keypoint_classifier import KeyPointClassifier
from .point_history_classifier import PointHistoryClassifier


Landmark = Tuple[int, int, int]
Point = Tuple[int, int]


def _load_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]


def _preprocess_landmarks(landmarks: Sequence[Sequence[int]]) -> list[float]:
    temp = [list(point) for point in landmarks]

    base_x, base_y = temp[0]
    for idx, (x, y) in enumerate(temp):
        temp[idx][0] = x - base_x
        temp[idx][1] = y - base_y

    flattened = list(np.array(temp).flatten())
    max_value = max(map(abs, flattened)) or 1.0

    return [value / max_value for value in flattened]


def _preprocess_point_history(
    image_shape: Tuple[int, int],
    point_history: Iterable[Point],
) -> list[float]:
    height, width = image_shape[:2]

    temp = [list(point) for point in point_history]
    if not temp:
        return []

    base_x, base_y = temp[0]
    for idx, (x, y) in enumerate(temp):
        temp[idx][0] = (x - base_x) / width
        temp[idx][1] = (y - base_y) / height

    return list(np.array(temp).flatten())


@dataclass
class GesturePrediction:
    """Container with the result of the TFLite pipeline."""

    hand_sign_id: int
    hand_sign_label: str
    finger_gesture_id: int
    finger_gesture_label: str


class GestureDataRecorder:
    """Handles CSV logging for training data collection."""

    def __init__(
        self,
        keypoint_csv: Path,
        point_history_csv: Path,
    ) -> None:
        self.keypoint_csv = keypoint_csv
        self.point_history_csv = point_history_csv
        self.mode: Literal["off", "keypoint", "point_history"] = "off"
        self.current_label: int = -1

        keypoint_csv.parent.mkdir(parents=True, exist_ok=True)
        point_history_csv.parent.mkdir(parents=True, exist_ok=True)

    def set_mode(self, mode: Literal["off", "keypoint", "point_history"]) -> None:
        self.mode = mode

    def set_label(self, label: int) -> None:
        self.current_label = label

    def log(
        self,
        landmark_list: Sequence[float],
        point_history_list: Sequence[float],
    ) -> None:
        if not (0 <= self.current_label <= 9):
            return

        if self.mode == "keypoint":
            with self.keypoint_csv.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.current_label, *landmark_list])
        elif self.mode == "point_history":
            with self.point_history_csv.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.current_label, *point_history_list])


class TFLiteGesturePipeline:
    """High-level interface used by the OpenCV main loop."""

    def __init__(
        self,
        history_length: int = 16,
        enable_logging: bool = False,
        point_history_trigger_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.history_length = history_length
        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        self.keypoint_labels = _load_labels(Config.GESTURE_KEYPOINT_LABELS)
        self.point_history_labels = _load_labels(
            Config.GESTURE_POINT_HISTORY_LABELS
        )

        self.trigger_ids = (
            list(point_history_trigger_ids)
            if point_history_trigger_ids is not None
            else Config.GESTURE_POINT_HISTORY_TRIGGER_IDS
        )

        self.logger = (
            GestureDataRecorder(
                Config.GESTURE_KEYPOINT_CSV,
                Config.GESTURE_POINT_HISTORY_CSV,
            )
            if enable_logging
            else None
        )

    def set_logging_mode(
        self, mode: Literal["off", "keypoint", "point_history"]
    ) -> None:
        if self.logger:
            self.logger.set_mode(mode)

    def set_logging_label(self, label: int) -> None:
        if self.logger:
            self.logger.set_label(label)

    def process(
        self,
        landmarks: Sequence[Landmark],
        image_shape: Tuple[int, int, int],
    ) -> Optional[GesturePrediction]:
        if not landmarks:
            self.point_history.append((0, 0))
            return None

        ordered_points: list[Point] = [(0, 0)] * 21
        for idx, x, y in landmarks:
            if 0 <= idx < len(ordered_points):
                ordered_points[idx] = (x, y)

        preprocessed_landmarks = _preprocess_landmarks(ordered_points)
        preprocessed_history = _preprocess_point_history(
            image_shape, self.point_history
        )

        if self.logger:
            self.logger.log(preprocessed_landmarks, preprocessed_history)

        hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
        hand_sign_label = (
            self.keypoint_labels[hand_sign_id]
            if hand_sign_id < len(self.keypoint_labels)
            else str(hand_sign_id)
        )

        fingertip = ordered_points[8]
        if hand_sign_id in self.trigger_ids:
            self.point_history.append(fingertip)
        else:
            self.point_history.append((0, 0))

        finger_gesture_id = 0
        history_vector = _preprocess_point_history(image_shape, self.point_history)
        if len(history_vector) == self.history_length * 2:
            finger_gesture_id = self.point_history_classifier(history_vector)

        self.finger_gesture_history.append(finger_gesture_id)
        most_common = Counter(self.finger_gesture_history).most_common()
        finger_label_idx = most_common[0][0] if most_common else 0

        finger_label = (
            self.point_history_labels[finger_label_idx]
            if finger_label_idx < len(self.point_history_labels)
            else str(finger_label_idx)
        )

        return GesturePrediction(
            hand_sign_id=hand_sign_id,
            hand_sign_label=hand_sign_label,
            finger_gesture_id=finger_label_idx,
            finger_gesture_label=finger_label,
        )






