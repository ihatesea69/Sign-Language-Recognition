"""
TFLite-based point history classifier adapted from the legacy project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from utils.config import Config


class PointHistoryClassifier:
    """Classifier that predicts finger motion gestures from keypoint history."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        score_threshold: float = 0.5,
        invalid_value: int = 0,
        num_threads: int = 1,
    ) -> None:
        resolved_model = Path(
            model_path or Config.GESTURE_POINT_HISTORY_TFLITE
        ).expanduser()

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"PointHistoryClassifier model not found: {resolved_model}"
            )

        self.interpreter = tf.lite.Interpreter(
            model_path=str(resolved_model),
            num_threads=num_threads,
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_threshold = score_threshold
        self.invalid_value = invalid_value

    def __call__(self, point_history: list[float]) -> int:
        input_idx = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_idx,
            np.array([point_history], dtype=np.float32),
        )
        self.interpreter.invoke()

        output_idx = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(output_idx)

        result_index = int(np.argmax(np.squeeze(result)))
        if np.squeeze(result)[result_index] < self.score_threshold:
            result_index = self.invalid_value

        return result_index






