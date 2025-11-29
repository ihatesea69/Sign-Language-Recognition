"""
TFLite-based keypoint classifier adapted from the legacy
`hand-gesture-recognition-mediapipe` project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from utils.config import Config


class KeyPointClassifier:
    """Wrapper around the legacy TFLite keypoint classifier."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        num_threads: int = 1,
    ) -> None:
        resolved_model = Path(
            model_path or Config.GESTURE_KEYPOINT_TFLITE
        ).expanduser()

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"KeyPointClassifier model not found: {resolved_model}"
            )

        self.interpreter = tf.lite.Interpreter(
            model_path=str(resolved_model),
            num_threads=num_threads,
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list: list[float]) -> int:
        input_idx = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_idx,
            np.array([landmark_list], dtype=np.float32),
        )
        self.interpreter.invoke()

        output_idx = self.output_details[0]["index"]
        output = self.interpreter.get_tensor(output_idx)

        return int(np.argmax(np.squeeze(output)))






