"""
Wake Word Detection Engine for JARVIS Voice Assistant.
Uses openWakeWord to detect "Hey Jarvis" locally and offline.
"""

import os
from typing import Optional
import numpy as np
import openwakeword
from openwakeword.model import Model

from config import DEFAULT_WAKEWORD_THRESHOLD


class WakeWordEngine:
    def __init__(
        self,
        model_name: str = "hey_jarvis_v0.1.onnx",
        threshold: float = DEFAULT_WAKEWORD_THRESHOLD,
    ):
        self.threshold = threshold
        self.model: Optional[Model] = None
        self.model_name_key = "hey_jarvis_v0.1"

        # Locate ONNX model
        resources_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
        self.model_path = os.path.join(resources_dir, model_name)

        # 1280 samples buffer (80ms at 16000 Hz)
        self.buffer = np.array([], dtype=np.int16)
        self.last_score = 0.0

    def load(self):
        """Initialize openWakeWord ONNX model."""
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            try:
                import openwakeword.utils
                openwakeword.utils.download_models([self.model_name_key])
            except Exception:
                pass

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Wake word model not found: {self.model_path}")

        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
        )
        if self.model.models:
            # Find key matching jarvis
            matched = next((k for k in self.model.models.keys() if "jarvis" in k.lower()), None)
            self.model_name_key = matched or list(self.model.models.keys())[0]

    def reset(self):
        """Reset internal detection state."""
        self.buffer = np.array([], dtype=np.int16)
        self.last_score = 0.0
        if self.model is not None:
            self.model.reset()

    def get_last_score(self) -> float:
        return self.last_score

    def process_pcm_chunk(self, raw_bytes: bytes) -> bool:
        """
        Process a chunk of 16-bit 16kHz PCM audio bytes.
        Returns True if "Hey Jarvis" wake word is detected.
        """
        if self.model is None:
            self.load()

        new_samples = np.frombuffer(raw_bytes, dtype=np.int16)
        if len(self.buffer) > 0:
            self.buffer = np.concatenate((self.buffer, new_samples))
        else:
            self.buffer = new_samples

        while len(self.buffer) >= 1280:
            chunk = self.buffer[:1280]
            self.buffer = self.buffer[1280:]

            predictions = self.model.predict(chunk)
            for k, score in predictions.items():
                if "jarvis" in k.lower():
                    self.last_score = float(score)
                    if score >= self.threshold:
                        saved_score = float(score)
                        self.reset()
                        self.last_score = saved_score
                        return True

        return False
