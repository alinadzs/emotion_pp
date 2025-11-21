from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import requests
import tensorflow as tf  # noqa: F401 (ensures tensorflow.keras is registered)
from deepface import DeepFace

EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def _load_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Empty payload")
    array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image")
    return frame


def _normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for emotion, score in emotions.items():
        value = float(score)
        if value > 1:
            value /= 100.0
        cleaned[emotion] = max(0.0, min(1.0, value))
    total = sum(cleaned.values())
    if total > 0:
        return {emotion: value / total for emotion, value in cleaned.items()}
    return {emotion: (1.0 / len(EMOTIONS)) for emotion in EMOTIONS}


@dataclass
class EmotionRecognizer:
    detector_backend: str = "opencv"
    _weights: List[tuple[str, str]] = (
        (
            "facial_expression_model_weights.h5",
            "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
        ),
    )

    def __post_init__(self) -> None:
        self._ensure_weights()
        self._model = DeepFace

    def _ensure_weights(self) -> None:
        weights_dir = Path.home() / ".deepface" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in self._weights:
            target = weights_dir / filename
            if target.exists():
                continue
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)

    def analyze(self, image_bytes: bytes, detect_face: bool = True) -> Dict[str, float]:
        frame = _load_image(image_bytes)
        result = self._model.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=detect_face,
            detector_backend=self.detector_backend,
            align=True,
            silent=True,
        )
        record = result[0] if isinstance(result, list) else result
        emotions = record.get("emotion") or {}
        if not emotions:
            return {"dominant": "neutral", "confidence": 1.0, "emotions": {"neutral": 1.0}}

        normalized = _normalize_emotions(
            {emotion: emotions.get(emotion, 0.0) for emotion in EMOTIONS}
        )
        dominant = max(normalized, key=normalized.get)
        return {
            "dominant": dominant,
            "confidence": normalized[dominant],
            "emotions": normalized,
        }


recognizer = EmotionRecognizer()


class MemeStore:
    def __init__(self, base_dir: str = "memes") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def list_emotions(self) -> List[str]:
        return EMOTIONS

    def has_meme(self, emotion: str) -> bool:
        folder = self.base_path / emotion
        return folder.exists() and any(folder.iterdir())

    def pick_random(self, emotion: str) -> Path | None:
        folder = self.base_path / emotion
        if not folder.exists():
            return None
        files = [item for item in folder.iterdir() if item.is_file()]
        if not files and emotion != "neutral":
            return self.pick_random("neutral")
        return random.choice(files) if files else None


memes = MemeStore()