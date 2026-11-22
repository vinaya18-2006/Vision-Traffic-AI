"""Image validation, MediaPipe landmark formatting, and result annotation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from modules.car_detection.errors import MediaInputError as CarMediaInputError
from modules.car_detection.utils import (
    decode_image as decode_car_image,
    supported_image_extensions,
    supported_video_extensions,
)

from .contracts import BoundingBox, HandObservation, SignPrediction
from .errors import LowQualityInputError, MediaInputError

MIN_IMAGE_SIDE = 64


def load_vocabulary(path: Path | str | None = None) -> tuple[str, ...]:
    """Load the checkpoint-matched static ASL label order."""
    vocabulary_path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("vocabulary.json")
    )
    try:
        payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        labels = tuple(payload["labels"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MediaInputError(
            f"Could not read the sign-language vocabulary at '{vocabulary_path}': {exc}"
        ) from exc
    if (
        len(labels) != 24
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or len(label) != 1 for label in labels)
    ):
        raise MediaInputError(
            f"The vocabulary at '{vocabulary_path}' must contain 24 unique letters."
        )
    return labels


def decode_image(source: Path | str | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a supported image and expose failures as Task 5 media errors."""
    try:
        image_bgr = decode_car_image(source)
    except CarMediaInputError as exc:
        raise MediaInputError(str(exc)) from exc
    validate_image(image_bgr)
    return image_bgr


def validate_image(image_bgr: np.ndarray) -> None:
    """Reject malformed, tiny, or visually degenerate input before hand inference."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
        or image_bgr.dtype != np.uint8
    ):
        raise MediaInputError("Expected a non-empty 8-bit BGR image.")
    height, width = image_bgr.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        raise LowQualityInputError(
            f"Image is too small for hand-landmark extraction; both sides must be at "
            f"least {MIN_IMAGE_SIDE} pixels."
        )
    if float(np.std(image_bgr)) < 1.0:
        raise LowQualityInputError(
            "Image has too little visual detail for reliable hand-landmark extraction."
        )


def landmark_vector(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Validate 21 MediaPipe xyz points and flatten them in the model's declared order."""
    points = np.asarray(values, dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise MediaInputError("Hand landmarks must contain 21 finite x/y/z points.")
    return points.reshape(63)


def annotate_prediction(
    image_bgr: np.ndarray,
    prediction: SignPrediction,
) -> np.ndarray:
    """Return a copy with the hand region and bounded recognition status marked."""
    annotated = image_bgr.copy()
    if prediction.bounding_box is not None:
        box = prediction.bounding_box
        height, width = image_bgr.shape[:2]
        x1 = max(0, min(width - 1, math.floor(box.x1)))
        y1 = max(0, min(height - 1, math.floor(box.y1)))
        x2 = max(0, min(width - 1, math.ceil(box.x2)))
        y2 = max(0, min(height - 1, math.ceil(box.y2)))
        colour = (0, 180, 0) if prediction.status == "recognized" else (0, 190, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        label = prediction.label or "Unknown"
        if prediction.confidence is not None:
            label = f"{label} {prediction.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    elif prediction.status == "no_hand":
        cv2.putText(
            annotated,
            "No hand detected",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


__all__ = [
    "MIN_IMAGE_SIDE",
    "annotate_prediction",
    "decode_image",
    "landmark_vector",
    "load_vocabulary",
    "supported_image_extensions",
    "supported_video_extensions",
    "validate_image",
]
