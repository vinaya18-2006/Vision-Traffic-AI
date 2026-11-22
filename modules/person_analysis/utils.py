"""Image decoding and person-box annotation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from modules.car_detection.utils import (
    clip_box,
    decode_image as decode_car_image,
    supported_image_extensions,
    supported_video_extensions,
)

from .contracts import BoundingBox, PersonDetection
from .errors import MediaInputError
from modules.car_detection.errors import MediaInputError as CarMediaInputError

PERSON_BOX_COLOUR = (0, 200, 0)


def decode_image(source: Path | str | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode an image and expose media failures through the Task 4 error contract."""
    try:
        return decode_car_image(source)
    except CarMediaInputError as exc:
        raise MediaInputError(str(exc)) from exc


def annotate_people(
    image_bgr: np.ndarray,
    detections: Sequence[PersonDetection],
) -> np.ndarray:
    """Return an annotated copy, clipping valid boxes to the source image."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("Annotation requires a non-empty BGR image.")

    annotated = image_bgr.copy()
    height, width = image_bgr.shape[:2]
    for detection in detections:
        box = clip_box(detection.bounding_box, width, height)
        if box is None or not math.isfinite(detection.confidence):
            continue
        x1, y1, x2, y2 = (int(value) for value in box.as_xyxy())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), PERSON_BOX_COLOUR, 2)
        label = f"person {detection.confidence:.2f}"
        text_y = max(14, y1 - 6)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            PERSON_BOX_COLOUR,
            1,
            cv2.LINE_AA,
        )
    return annotated


def bounding_box_from_values(values: Sequence[float]) -> BoundingBox | None:
    """Convert finite XYXY model coordinates into a valid box or return None."""
    if len(values) != 4:
        return None
    coordinates = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in coordinates):
        return None
    box = BoundingBox(*coordinates)
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return None
    return box


__all__ = [
    "PERSON_BOX_COLOUR",
    "annotate_people",
    "bounding_box_from_values",
    "decode_image",
    "supported_image_extensions",
    "supported_video_extensions",
]
