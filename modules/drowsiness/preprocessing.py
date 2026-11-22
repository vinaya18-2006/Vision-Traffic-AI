"""Image validation and status-aware annotations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from modules.car_detection.errors import MediaInputError as CarMediaInputError
from modules.car_detection.utils import decode_image as decode_car_image
from modules.person_analysis.contracts import BoundingBox

from .contracts import DROWSY_STATES, PersonAssessment
from .errors import DrowsinessMediaError

UNKNOWN_BOX_COLOUR = (0, 190, 255)
ALERT_BOX_COLOUR = (0, 180, 0)
RED_BOX_COLOUR = (0, 0, 255)
MIN_IMAGE_SIDE = 32


def decode_image(source: Path | str | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a supported image and expose failures as drowsiness media errors."""
    try:
        image_bgr = decode_car_image(source)
    except CarMediaInputError as exc:
        raise DrowsinessMediaError(str(exc)) from exc
    validate_image(image_bgr)
    return image_bgr


def validate_image(image_bgr: np.ndarray) -> None:
    """Reject malformed or too-small BGR images."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
        or image_bgr.dtype != np.uint8
    ):
        raise DrowsinessMediaError("Expected a non-empty 8-bit BGR image.")
    if min(image_bgr.shape[:2]) < MIN_IMAGE_SIDE:
        raise DrowsinessMediaError(
            f"Image dimensions must be at least {MIN_IMAGE_SIDE} by {MIN_IMAGE_SIDE} pixels."
        )


def image_quality_warnings(image_bgr: np.ndarray) -> tuple[str, ...]:
    """Return a conservative lighting notice without treating it as a model prediction."""
    if float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))) < 20.0:
        return ("The image is very dark; person detection and visible cues may be unreliable.",)
    return ()


def annotation_colour(person: PersonAssessment) -> tuple[int, int, int]:
    """Use red only for model-backed positive states with temporal evidence."""
    if person.status in DROWSY_STATES:
        if (
            not person.status_model_id
            or person.status_confidence is None
            or person.evidence_window is None
        ):
            raise ValueError("A red box requires model-backed temporal drowsiness evidence.")
        return RED_BOX_COLOUR
    if person.status == "alert":
        if not person.status_model_id or person.status_confidence is None:
            raise ValueError("An alert label requires a model ID and confidence score.")
        return ALERT_BOX_COLOUR
    if person.status == "unknown":
        return UNKNOWN_BOX_COLOUR
    raise ValueError(f"Unsupported drowsiness status: {person.status!r}.")


def annotate_people(
    image_bgr: np.ndarray,
    people: Sequence[PersonAssessment],
) -> np.ndarray:
    """Return an annotated copy; unavailable state is yellow, not a red warning."""
    validate_image(image_bgr)
    annotated = image_bgr.copy()
    height, width = image_bgr.shape[:2]
    for person in people:
        box = person.detection.bounding_box
        x1 = max(0, min(width - 1, math.floor(box.x1)))
        y1 = max(0, min(height - 1, math.floor(box.y1)))
        x2 = max(0, min(width - 1, math.ceil(box.x2)))
        y2 = max(0, min(height - 1, math.ceil(box.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        colour = annotation_colour(person)
        label = f"person {person.detection.confidence:.2f} | {person.status}"
        if person.track_id is not None:
            label = f"#{person.track_id} " + label
        if person.status_confidence is not None:
            label += f" {person.status_confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
            cv2.LINE_AA,
        )
    return annotated


__all__ = [
    "ALERT_BOX_COLOUR",
    "MIN_IMAGE_SIDE",
    "RED_BOX_COLOUR",
    "UNKNOWN_BOX_COLOUR",
    "annotate_people",
    "annotation_colour",
    "decode_image",
    "image_quality_warnings",
    "validate_image",
]
