"""Image decoding, safe crop helpers, and annotation utilities."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .contracts import (
    BoundingBox,
    CarDetection,
    ColourLabel,
    DetectionCounts,
    PersonDetection,
)
from .errors import MediaInputError

IMAGE_FORMAT_EXTENSIONS = {
    "JPEG": (".jpg", ".jpeg"),
    "PNG": (".png",),
    "WEBP": (".webp",),
}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
KNOWN_COLOUR_LABELS = frozenset(
    {"red", "white", "black", "gray", "silver", "green", "yellow", "orange", "brown", "other"}
)

PERSON_BOX_COLOUR = (0, 200, 0)
UNKNOWN_CAR_BOX_COLOUR = (0, 215, 255)
BLUE_CAR_BOX_COLOUR = (0, 0, 255)
OTHER_KNOWN_CAR_BOX_COLOUR = (255, 0, 0)


def supported_image_extensions() -> tuple[str, ...]:
    """Return Task 1 image extensions with an installed Pillow decoder."""
    Image.init()
    available_formats = set(Image.OPEN)
    extensions = {
        extension
        for image_format, format_extensions in IMAGE_FORMAT_EXTENSIONS.items()
        if image_format in available_formats
        for extension in format_extensions
    }
    return tuple(sorted(extensions))


def supported_video_extensions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return candidate video containers and the installed OpenCV backends."""
    available_backends = cv2.videoio_registry.getBackends()
    backend_names = tuple(
        cv2.videoio_registry.getBackendName(backend) for backend in available_backends
    )
    backend_ids = set(available_backends)
    supported = set()
    if {cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_AVFOUNDATION} & backend_ids:
        supported.update(VIDEO_EXTENSIONS)
    return tuple(extension for extension in VIDEO_EXTENSIONS if extension in supported), backend_names


def decode_image(source: Path | str | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a supported image source as an OpenCV-compatible BGR array."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in supported_image_extensions():
            raise MediaInputError(f"Unsupported image format: {path.suffix or 'no extension'}.")
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            raise MediaInputError(f"Could not read image file '{path}': {exc}") from exc
    else:
        image_bytes = bytes(source)

    if not image_bytes:
        raise MediaInputError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            if image_format not in IMAGE_FORMAT_EXTENSIONS:
                raise MediaInputError(f"Unsupported image format: {image_format or 'unknown'}.")
            image.load()
            rgb_image = image.convert("RGB")
    except MediaInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaInputError("The image is malformed or cannot be decoded.") from exc
    return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)


def clip_box(box: BoundingBox, image_width: int, image_height: int) -> BoundingBox | None:
    """Clip an XYXY box to image bounds, returning None for an empty crop."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not all(math.isfinite(value) for value in box.as_xyxy()):
        return None
    x1 = max(0, min(image_width, math.floor(box.x1)))
    y1 = max(0, min(image_height, math.floor(box.y1)))
    x2 = max(0, min(image_width, math.ceil(box.x2)))
    y2 = max(0, min(image_height, math.ceil(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2, y2)


def crop_box(
    image_bgr: np.ndarray,
    box: BoundingBox,
    minimum_size: int = 2,
) -> np.ndarray | None:
    """Safely crop a clipped box, returning None if it is too small."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("Crop source must be a non-empty BGR image.")
    if minimum_size < 1:
        raise ValueError("Minimum crop size must be at least one pixel.")
    height, width = image_bgr.shape[:2]
    clipped = clip_box(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in clipped.as_xyxy())
    crop = image_bgr[y1:y2, x1:x2]
    if crop.shape[0] < minimum_size or crop.shape[1] < minimum_size:
        return None
    return crop


def car_box_colour(colour_label: ColourLabel) -> tuple[int, int, int]:
    """Return the BGR annotation colour for a car's apparent colour."""
    if colour_label == "blue":
        return BLUE_CAR_BOX_COLOUR
    if colour_label == "unknown":
        return UNKNOWN_CAR_BOX_COLOUR
    if colour_label in KNOWN_COLOUR_LABELS:
        return OTHER_KNOWN_CAR_BOX_COLOUR
    raise ValueError(f"Unsupported car colour label: {colour_label}")


def count_detections(
    cars: Sequence[CarDetection],
    people: Sequence[PersonDetection],
) -> DetectionCounts:
    """Count retained detections for one image or one video frame."""
    return DetectionCounts(car_count=len(cars), person_count=len(people))


def annotate_detections(
    image_bgr: np.ndarray,
    cars: Sequence[CarDetection],
    people: Sequence[PersonDetection],
) -> np.ndarray:
    """Return an annotated copy with distinct car and person box styles."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("Annotation requires a non-empty BGR image.")
    annotated = image_bgr.copy()
    labels: list[tuple[BoundingBox, tuple[int, int, int], str]] = []
    for detection in cars:
        colour = car_box_colour(detection.estimated_colour.label)
        label = (
            f"car {detection.confidence:.2f} | {detection.estimated_colour.label} "
            f"{detection.estimated_colour.confidence:.2f}"
        )
        labels.append((detection.bounding_box, colour, label))
    for detection in people:
        label = f"person {detection.confidence:.2f}"
        labels.append((detection.bounding_box, PERSON_BOX_COLOUR, label))
    for box, colour, label in labels:
        _draw_label(annotated, box, colour, label)
    for box, colour, _ in labels:
        _draw_box(annotated, box, colour)
    return annotated


def _draw_box(
    image: np.ndarray,
    box: BoundingBox,
    colour: tuple[int, int, int],
) -> None:
    clipped = clip_box(box, image.shape[1], image.shape[0])
    if clipped is None:
        return
    x1, y1, x2, y2 = (int(value) for value in clipped.as_xyxy())
    thickness = max(1, round(min(image.shape[:2]) / 400))
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)


def _draw_label(
    image: np.ndarray,
    box: BoundingBox,
    colour: tuple[int, int, int],
    label: str,
) -> None:
    clipped = clip_box(box, image.shape[1], image.shape[0])
    if clipped is None:
        return
    x1, y1, _, _ = (int(value) for value in clipped.as_xyxy())
    thickness = max(1, round(min(image.shape[:2]) / 400))
    font_scale = max(0.4, min(image.shape[:2]) / 1200)
    text_thickness = max(1, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    label_y = y1 - 5 if y1 - text_height - baseline - 5 >= 0 else min(
        image.shape[0] - 1, y1 + text_height + baseline + 5
    )
    background_top = max(0, label_y - text_height - baseline - 3)
    background_bottom = min(image.shape[0], label_y + baseline + 3)
    background_right = min(image.shape[1], x1 + text_width + 6)
    cv2.rectangle(
        image,
        (x1, background_top),
        (background_right, background_bottom),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )