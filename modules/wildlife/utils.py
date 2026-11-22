"""Image validation, detection summaries, and wildlife annotation helpers."""

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
    DetectionSummary,
    WildlifeDetection,
)
from .errors import MediaInputError

IMAGE_FORMAT_EXTENSIONS = {
    "JPEG": (".jpg", ".jpeg"),
    "PNG": (".png",),
    "WEBP": (".webp",),
}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

CARNIVORE_BOX_COLOUR = (0, 0, 255)
NON_CARNIVORE_BOX_COLOUR = (255, 0, 0)
UNKNOWN_BOX_COLOUR = (128, 128, 128)


def supported_image_extensions() -> tuple[str, ...]:
    """Return common image formats for which Pillow has a decoder."""
    Image.init()
    available_formats = set(Image.OPEN)
    return tuple(
        sorted(
            extension
            for image_format, extensions in IMAGE_FORMAT_EXTENSIONS.items()
            if image_format in available_formats
            for extension in extensions
        )
    )


def supported_video_extensions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return candidate video formats and the OpenCV backend names."""
    try:
        available_backends = cv2.videoio_registry.getBackends()
        backend_names = tuple(
            cv2.videoio_registry.getBackendName(backend) for backend in available_backends
        )
    except cv2.error:
        return (), ()
    backend_ids = set(available_backends)
    supported = set()
    if {cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_AVFOUNDATION} & backend_ids:
        supported.update(VIDEO_EXTENSIONS)
    return tuple(extension for extension in VIDEO_EXTENSIONS if extension in supported), backend_names


def decode_image(source: Path | str | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a supported image source into an OpenCV BGR array."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in supported_image_extensions():
            raise MediaInputError(f"Unsupported image format: {path.suffix or 'no extension'}.")
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            raise MediaInputError(f"Could not read image file '{path}': {exc}") from exc
    else:
        try:
            image_bytes = bytes(source)
        except (TypeError, ValueError) as exc:
            raise MediaInputError("The supplied image data is invalid.") from exc
    if not image_bytes:
        raise MediaInputError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in IMAGE_FORMAT_EXTENSIONS:
                raise MediaInputError(
                    f"Unsupported image format: {image.format or 'unknown'}."
                )
            image.load()
            rgb_image = image.convert("RGB")
    except MediaInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaInputError("The image is malformed or cannot be decoded.") from exc
    return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)


def box_colour(classification: str) -> tuple[int, int, int]:
    """Return BGR red/blue/neutral for carnivore/non-carnivore/unknown."""
    if classification == "carnivore":
        return CARNIVORE_BOX_COLOUR
    if classification == "non_carnivore":
        return NON_CARNIVORE_BOX_COLOUR
    if classification == "unknown":
        return UNKNOWN_BOX_COLOUR
    raise ValueError(f"Unsupported wildlife classification: {classification}")


def summarize_detections(detections: Sequence[WildlifeDetection]) -> DetectionSummary:
    """Aggregate supplied real detection records without deduplicating them."""
    species_counts: dict[str, int] = {}
    carnivore_species_counts: dict[str, int] = {}
    carnivores = 0
    non_carnivores = 0
    unknown = 0
    for detection in detections:
        species_counts[detection.species] = species_counts.get(detection.species, 0) + 1
        if detection.classification == "carnivore":
            carnivores += 1
            carnivore_species_counts[detection.species] = (
                carnivore_species_counts.get(detection.species, 0) + 1
            )
        elif detection.classification == "non_carnivore":
            non_carnivores += 1
        elif detection.classification == "unknown":
            unknown += 1
        else:
            raise ValueError(
                f"Unsupported wildlife classification: {detection.classification}"
            )
    confidence = (
        sum(detection.confidence for detection in detections) / len(detections)
        if detections
        else None
    )
    return DetectionSummary(
        total_animals=len(detections),
        carnivores=carnivores,
        non_carnivores=non_carnivores,
        unknown=unknown,
        species_counts=dict(sorted(species_counts.items())),
        carnivore_species_counts=dict(sorted(carnivore_species_counts.items())),
        average_confidence=confidence,
    )


def annotate_detections(
    image_bgr: np.ndarray,
    detections: Sequence[WildlifeDetection],
) -> np.ndarray:
    """Return an annotated image copy using category-specific box colours."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("Annotation requires a non-empty BGR image.")
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        clipped = clip_box(detection.bounding_box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in clipped.as_xyxy())
        colour = box_colour(detection.classification)
        thickness = max(1, round(min(height, width) / 400))
        label = f"{detection.species} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness
        )
        label_y = y1 - 5 if y1 - text_height - baseline - 5 >= 0 else min(
            height - 1, y1 + text_height + baseline + 5
        )
        cv2.rectangle(
            annotated,
            (x1, y1),
            (min(width - 1, x1 + text_width + 6), min(height - 1, label_y + 4)),
            colour,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
    return annotated


def clip_box(
    box: BoundingBox, image_width: int, image_height: int
) -> BoundingBox | None:
    """Clip a finite XYXY box to an image and reject empty areas."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not all(math.isfinite(value) for value in box.as_xyxy()):
        return None
    x1 = max(0, min(image_width - 1, math.floor(box.x1)))
    y1 = max(0, min(image_height - 1, math.floor(box.y1)))
    x2 = max(0, min(image_width - 1, math.ceil(box.x2)))
    y2 = max(0, min(image_height - 1, math.ceil(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1, y1, x2, y2)
