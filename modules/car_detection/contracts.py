"""Typed result contracts for Task 1 car and person detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

ColourLabel = Literal[
    "blue",
    "red",
    "white",
    "black",
    "gray",
    "silver",
    "green",
    "yellow",
    "orange",
    "brown",
    "other",
    "unknown",
]


@dataclass(frozen=True)
class BoundingBox:
    """An XYXY bounding box in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Return the coordinates in XYXY order."""
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class ColourEstimate:
    """A crop-based apparent-colour label and its evidence score."""

    label: ColourLabel
    confidence: float
    reason: str | None = None


@dataclass(frozen=True)
class ObjectDetection:
    """An object detector's class, confidence, and XYXY box."""

    class_name: Literal["car", "person"]
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class CarDetection:
    """A detected car and its independently estimated apparent body colour."""

    confidence: float
    bounding_box: BoundingBox
    estimated_colour: ColourEstimate
    class_name: Literal["car"] = "car"


@dataclass(frozen=True)
class PersonDetection:
    """A detected person."""

    confidence: float
    bounding_box: BoundingBox
    class_name: Literal["person"] = "person"


@dataclass(frozen=True)
class DetectionCounts:
    """Detection totals for one image or video frame."""

    car_count: int
    person_count: int


@dataclass(frozen=True)
class ImageInferenceResult:
    """Structured image inference output and the annotated image copy."""

    cars: tuple[CarDetection, ...]
    people: tuple[PersonDetection, ...]
    annotated_image: np.ndarray = field(repr=False, compare=False)
    inference_metadata: Mapping[str, object]
    source_shape: tuple[int, int, int]

    @property
    def car_count(self) -> int:
        """Return the number of retained car detections in this image."""
        return len(self.cars)

    @property
    def person_count(self) -> int:
        """Return the number of retained person detections in this image."""
        return len(self.people)


@dataclass(frozen=True)
class VideoFrameResult:
    """Detections for one frame; these are not unique tracked objects."""

    frame_index: int
    cars: tuple[CarDetection, ...]
    people: tuple[PersonDetection, ...]

    @property
    def car_count(self) -> int:
        """Return this frame's retained car detections."""
        return len(self.cars)

    @property
    def person_count(self) -> int:
        """Return this frame's retained person detections."""
        return len(self.people)

    @property
    def blue_car_count(self) -> int:
        """Return the blue-car detections in this frame."""
        return sum(car.estimated_colour.label == "blue" for car in self.cars)

    @property
    def other_known_colour_car_count(self) -> int:
        """Return known non-blue car detections in this frame."""
        return sum(
            car.estimated_colour.label not in {"blue", "unknown"} for car in self.cars
        )


@dataclass(frozen=True)
class VideoInferenceResult:
    """Annotated video output with per-frame counts and measured FPS."""

    output_video_path: Path
    frame_count: int
    source_fps: float | None
    output_fps: float
    processing_fps: float
    frames: tuple[VideoFrameResult, ...]
    inference_metadata: Mapping[str, object]
    original_preview_frame: np.ndarray = field(repr=False, compare=False)
    annotated_preview_frame: np.ndarray = field(repr=False, compare=False)
    max_frames: int
    frame_limit_reached: bool
    unique_object_counts: None = None
