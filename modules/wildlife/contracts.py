"""Typed inference and analytics contracts for wildlife detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

DietaryCategory = Literal["carnivore", "non_carnivore", "unknown"]


@dataclass(frozen=True)
class BoundingBox:
    """An XYXY bounding box in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Return the box coordinates in XYXY order."""
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class ModelDetection:
    """A detector output before its label is assigned a dietary category."""

    class_name: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class WildlifeDetection:
    """A retained animal model-class detection and taxonomy lookup result."""

    species: str
    confidence: float
    bounding_box: BoundingBox
    classification: DietaryCategory


@dataclass(frozen=True)
class DetectionSummary:
    """Actual detection totals and distributions for an image or frame set."""

    total_animals: int
    carnivores: int
    non_carnivores: int
    unknown: int
    species_counts: Mapping[str, int]
    carnivore_species_counts: Mapping[str, int]
    average_confidence: float | None


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Structured detections, summary, metadata, and annotated BGR image."""

    detections: tuple[WildlifeDetection, ...]
    summary: DetectionSummary
    annotated_image: np.ndarray = field(repr=False, compare=False)
    inference_metadata: Mapping[str, object]
    source_shape: tuple[int, int, int]


@dataclass(frozen=True)
class VideoFrameResult:
    """Retained detections in one frame; no identity tracking is implied."""

    frame_index: int
    detections: tuple[WildlifeDetection, ...]

    @property
    def animal_count(self) -> int:
        """Return the animal detections in this frame."""
        return len(self.detections)

    @property
    def carnivore_count(self) -> int:
        """Return mapped carnivore detections in this frame."""
        return sum(detection.classification == "carnivore" for detection in self.detections)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Annotated video output and analytics for processed frame-level detections."""

    output_video_path: Path
    frame_count: int
    source_fps: float | None
    processing_fps: float
    frames: tuple[VideoFrameResult, ...]
    summary: DetectionSummary
    inference_metadata: Mapping[str, object]
    original_preview_frame: np.ndarray = field(repr=False, compare=False)
    annotated_preview_frame: np.ndarray = field(repr=False, compare=False)
    max_frames: int
    frame_limit_reached: bool
    unique_animal_count: None = None

    @property
    def total_per_frame_detections(self) -> int:
        """Return detections summed over frames, not unique animals."""
        return self.summary.total_animals
