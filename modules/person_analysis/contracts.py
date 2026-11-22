"""Typed contracts for person-analysis inference results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """An XYXY bounding box in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class PersonDetection:
    """One retained COCO person detection."""

    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Person detections and an annotated copy for one image."""

    detections: tuple[PersonDetection, ...]
    annotated_image: np.ndarray = field(repr=False, compare=False)
    inference_metadata: Mapping[str, object]
    source_shape: tuple[int, int, int]

    @property
    def person_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class VideoFrameResult:
    """Person detections for one processed video frame."""

    frame_index: int
    detections: tuple[PersonDetection, ...]

    @property
    def person_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Annotated video and per-frame results; no tracking is implied."""

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

