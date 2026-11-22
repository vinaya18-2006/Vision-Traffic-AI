"""Typed contracts for static ASL sign-recognition results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

SignStatus = Literal["recognized", "unknown", "no_hand"]


@dataclass(frozen=True)
class BoundingBox:
    """XYXY coordinates in source-image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class HandObservation:
    """One hand's 21 MediaPipe xyz landmarks and image-space box."""

    landmarks: np.ndarray = field(repr=False, compare=False)
    bounding_box: BoundingBox


@dataclass(frozen=True)
class SignPrediction:
    """One frame-level static letter result; no word or temporal inference is implied."""

    label: str
    confidence: float | None
    status: SignStatus
    bounding_box: BoundingBox | None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    warning: str | None = None


@dataclass(frozen=True)
class ImageRecognitionResult:
    """A frame-level result and annotated copy for one image."""

    prediction: SignPrediction
    annotated_image: np.ndarray = field(repr=False, compare=False)
    source_shape: tuple[int, int, int]
    inference_metadata: Mapping[str, object]


@dataclass(frozen=True)
class VideoFramePrediction:
    """One frame's independent static-letter classification."""

    frame_index: int
    timestamp_seconds: float
    prediction: SignPrediction


@dataclass(frozen=True)
class VideoRecognitionResult:
    """Annotated video with per-frame labels; temporal recognition is not performed."""

    output_video_path: Path
    frame_count: int
    source_fps: float | None
    output_fps: float
    frames: tuple[VideoFramePrediction, ...]
    inference_metadata: Mapping[str, object]
    original_preview_frame: np.ndarray = field(repr=False, compare=False)
    annotated_preview_frame: np.ndarray = field(repr=False, compare=False)
    max_frames: int
    frame_limit_reached: bool
