"""Typed inference results; unavailable drowsiness is never reported as a zero count."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from modules.person_analysis.contracts import PersonDetection

DrowsinessState = Literal["alert", "drowsy", "sleeping", "unknown"]
DROWSY_STATES = frozenset({"drowsy", "sleeping"})


@dataclass(frozen=True)
class PersonAssessment:
    """A person detection and its model-supported state, if one is available."""

    detection: PersonDetection
    status: DrowsinessState = "unknown"
    track_id: int | None = None
    status_confidence: float | None = None
    evidence_window: tuple[float, float] | None = None
    status_model_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"alert", "drowsy", "sleeping", "unknown"}:
            raise ValueError(f"Unsupported drowsiness status: {self.status!r}.")
        if self.track_id is not None and (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 1
        ):
            raise ValueError("Track IDs must be positive integers.")
        if self.status_confidence is not None:
            if (
                isinstance(self.status_confidence, bool)
                or not isinstance(self.status_confidence, (int, float))
                or not math.isfinite(self.status_confidence)
                or not 0.0 <= self.status_confidence <= 1.0
            ):
                raise ValueError("Drowsiness confidence must be between 0 and 1.")
        if self.status != "unknown" and (
            not self.status_model_id or self.status_confidence is None
        ):
            raise ValueError(
                "A non-unknown state requires a model ID and confidence score."
            )
        if self.status in DROWSY_STATES and self.evidence_window is None:
            raise ValueError("A drowsy or sleeping state requires a temporal evidence window.")
        if self.evidence_window is not None:
            if not isinstance(self.evidence_window, tuple) or len(self.evidence_window) != 2:
                raise ValueError("Evidence windows must contain start and end timestamps.")
            start, end = self.evidence_window
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError("Evidence windows must be finite, non-negative, and non-empty.")


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Person detections and annotated image; state analysis may be unavailable."""

    people: tuple[PersonAssessment, ...]
    source_image: np.ndarray = field(repr=False, compare=False)
    annotated_image: np.ndarray = field(repr=False, compare=False)
    inference_metadata: Mapping[str, object]
    drowsiness_model_available: bool
    quality_warnings: tuple[str, ...] = ()

    @property
    def person_count(self) -> int:
        return len(self.people)

    @property
    def sleeping_drowsy_count(self) -> int | None:
        if not self.drowsiness_model_available:
            return None
        return sum(person.status in DROWSY_STATES for person in self.people)


@dataclass(frozen=True)
class VideoFrameResult:
    """Frame-ordered person assessments and source timestamp."""

    frame_index: int
    timestamp_seconds: float | None
    people: tuple[PersonAssessment, ...]
    drowsiness_model_available: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("Video frame index must be a non-negative integer.")
        if self.frame_index < 0:
            raise ValueError("Video frame index must be a non-negative integer.")
        if self.timestamp_seconds is not None and (
            isinstance(self.timestamp_seconds, bool)
            or not isinstance(self.timestamp_seconds, (int, float))
            or not math.isfinite(self.timestamp_seconds)
            or self.timestamp_seconds < 0
        ):
            raise ValueError("Video timestamps must be finite and non-negative.")

    @property
    def person_count(self) -> int:
        return len(self.people)

    @property
    def sleeping_drowsy_count(self) -> int | None:
        if not self.drowsiness_model_available:
            return None
        return sum(person.status in DROWSY_STATES for person in self.people)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Annotated video and per-frame results; tracklets are not verified identities."""

    output_video_path: Path
    frame_count: int
    source_fps: float | None
    output_fps: float
    frames: tuple[VideoFrameResult, ...]
    unique_tracklet_count: int
    temporal_window_seconds: float
    drowsiness_model_available: bool
    warnings: tuple[str, ...]
    inference_metadata: Mapping[str, object]
    frame_limit_reached: bool

    @property
    def sleeping_drowsy_count(self) -> int | None:
        return None
