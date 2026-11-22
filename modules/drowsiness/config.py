"""Validated person-detector and temporal-window settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from modules.person_analysis.detector import default_model_path


@dataclass(frozen=True)
class DrowsinessConfig:
    """Settings for real person detection and a future validated temporal model."""

    person_model_path: Path | None = None
    confidence_threshold: float = 0.25
    temporal_window_seconds: float = 10.0
    max_video_frames: int = 3000
    tracker_iou_threshold: float = 0.3
    tracker_max_gap_frames: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.confidence_threshold, bool) or not isinstance(
            self.confidence_threshold, (int, float)
        ):
            raise ValueError("Person confidence threshold must be between 0 and 1.")
        if not math.isfinite(self.confidence_threshold) or not (
            0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("Person confidence threshold must be between 0 and 1.")
        if isinstance(self.temporal_window_seconds, bool) or not isinstance(
            self.temporal_window_seconds, (int, float)
        ):
            raise ValueError("Temporal window must be a positive number of seconds.")
        if not math.isfinite(self.temporal_window_seconds) or not (
            1.0 <= self.temporal_window_seconds <= 60.0
        ):
            raise ValueError("Temporal window must be between 1 and 60 seconds.")
        if (
            isinstance(self.max_video_frames, bool)
            or not isinstance(self.max_video_frames, int)
            or self.max_video_frames < 1
        ):
            raise ValueError("Maximum video frame count must be a positive integer.")
        if isinstance(self.tracker_iou_threshold, bool) or not isinstance(
            self.tracker_iou_threshold, (int, float)
        ):
            raise ValueError("Tracker IoU threshold must be between 0 and 1.")
        if not math.isfinite(self.tracker_iou_threshold) or not (
            0.0 <= self.tracker_iou_threshold <= 1.0
        ):
            raise ValueError("Tracker IoU threshold must be between 0 and 1.")
        if (
            isinstance(self.tracker_max_gap_frames, bool)
            or not isinstance(self.tracker_max_gap_frames, int)
            or self.tracker_max_gap_frames < 0
        ):
            raise ValueError("Tracker maximum gap must be a non-negative integer.")
        model_path = (
            self.person_model_path
            if self.person_model_path is not None
            else default_model_path()
        )
        object.__setattr__(self, "person_model_path", Path(model_path).expanduser().resolve())
