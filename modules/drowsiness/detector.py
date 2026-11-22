"""Real local person detection with drowsiness analysis explicitly unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from modules.person_analysis.contracts import PersonDetection
from modules.person_analysis.detector import PersonYoloDetector

from .config import DrowsinessConfig
from .contracts import ImageAnalysisResult, PersonAssessment
from .errors import DrowsinessInferenceError
from .preprocessing import annotate_people, decode_image, image_quality_warnings, validate_image


class PersonDetector(Protocol):
    """Interface implemented by the existing local YOLO person detector."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[PersonDetection, ...]: ...


class DrowsinessPipeline:
    """Detect people and return unknown states until a validated model is configured."""

    def __init__(
        self,
        config: DrowsinessConfig | None = None,
        person_detector: PersonDetector | None = None,
    ) -> None:
        self.config = config or DrowsinessConfig()
        self.person_detector = (
            person_detector
            if person_detector is not None
            else PersonYoloDetector(self.config.person_model_path)
        )
        self.drowsiness_model_available = False
        self.age_model_available = False

    def infer_image(
        self,
        image_source: Path | str | bytes | bytearray | memoryview,
    ) -> ImageAnalysisResult:
        """Run real person detection; never infer state from one still image."""
        image_bgr = decode_image(image_source)
        detections = self.detect_people(image_bgr)
        people = tuple(PersonAssessment(detection) for detection in detections)
        metadata = self.metadata
        metadata.update(
            {
                "count_policy": "person detections in this image",
                "drowsiness_inference": "unavailable; no validated drowsiness model is configured",
                "age_estimation": "unavailable; no age model is configured",
            }
        )
        return ImageAnalysisResult(
            people=people,
            source_image=image_bgr.copy(),
            annotated_image=annotate_people(image_bgr, people),
            inference_metadata=metadata,
            drowsiness_model_available=False,
            quality_warnings=image_quality_warnings(image_bgr),
        )

    def detect_people(self, image_bgr: np.ndarray) -> tuple[PersonDetection, ...]:
        """Run the configured person detector on a validated BGR frame."""
        validate_image(image_bgr)
        try:
            return self.person_detector.detect(
                image_bgr,
                self.config.confidence_threshold,
            )
        except Exception as exc:
            if isinstance(exc, DrowsinessInferenceError):
                raise
            raise DrowsinessInferenceError(f"Person detection failed: {exc}") from exc

    def assess_frame(
        self,
        image_bgr: np.ndarray,
        track_ids: tuple[int, ...] | None = None,
    ) -> tuple[PersonAssessment, ...]:
        """Return unknown status for detections until a real state model is available."""
        detections = self.detect_people(image_bgr)
        if track_ids is not None and len(track_ids) != len(detections):
            raise ValueError("One track ID is required for each person detection.")
        return tuple(
            PersonAssessment(
                detection=detection,
                track_id=None if track_ids is None else track_ids[index],
            )
            for index, detection in enumerate(detections)
        )

    @property
    def metadata(self) -> dict[str, object]:
        try:
            metadata = dict(self.person_detector.metadata)
        except Exception as exc:
            raise DrowsinessInferenceError(
                f"Could not read person-detector metadata: {exc}"
            ) from exc
        metadata.update(
            {
                "drowsiness_model": None,
                "drowsiness_model_available": False,
                "age_model": None,
                "age_model_available": False,
                "temporal_window_seconds": self.config.temporal_window_seconds,
            }
        )
        return metadata
