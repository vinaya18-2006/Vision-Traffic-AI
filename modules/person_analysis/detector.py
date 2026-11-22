"""Local YOLO11n person detection and image-level inference."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from modules.car_detection.detector import default_model_path as task1_model_path

from .contracts import (
    BoundingBox,
    ImageAnalysisResult,
    PersonDetection,
)
from .errors import InferenceError, MediaInputError, ModelLoadError
from .utils import annotate_people, bounding_box_from_values, decode_image

PERSON_CLASS_ID = 0
MODEL_FILENAME = "yolo11n.pt"


def default_model_path() -> Path:
    """Use a Task 4 override or the existing local Task 1 YOLO11n path."""
    configured = os.environ.get("PERSON_ANALYSIS_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return task1_model_path()


class PersonDetector(Protocol):
    """Detector interface consumed by the image/video pipelines."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[PersonDetection, ...]: ...


class PersonYoloDetector:
    """Load local YOLO11n weights and request only COCO class 0 (person)."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else default_model_path()
        )
        if not self.model_path.is_file():
            raise ModelLoadError(
                f"Person detector weights were not found at '{self.model_path}'. "
                f"Place the existing {MODEL_FILENAME} file there, set "
                "PERSON_ANALYSIS_MODEL_PATH, or configure CAR_DETECTION_MODEL_PATH. "
                "No model will be downloaded automatically."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load person detector weights from '{self.model_path}': {exc}"
            ) from exc

        names = self._model.names
        try:
            person_name = names[PERSON_CLASS_ID]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelLoadError(
                "The configured detector has no COCO person class at class ID 0."
            ) from exc
        if str(person_name).strip().lower() != "person":
            raise ModelLoadError(
                "The configured detector class map is incompatible: expected "
                "COCO class 0=person."
            )
        try:
            self._model_sha256 = _sha256_file(self.model_path)
        except OSError as exc:
            raise ModelLoadError(
                f"Could not read person detector weights at '{self.model_path}': {exc}"
            ) from exc

    @property
    def metadata(self) -> Mapping[str, object]:
        import ultralytics

        return {
            "model_name": "YOLO11n",
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
            "ultralytics_version": ultralytics.__version__,
            "class_map": {PERSON_CLASS_ID: "person"},
            "count_policy": "retained person detections per image or video frame",
        }

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[PersonDetection, ...]:
        _validate_confidence_threshold(confidence_threshold)
        try:
            results = self._model.predict(
                source=image_bgr,
                classes=[PERSON_CLASS_ID],
                conf=confidence_threshold,
                verbose=False,
            )
            detections: list[PersonDetection] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                coordinates = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                class_ids = boxes.cls.detach().cpu().tolist()
                for values, raw_confidence, raw_class_id in zip(
                    coordinates, confidences, class_ids, strict=True
                ):
                    if int(raw_class_id) != PERSON_CLASS_ID:
                        continue
                    confidence = float(raw_confidence)
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise InferenceError(
                            "The person detector returned an invalid confidence score."
                        )
                    if confidence < confidence_threshold:
                        continue
                    box = bounding_box_from_values(values)
                    if box is None:
                        continue
                    detections.append(PersonDetection(confidence, box))
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"YOLO11n person inference failed: {exc}") from exc
        return tuple(detections)


class PersonAnalysisPipeline:
    """Validate images and run real local person detection."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.25,
        detector: PersonDetector | None = None,
    ) -> None:
        _validate_confidence_threshold(confidence_threshold)
        if detector is not None and model_path is not None:
            raise ValueError("Pass either a model path or a loaded detector, not both.")
        self.detector = detector if detector is not None else PersonYoloDetector(model_path)
        self.confidence_threshold = confidence_threshold

    def infer_image(
        self,
        image_source: Path | str | bytes | bytearray | memoryview,
    ) -> ImageAnalysisResult:
        image_bgr = decode_image(image_source)
        started = time.perf_counter()
        detections = self.detect_people(image_bgr)
        annotated = annotate_people(image_bgr, detections)
        metadata = dict(self.detector.metadata)
        metadata.update(
            {
                "confidence_threshold": self.confidence_threshold,
                "inference_duration_ms": (time.perf_counter() - started) * 1000.0,
                "score_is_calibrated_probability": False,
            }
        )
        return ImageAnalysisResult(
            detections=detections,
            annotated_image=annotated,
            inference_metadata=metadata,
            source_shape=image_bgr.shape,
        )

    def detect_people(self, image_bgr: np.ndarray) -> tuple[PersonDetection, ...]:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.size == 0
        ):
            raise MediaInputError("Inference requires a non-empty BGR image.")
        return self.detector.detect(image_bgr, self.confidence_threshold)


def _validate_confidence_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Detection confidence threshold must be between 0 and 1.")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Detection confidence threshold must be between 0 and 1.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()

