"""Real YOLO model loading and species-label inference for wildlife analysis."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from .contracts import (
    BoundingBox,
    ImageAnalysisResult,
    ModelDetection,
    WildlifeDetection,
)
from .errors import InferenceError, ModelLoadError
from .taxonomy import WildlifeTaxonomy, load_taxonomy
from .utils import annotate_detections, decode_image, summarize_detections

MODEL_FILENAME = "yolo11n.pt"


class WildlifeDetector(Protocol):
    """The detector contract consumed by the inference pipeline."""

    metadata: Mapping[str, object]
    class_names: tuple[str, ...]

    def detect(
        self, image_bgr: np.ndarray, confidence_threshold: float
    ) -> tuple[ModelDetection, ...]: ...


def default_model_path() -> Path:
    """Return the local default YOLO weights path without downloading weights."""
    configured = os.environ.get("WILDLIFE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / MODEL_FILENAME


class WildlifeYoloDetector:
    """Load local Ultralytics weights and expose configured animal class labels."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        animal_classes: Sequence[str] | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else default_model_path()
        )
        if not self.model_path.is_file():
            raise ModelLoadError(
                f"Wildlife model weights were not found at '{self.model_path}'. "
                "Place valid local weights there or set WILDLIFE_MODEL_PATH. "
                "No model will be downloaded automatically."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load wildlife model weights from '{self.model_path}': {exc}"
            ) from exc

        try:
            all_names = _normalize_class_names(self._model.names)
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(f"The configured model has an invalid class map: {exc}") from exc
        configured_labels = {
            _normalize(label) for label in (animal_classes or load_taxonomy().animal_classes)
        }
        self._class_ids = {
            class_id: label
            for class_id, label in all_names.items()
            if _normalize(label) in configured_labels
        }
        if not self._class_ids:
            available = ", ".join(all_names.values())
            raise ModelLoadError(
                "The configured model has no labels matching the configured animal_classes. "
                f"Model labels: {available or 'none'}."
            )
        self.class_names = tuple(self._class_ids.values())
        try:
            self._model_sha256 = _sha256_file(self.model_path)
        except OSError as exc:
            raise ModelLoadError(
                f"Could not read wildlife model weights at '{self.model_path}': {exc}"
            ) from exc

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return the actual local model identity and animal label vocabulary."""
        import ultralytics

        return {
            "model_name": self.model_path.stem,
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
            "ultralytics_version": ultralytics.__version__,
            "class_map": dict(self._class_ids),
        }

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[ModelDetection, ...]:
        """Run model inference only for labels configured as animal classes."""
        _validate_threshold(confidence_threshold)
        try:
            results = self._model.predict(
                source=image_bgr,
                classes=list(self._class_ids),
                conf=confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"Wildlife model inference failed: {exc}") from exc

        detections: list[ModelDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for values, confidence, raw_class_id in zip(
                coordinates, confidences, class_ids, strict=True
            ):
                class_id = int(raw_class_id)
                label = self._class_ids.get(class_id)
                if label is None:
                    continue
                detections.append(
                    ModelDetection(
                        class_name=label,
                        confidence=float(confidence),
                        bounding_box=BoundingBox(*(float(value) for value in values)),
                    )
                )
        return tuple(detections)


class WildlifePipeline:
    """Decode images, apply real detector outputs and the configured taxonomy."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.25,
        taxonomy: WildlifeTaxonomy | None = None,
        detector: WildlifeDetector | None = None,
    ) -> None:
        _validate_threshold(confidence_threshold)
        if detector is not None and model_path is not None:
            raise ValueError("Pass either model_path or a loaded detector, not both.")
        self.taxonomy = taxonomy or load_taxonomy()
        self.detector = detector or WildlifeYoloDetector(
            model_path,
            animal_classes=self.taxonomy.animal_classes,
        )
        self.confidence_threshold = confidence_threshold
        unsupported_labels = [
            label for label in self.detector.class_names if not self.taxonomy.supports(label)
        ]
        if unsupported_labels:
            raise ModelLoadError(
                "The detector exposes labels not configured as animal classes: "
                f"{', '.join(unsupported_labels)}."
            )

    def infer_image(
        self,
        image_source: Path | str | bytes | bytearray | memoryview,
    ) -> ImageAnalysisResult:
        """Decode one image and return actual detections, analytics, and annotation."""
        image_bgr = decode_image(image_source)
        started = time.perf_counter()
        detections = self.detect_objects(image_bgr)
        annotated = annotate_detections(image_bgr, detections)
        metadata = dict(self.detector.metadata)
        metadata.update(self._analysis_metadata((time.perf_counter() - started) * 1000.0))
        return ImageAnalysisResult(
            detections=detections,
            summary=summarize_detections(detections),
            annotated_image=annotated,
            inference_metadata=metadata,
            source_shape=image_bgr.shape,
        )

    def detect_objects(self, image_bgr: np.ndarray) -> tuple[WildlifeDetection, ...]:
        """Run detector inference and map supported labels to explicit diet classes."""
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.size == 0
        ):
            raise ValueError("Inference requires a non-empty BGR image.")
        raw_detections = self.detector.detect(image_bgr, self.confidence_threshold)
        retained: list[WildlifeDetection] = []
        for detection in raw_detections:
            if not self.taxonomy.supports(detection.class_name):
                continue
            if not math.isfinite(detection.confidence) or not 0.0 <= detection.confidence <= 1.0:
                raise InferenceError("The model returned a non-finite or invalid confidence score.")
            if detection.confidence < self.confidence_threshold:
                continue
            if not all(math.isfinite(value) for value in detection.bounding_box.as_xyxy()):
                raise InferenceError("The model returned a non-finite bounding box.")
            if (
                detection.bounding_box.x2 <= detection.bounding_box.x1
                or detection.bounding_box.y2 <= detection.bounding_box.y1
            ):
                raise InferenceError("The model returned an empty or invalid bounding box.")
            retained.append(
                WildlifeDetection(
                    species=detection.class_name,
                    confidence=detection.confidence,
                    bounding_box=detection.bounding_box,
                    classification=self.taxonomy.classification_for(detection.class_name),
                )
            )
        return tuple(retained)

    def _analysis_metadata(self, duration_ms: float) -> Mapping[str, object]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "taxonomy_version": self.taxonomy.version,
            "taxonomy_source": self.taxonomy.source,
            "animal_classes": self.taxonomy.animal_classes,
            "taxonomy_notes": self.taxonomy.notes,
            "taxonomy_references": self.taxonomy.references,
            "inference_duration_ms": duration_ms,
        }


def _normalize_class_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    normalized: dict[int, str] = {}
    seen_names: set[str] = set()
    for raw_class_id, raw_name in items:
        class_id = int(raw_class_id)
        if class_id < 0 or not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Class IDs must be non-negative and labels must be non-empty.")
        name = raw_name.strip()
        canonical = _normalize(name)
        if canonical in seen_names:
            raise ValueError(f"Duplicate model label after normalization: '{name}'.")
        seen_names.add(canonical)
        normalized[class_id] = name
    return normalized


def _normalize(label: str) -> str:
    return label.strip().casefold()


def _validate_threshold(confidence_threshold: float) -> None:
    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("Detection confidence threshold must be between 0 and 1.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
