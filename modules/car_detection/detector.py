"""YOLO11n loading and image-level car/person inference."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .colour_classifier import ColourClassifier
from .contracts import (
    BoundingBox,
    CarDetection,
    ColourEstimate,
    ImageInferenceResult,
    ObjectDetection,
    PersonDetection,
)
from .errors import InferenceError, ModelLoadError
from .utils import annotate_detections, crop_box, decode_image

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
MODEL_FILENAME = "yolo11n.pt"


def default_model_path() -> Path:
    """Return the configured YOLO11n weights path, without downloading weights."""
    configured = os.environ.get("CAR_DETECTION_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / MODEL_FILENAME


class YoloDetector:
    """Load local YOLO11n weights and return only COCO person/car detections."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else default_model_path()
        )
        if not self.model_path.is_file():
            raise ModelLoadError(
                f"YOLO11n weights were not found at '{self.model_path}'. "
                "Place yolo11n.pt there or set CAR_DETECTION_MODEL_PATH. "
                "No model will be downloaded automatically."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load YOLO11n weights from '{self.model_path}': {exc}"
            ) from exc

        self._validate_class_map(self._model.names)
        try:
            self._model_sha256 = _sha256_file(self.model_path)
        except OSError as exc:
            raise ModelLoadError(
                f"Could not read YOLO11n weights at '{self.model_path}': {exc}"
            ) from exc

    @staticmethod
    def _validate_class_map(names: Mapping[int, str] | Sequence[str]) -> None:
        try:
            person_name = names[PERSON_CLASS_ID]
            car_name = names[CAR_CLASS_ID]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelLoadError(
                "The configured detector does not provide the expected COCO class IDs "
                "0=person and 2=car."
            ) from exc
        if person_name != "person" or car_name != "car":
            raise ModelLoadError(
                "The configured detector class map is incompatible: expected "
                "COCO class 0=person and class 2=car."
            )

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return model identity details used in inference results."""
        import ultralytics

        return {
            "model_name": "YOLO11n",
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
            "ultralytics_version": ultralytics.__version__,
            "class_map": {PERSON_CLASS_ID: "person", CAR_CLASS_ID: "car"},
        }

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[ObjectDetection, ...]:
        """Run real YOLO inference and return the requested COCO classes."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Detection confidence threshold must be between 0 and 1.")
        try:
            results = self._model.predict(
                source=image_bgr,
                classes=[PERSON_CLASS_ID, CAR_CLASS_ID],
                conf=confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"YOLO11n inference failed: {exc}") from exc

        detections: list[ObjectDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for coordinates_xyxy, confidence, class_id in zip(
                coordinates, confidences, class_ids
            ):
                class_index = int(class_id)
                if class_index == PERSON_CLASS_ID:
                    class_name = "person"
                elif class_index == CAR_CLASS_ID:
                    class_name = "car"
                else:
                    continue
                detections.append(
                    ObjectDetection(
                        class_name=class_name,
                        confidence=float(confidence),
                        bounding_box=BoundingBox(*(float(value) for value in coordinates_xyxy)),
                    )
                )
        return tuple(detections)


class CarDetectionPipeline:
    """Run YOLO car/person detection and separate crop-based colour estimation."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.25,
        colour_confidence_threshold: float = 0.15,
        detector: YoloDetector | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Detection confidence threshold must be between 0 and 1.")
        if detector is not None and model_path is not None:
            raise ValueError("Pass either a model path or a loaded detector, not both.")
        self.colour_classifier = ColourClassifier(colour_confidence_threshold)
        self.detector = detector or YoloDetector(model_path)
        self.confidence_threshold = confidence_threshold

    def infer_image(
        self,
        image_source: Path | str | bytes | bytearray | memoryview,
    ) -> ImageInferenceResult:
        """Decode an image, run inference, and return structured annotated output."""
        image_bgr = decode_image(image_source)
        started = time.perf_counter()
        cars, people = self.detect_objects(image_bgr)
        annotated = annotate_detections(image_bgr, cars, people)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metadata = dict(self.detector.metadata)
        metadata.update(
            {
                "confidence_threshold": self.confidence_threshold,
                "colour_classifier": "HSV crop colour features",
                "colour_confidence_threshold": self.colour_classifier.minimum_confidence,
                "inference_duration_ms": elapsed_ms,
            }
        )
        return ImageInferenceResult(
            cars=cars,
            people=people,
            annotated_image=annotated,
            inference_metadata=metadata,
            source_shape=image_bgr.shape,
        )

    def detect_objects(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[tuple[CarDetection, ...], tuple[PersonDetection, ...]]:
        """Detect and classify objects in a validated BGR image frame."""
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.size == 0
        ):
            raise ValueError("Inference requires a non-empty BGR image.")
        detections = self.detector.detect(image_bgr, self.confidence_threshold)
        cars: list[CarDetection] = []
        people: list[PersonDetection] = []
        for detection in detections:
            if detection.class_name == "car":
                car_crop = crop_box(image_bgr, detection.bounding_box, minimum_size=2)
                if car_crop is None:
                    colour = ColourEstimate("unknown", 0.0, "car crop is too small")
                else:
                    colour = self.colour_classifier.classify(car_crop)
                cars.append(
                    CarDetection(
                        confidence=detection.confidence,
                        bounding_box=detection.bounding_box,
                        estimated_colour=colour,
                    )
                )
            elif detection.class_name == "person":
                people.append(
                    PersonDetection(
                        confidence=detection.confidence,
                        bounding_box=detection.bounding_box,
                    )
                )
        return tuple(cars), tuple(people)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()