"""Lazy MediaPipe hand landmarks and local Keras ASL-letter inference."""

from __future__ import annotations

import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from .config import SignLanguageConfig
from .contracts import BoundingBox, HandObservation, ImageRecognitionResult, SignPrediction
from .errors import InferenceError, ModelLoadError
from .preprocessing import (
    annotate_prediction,
    decode_image,
    landmark_vector,
    load_vocabulary,
    validate_image,
)

_MODEL_REVISION = "9e5cde38db66ae889322aeea3c547848676fa96e"
_MODEL_SHA256 = "9A73CD144A41C2CA92BD4D07ECEC2FE32F5F4DDE5E28C8BCC0B0ADA013502F11"


class LandmarkExtractor(Protocol):
    """Interface for extracting hands without binding inference to MediaPipe."""

    def detect_hands(self, image_bgr: np.ndarray) -> tuple[HandObservation, ...]: ...


class LetterClassifier(Protocol):
    """Interface for a loaded static-letter classifier."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def predict(self, features: np.ndarray) -> tuple[str, float]: ...


class MediaPipeHandExtractor:
    """Find up to two hands and return normalized image xyz landmarks."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise ModelLoadError(
                "MediaPipe Hands is unavailable or incompatible. Install the pinned "
                "Task 5 dependencies from requirements-sign-language.txt."
            ) from exc
        self._lock = threading.RLock()

    def detect_hands(self, image_bgr: np.ndarray) -> tuple[HandObservation, ...]:
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            try:
                results = self._hands.process(rgb_image)
            except Exception as exc:
                raise InferenceError(f"MediaPipe hand-landmark inference failed: {exc}") from exc
        image_height, image_width = image_bgr.shape[:2]
        observations: list[HandObservation] = []
        for hand_landmarks in results.multi_hand_landmarks or ():
            points = np.asarray(
                [(point.x, point.y, point.z) for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise InferenceError("MediaPipe returned malformed hand landmarks.")
            x1 = float(np.clip(points[:, 0].min() * image_width, 0, image_width - 1))
            y1 = float(np.clip(points[:, 1].min() * image_height, 0, image_height - 1))
            x2 = float(np.clip(points[:, 0].max() * image_width, 0, image_width - 1))
            y2 = float(np.clip(points[:, 1].max() * image_height, 0, image_height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            observations.append(
                HandObservation(points, BoundingBox(x1, y1, x2, y2))
            )
        return tuple(observations)

    def close(self) -> None:
        self._hands.close()


class KerasLandmarkClassifier:
    """Load the pinned local checkpoint with Keras' existing PyTorch backend."""

    def __init__(
        self,
        model_path: Path | str,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ModelLoadError(
                f"ASL landmark model not found at '{self.model_path}'. Run "
                "'.\\.venv\\Scripts\\python.exe -m modules.sign_language.provision_model' "
                "to acquire the pinned checkpoint explicitly."
            )
        self.labels = tuple(labels) if labels is not None else load_vocabulary()
        if len(self.labels) != 24 or len(set(self.labels)) != 24:
            raise ModelLoadError("The configured model vocabulary must contain 24 unique labels.")
        self._sha256 = _sha256_file(self.model_path)
        if self._sha256 != _MODEL_SHA256:
            raise ModelLoadError(
                f"ASL model checksum mismatch for '{self.model_path}'; expected "
                f"{_MODEL_SHA256}, got {self._sha256}."
            )
        try:
            os.environ.setdefault("KERAS_BACKEND", "torch")
            import keras

            backend = keras.backend.backend()
            if backend != "torch":
                raise ModelLoadError(
                    f"Task 5 requires the Keras PyTorch backend, but Keras selected "
                    f"{backend!r}. Set KERAS_BACKEND=torch before importing Keras."
                )
            self._model = keras.models.load_model(
                self.model_path,
                compile=False,
                safe_mode=True,
            )
        except ModelLoadError:
            raise
        except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
            raise ModelLoadError(
                f"Could not load the local ASL model at '{self.model_path}': {exc}"
            ) from exc
        input_shape = _single_shape(self._model.input_shape)
        output_shape = _single_shape(self._model.output_shape)
        if input_shape not in {(None, 63), (None, 21, 3)}:
            raise ModelLoadError(
                f"ASL model input shape {input_shape} is incompatible with 21 xyz landmarks."
            )
        if output_shape != (None, len(self.labels)):
            raise ModelLoadError(
                f"ASL model output shape {output_shape} does not match the "
                f"{len(self.labels)}-label vocabulary."
            )
        self._input_shape = input_shape
    @property
    def metadata(self) -> Mapping[str, object]:
        return {
            "model_id": "AmimulBmeIU/asl-alphabet-classifier",
            "model_revision": _MODEL_REVISION,
            "model_path": str(self.model_path),
            "model_sha256": self._sha256,
            "vocabulary": self.labels,
            "backend": "keras-pytorch",
            "score_is_calibrated_probability": False,
            "training_data_provenance": "not disclosed by publisher",
        }

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (63,) or not np.isfinite(vector).all():
            raise InferenceError("The classifier requires 63 finite MediaPipe xyz values.")
        model_input = vector.reshape((1, 63) if len(self._input_shape) == 2 else (1, 21, 3))
        try:
            scores = np.asarray(
                self._model(model_input, training=False),
                dtype=np.float32,
            ).reshape(-1)
        except Exception as exc:
            raise InferenceError(f"ASL landmark classification failed: {exc}") from exc
        if (
            scores.shape != (len(self.labels),)
            or not np.isfinite(scores).all()
            or np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            raise InferenceError("ASL model returned invalid class scores.")
        index = int(np.argmax(scores))
        confidence = float(scores[index])
        if not math.isfinite(confidence):
            raise InferenceError("ASL model returned a non-finite score.")
        return self.labels[index], confidence


class SignLanguagePipeline:
    """Authorize first, then lazily load and run static ASL fingerspelling inference."""

    def __init__(
        self,
        config: SignLanguageConfig | None = None,
        extractor: LandmarkExtractor | None = None,
        classifier: LetterClassifier | None = None,
        now_provider=None,
    ) -> None:
        self.config = config or SignLanguageConfig.from_environment()
        self._extractor = extractor
        self._classifier = classifier
        self._now_provider = now_provider

    @property
    def model_metadata(self) -> Mapping[str, object]:
        if self._classifier is None:
            return {
                "model_id": "AmimulBmeIU/asl-alphabet-classifier",
                "model_path": str(self.config.model_path),
                "available": self.config.model_path.is_file(),
                "training_data_provenance": "not disclosed by publisher",
            }
        return self._classifier.metadata

    def ensure_authorized(self) -> None:
        now = self._now_provider() if self._now_provider is not None else None
        self.config.require_authorized(now)

    def infer_image(
        self,
        image_source: Path | str | bytes | bytearray | memoryview,
    ) -> ImageRecognitionResult:
        self.ensure_authorized()
        image_bgr = decode_image(image_source)
        prediction = self.infer_frame(image_bgr)
        return ImageRecognitionResult(
            prediction=prediction,
            annotated_image=annotate_prediction(image_bgr, prediction),
            source_shape=image_bgr.shape,
            inference_metadata=self._inference_metadata(),
        )

    def prepare(self) -> None:
        """Check authorization before loading the classifier and hand-landmark model."""
        self.ensure_authorized()
        self._get_classifier()
        self._get_extractor()

    def infer_frame(
        self,
        image_bgr: np.ndarray,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> SignPrediction:
        self.ensure_authorized()
        validate_image(image_bgr)
        classifier = self._get_classifier()
        extractor = self._get_extractor()
        hands = extractor.detect_hands(image_bgr)
        if not hands:
            return SignPrediction(
                label="unknown",
                confidence=None,
                status="no_hand",
                bounding_box=None,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                warning="No hand was detected; no sign label was predicted.",
            )
        if len(hands) != 1:
            return SignPrediction(
                label="unknown",
                confidence=None,
                status="unknown",
                bounding_box=None,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                warning="This single-hand model cannot classify a frame containing multiple hands.",
            )
        hand = hands[0]
        features = landmark_vector(hand.landmarks)
        label, confidence = classifier.predict(features)
        if label not in classifier.metadata["vocabulary"]:
            raise InferenceError(f"Classifier returned unsupported label {label!r}.")
        status = "recognized" if confidence >= self.config.minimum_score else "unknown"
        return SignPrediction(
            label=label if status == "recognized" else "unknown",
            confidence=confidence,
            status=status,
            bounding_box=hand.bounding_box,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            warning=(
                None
                if status == "recognized"
                else "Model score is below the configured threshold; the label is withheld."
            ),
        )

    def _get_classifier(self) -> LetterClassifier:
        if self._classifier is None:
            self._classifier = KerasLandmarkClassifier(
                self.config.model_path,
                load_vocabulary(),
            )
        return self._classifier

    def _get_extractor(self) -> LandmarkExtractor:
        if self._extractor is None:
            self._extractor = MediaPipeHandExtractor()
        return self._extractor

    def close(self) -> None:
        if self._extractor is not None and hasattr(self._extractor, "close"):
            self._extractor.close()

    def _inference_metadata(self) -> Mapping[str, object]:
        return {
            **self._get_classifier().metadata,
            "language": "American Sign Language (ASL)",
            "recognition_scope": "static fingerspelled letters A-Y excluding J and Z",
            "minimum_score": self.config.minimum_score,
            "operation_window": (
                f"{self.config.allowed_start:%H:%M}-{self.config.allowed_end:%H:%M} "
                f"{self.config.timezone_name} (start-inclusive/end-exclusive)"
            ),
            "video_temporal_inference": False,
        }


def _single_shape(shape) -> tuple[int | None, ...]:
    if isinstance(shape, list):
        if len(shape) != 1:
            return ()
        shape = shape[0]
    try:
        return tuple(None if dimension is None else int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()
