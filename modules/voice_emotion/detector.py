"""Local-only inference adapter for the pretrained SUPERB emotion model."""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .contracts import AudioInferenceResult, EmotionLabel, EmotionPrediction
from .errors import InferenceError, MediaInputError, ModelLoadError
from .utils import decode_audio, resample_audio

MODEL_ID = "superb/wav2vec2-base-superb-er"
MODEL_SOURCE_URL = f"https://huggingface.co/{MODEL_ID}"
MODEL_SAMPLE_RATE = 16_000
MODEL_LABELS: Mapping[str, EmotionLabel] = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
}
REQUIRED_MODEL_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def default_model_path() -> Path:
    """Return the explicit local directory for provisioned model files."""
    configured = os.environ.get("VOICE_EMOTION_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "models" / "voice_emotion"


class EmotionClassifier(Protocol):
    """Inference-only model contract, independent of Streamlit and audio decoding."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def predict(self, waveform: np.ndarray, sample_rate: int) -> EmotionPrediction: ...


class TransformersEmotionClassifier:
    """Load and run a local Transformers checkpoint without contacting the Hub."""

    def __init__(self, model_directory: Path | str) -> None:
        self.model_directory = Path(model_directory).expanduser().resolve()
        missing = [
            filename
            for filename in REQUIRED_MODEL_FILES
            if not (self.model_directory / filename).is_file()
        ]
        if missing:
            raise ModelLoadError(
                f"The pretrained {MODEL_ID} model is not provisioned in "
                f"'{self.model_directory}'. Missing files: {', '.join(missing)}. "
                f"Model source: {MODEL_SOURCE_URL}. Files must be obtained explicitly; "
                "the application never downloads weights."
            )

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise ModelLoadError(
                "The local model files exist, but the 'transformers' package is not installed. "
                "Install dependencies only after reviewing and approving the model setup."
            ) from exc

        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                str(self.model_directory),
                local_files_only=True,
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                str(self.model_directory),
                local_files_only=True,
            )
            self._model.eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load the local {MODEL_ID} checkpoint at "
                f"'{self.model_directory}': {exc}"
            ) from exc

        model_labels = {
            str(index): str(label).lower()
            for index, label in self._model.config.id2label.items()
        }
        expected_ids = {str(index) for index in range(len(MODEL_LABELS))}
        if set(model_labels) != expected_ids or set(model_labels.values()) != set(MODEL_LABELS):
            raise ModelLoadError(
                "The local checkpoint label map is incompatible. Expected the SUPERB "
                "emotion classes neu, hap, ang, and sad."
            )
        self._labels = {
            int(index): MODEL_LABELS[label] for index, label in model_labels.items()
        }

    @property
    def metadata(self) -> Mapping[str, object]:
        """Describe the actual model and its local loading mode."""
        return {
            "model_id": MODEL_ID,
            "model_directory": str(self.model_directory),
            "model_labels": {index: label for index, label in self._labels.items()},
            "sample_rate": MODEL_SAMPLE_RATE,
            "weights_source": MODEL_SOURCE_URL,
            "loading_mode": "local_files_only",
        }

    def predict(self, waveform: np.ndarray, sample_rate: int) -> EmotionPrediction:
        """Run the loaded model on one validated mono waveform."""
        if sample_rate != MODEL_SAMPLE_RATE:
            raise MediaInputError(
                f"The {MODEL_ID} model requires {MODEL_SAMPLE_RATE} Hz audio; "
                f"received {sample_rate} Hz."
            )
        try:
            import torch

            model_inputs = self._feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._model(**model_inputs).logits
                scores = torch.softmax(logits, dim=-1)[0]
                index = int(torch.argmax(scores).item())
                score = float(scores[index].item())
        except Exception as exc:
            raise InferenceError(f"{MODEL_ID} inference failed: {exc}") from exc

        return EmotionPrediction(
            label=self._labels[index],
            confidence=score,
            raw_score=score,
        )


class VoiceEmotionPipeline:
    """Validate uploaded audio and run a real locally provisioned SER model."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.0,
        classifier: EmotionClassifier | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else default_model_path()
        )
        self.confidence_threshold = confidence_threshold
        self.classifier = (
            classifier
            if classifier is not None
            else TransformersEmotionClassifier(self.model_path)
        )

    def infer_audio(
        self,
        audio_source: Path | str | bytes | bytearray | memoryview,
        source_name: str | None = None,
    ) -> AudioInferenceResult:
        """Decode audio and return an actual model classification and score."""
        started = time.perf_counter()
        waveform, source_sample_rate = decode_audio(audio_source)
        model_waveform = resample_audio(
            waveform,
            source_sample_rate,
            MODEL_SAMPLE_RATE,
        )
        prediction = self.classifier.predict(model_waveform, MODEL_SAMPLE_RATE)
        if (
            prediction.label not in MODEL_LABELS.values()
            or not math.isfinite(prediction.confidence)
            or not 0.0 <= prediction.confidence <= 1.0
        ):
            raise InferenceError("The emotion model returned an invalid label or score.")
        if prediction.confidence < self.confidence_threshold:
            warning = (
                "The model score is below the configured threshold. "
                "The model's class is shown without changing it to another label."
            )
        else:
            warning = None

        metadata = dict(self.classifier.metadata)
        metadata.update(
            {
                "score_threshold": self.confidence_threshold,
                "score_is_calibrated_probability": False,
                "source_sample_rate": source_sample_rate,
                "sample_rate": MODEL_SAMPLE_RATE,
                "duration_seconds": model_waveform.size / MODEL_SAMPLE_RATE,
                "inference_duration_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return AudioInferenceResult(
            label=prediction.label,
            confidence=prediction.confidence,
            source_name=source_name,
            sample_rate=MODEL_SAMPLE_RATE,
            duration_seconds=model_waveform.size / MODEL_SAMPLE_RATE,
            inference_metadata=metadata,
            warning=warning,
        )


__all__ = [
    "EmotionClassifier",
    "MODEL_ID",
    "MODEL_LABELS",
    "MODEL_SAMPLE_RATE",
    "TransformersEmotionClassifier",
    "VoiceEmotionPipeline",
    "default_model_path",
]
