"""Typed inference contracts for voice-emotion classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

EmotionLabel = Literal["angry", "happy", "neutral", "sad"]


@dataclass(frozen=True)
class EmotionPrediction:
    """One model class and its uncalibrated output score."""

    label: EmotionLabel
    confidence: float
    raw_score: float


@dataclass(frozen=True)
class AudioInferenceResult:
    """Structured classification from an actually loaded emotion model."""

    label: EmotionLabel
    confidence: float
    source_name: str | None
    sample_rate: int | None
    duration_seconds: float | None
    inference_metadata: Mapping[str, object] = field(default_factory=dict)
    warning: str | None = None

    @property
    def prediction(self) -> EmotionPrediction:
        """Expose the classification in the model-agnostic contract shape."""
        return EmotionPrediction(self.label, self.confidence, self.confidence)
