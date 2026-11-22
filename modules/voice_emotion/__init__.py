"""Voice-emotion detection module."""

from .contracts import AudioInferenceResult, EmotionLabel, EmotionPrediction
from .detector import (
    EmotionClassifier,
    TransformersEmotionClassifier,
    VoiceEmotionPipeline,
    default_model_path,
)

__all__ = [
    "AudioInferenceResult",
    "EmotionLabel",
    "EmotionPrediction",
    "EmotionClassifier",
    "TransformersEmotionClassifier",
    "VoiceEmotionPipeline",
    "default_model_path",
]
