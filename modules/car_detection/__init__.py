"""Task 1 car colour detection module."""

from .colour_classifier import ColourClassifier
from .contracts import (
    BoundingBox,
    CarDetection,
    ColourEstimate,
    DetectionCounts,
    ImageInferenceResult,
    ObjectDetection,
    PersonDetection,
    VideoFrameResult,
    VideoInferenceResult,
)
from .detector import CarDetectionPipeline, YoloDetector, default_model_path
from .video import infer_video

__all__ = [
    "BoundingBox",
    "CarDetection",
    "CarDetectionPipeline",
    "ColourClassifier",
    "ColourEstimate",
    "DetectionCounts",
    "ImageInferenceResult",
    "ObjectDetection",
    "PersonDetection",
    "VideoFrameResult",
    "VideoInferenceResult",
    "YoloDetector",
    "default_model_path",
    "infer_video",
]