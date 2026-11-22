"""Wildlife-risk detection, dietary taxonomy, and video analytics."""

from .contracts import (
    BoundingBox,
    DetectionSummary,
    ImageAnalysisResult,
    ModelDetection,
    WildlifeDetection,
    VideoAnalysisResult,
    VideoFrameResult,
)
from .detector import WildlifePipeline, WildlifeYoloDetector, default_model_path
from .taxonomy import WildlifeTaxonomy, load_taxonomy
from .video import infer_video

__all__ = [
    "BoundingBox",
    "DetectionSummary",
    "ImageAnalysisResult",
    "ModelDetection",
    "VideoAnalysisResult",
    "VideoFrameResult",
    "WildlifeDetection",
    "WildlifePipeline",
    "WildlifeTaxonomy",
    "WildlifeYoloDetector",
    "default_model_path",
    "infer_video",
    "load_taxonomy",
]
