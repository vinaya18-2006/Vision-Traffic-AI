"""Person detection and analysis using locally available model weights."""

from .detector import PersonAnalysisPipeline, PersonYoloDetector, default_model_path
from .video import infer_video

__all__ = ["PersonAnalysisPipeline", "PersonYoloDetector", "default_model_path", "infer_video"]
