"""User-facing errors raised by the Task 1 module."""


class CarDetectionError(Exception):
    """Base class for expected Task 1 errors that can be shown in the UI."""


class ModelLoadError(CarDetectionError):
    """The configured detector weights are missing, invalid, or incompatible."""


class MediaInputError(CarDetectionError):
    """The supplied image or video cannot be decoded or is unsupported."""


class VideoOutputError(CarDetectionError):
    """The annotated video could not be created or written."""


class InferenceError(CarDetectionError):
    """The detector failed while processing an otherwise valid frame."""
