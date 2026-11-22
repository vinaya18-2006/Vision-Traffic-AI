"""User-facing errors for sign-language recognition."""


class SignLanguageError(Exception):
    """Base class for expected sign-recognition errors."""


class ConfigurationError(SignLanguageError):
    """The model, score threshold, or authorization schedule is invalid."""


class AuthorizationError(SignLanguageError):
    """Recognition is outside the configured operating window."""


class MediaInputError(SignLanguageError):
    """The supplied image or video is invalid or unsupported."""


class LowQualityInputError(MediaInputError):
    """The media is too small or visually degenerate to analyze reliably."""


class ModelLoadError(SignLanguageError):
    """A local model or hand-landmark component is missing or incompatible."""


class InferenceError(SignLanguageError):
    """A loaded recognition component failed during inference."""


class VideoOutputError(SignLanguageError):
    """An annotated video could not be written or validated."""
