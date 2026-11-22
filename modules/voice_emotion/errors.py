"""Expected user-facing errors for speech-emotion analysis."""


class VoiceEmotionError(Exception):
    """Base class for expected voice-emotion module errors."""


class ModelLoadError(VoiceEmotionError):
    """The configured model weights are missing, corrupt, or incompatible."""


class MediaInputError(VoiceEmotionError):
    """The supplied audio cannot be decoded, is empty, or is unsupported."""


class InferenceError(VoiceEmotionError):
    """The voice model failed while processing valid audio."""

