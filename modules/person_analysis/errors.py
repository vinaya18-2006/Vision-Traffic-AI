"""Expected user-facing errors for person analysis."""


class PersonAnalysisError(Exception):
    """Base class for expected person-analysis errors."""


class ModelLoadError(PersonAnalysisError):
    """Local detector weights are missing, invalid, or incompatible."""


class MediaInputError(PersonAnalysisError):
    """The supplied image or video is invalid or unsupported."""


class VideoOutputError(PersonAnalysisError):
    """Annotated video could not be created or validated."""


class InferenceError(PersonAnalysisError):
    """The detector failed while processing valid media."""

