"""Expected user-facing errors for wildlife risk analysis."""


class WildlifeError(Exception):
    """Base class for expected wildlife module errors."""


class ModelLoadError(WildlifeError):
    """The configured model weights are missing, corrupt, or incompatible."""


class MediaInputError(WildlifeError):
    """The supplied image or video cannot be decoded or is unsupported."""


class VideoOutputError(WildlifeError):
    """The annotated video could not be encoded or written."""


class InferenceError(WildlifeError):
    """The detector failed while processing valid media."""


class TaxonomyConfigError(WildlifeError):
    """The configured animal taxonomy is missing or invalid."""
