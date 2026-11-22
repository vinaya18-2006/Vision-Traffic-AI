"""Expected errors raised by the drowsiness module."""

from modules.person_analysis.errors import PersonAnalysisError


class DrowsinessError(PersonAnalysisError):
    """Base class for expected drowsiness-module errors."""


class DrowsinessMediaError(DrowsinessError):
    """The supplied image or video is invalid or unsupported."""


class DrowsinessVideoError(DrowsinessError):
    """An annotated video could not be created or verified."""


class DrowsinessInferenceError(DrowsinessError):
    """Person inference failed while processing valid media."""
