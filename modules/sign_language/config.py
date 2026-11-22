"""Validated local model and daily authorization-window configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import AuthorizationError, ConfigurationError


def default_model_path() -> Path:
    """Return the local checkpoint path without triggering a download."""
    configured = os.environ.get("SIGN_LANGUAGE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "models" / "sign_language" / "asl_landmarks.keras"


def _parse_time(value: str, name: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a valid 24-hour HH:MM time.") from exc


@dataclass(frozen=True)
class SignLanguageConfig:
    """Static ASL model configuration; authorization uses a daily local-time window."""

    model_path: Path | None = None
    minimum_score: float = 0.6
    allowed_start: time = time(9, 0)
    allowed_end: time = time(17, 0)
    timezone_name: str = "America/New_York"

    def __post_init__(self) -> None:
        if isinstance(self.minimum_score, bool) or not isinstance(
            self.minimum_score, (int, float)
        ):
            raise ConfigurationError("Minimum model score must be between 0 and 1.")
        if not math.isfinite(self.minimum_score) or not 0 <= self.minimum_score <= 1:
            raise ConfigurationError("Minimum model score must be between 0 and 1.")
        if not isinstance(self.allowed_start, time) or not isinstance(self.allowed_end, time):
            raise ConfigurationError("Authorization start and end must be local times.")
        if self.allowed_start.tzinfo is not None or self.allowed_end.tzinfo is not None:
            raise ConfigurationError("Authorization local times must not include UTC offsets.")
        if self.allowed_start >= self.allowed_end:
            raise ConfigurationError(
                "The daily authorization window must start before it ends."
            )
        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ConfigurationError(
                f"Unknown authorization timezone: {self.timezone_name!r}."
            ) from exc
        model_path = self.model_path if self.model_path is not None else default_model_path()
        object.__setattr__(self, "model_path", Path(model_path).expanduser().resolve())

    @classmethod
    def from_environment(cls) -> SignLanguageConfig:
        """Load validated runtime settings from environment variables."""
        try:
            minimum_score = float(os.environ.get("SIGN_LANGUAGE_MINIMUM_SCORE", "0.6"))
        except ValueError as exc:
            raise ConfigurationError(
                "SIGN_LANGUAGE_MINIMUM_SCORE must be a number between 0 and 1."
            ) from exc
        return cls(
            model_path=default_model_path(),
            minimum_score=minimum_score,
            allowed_start=_parse_time(
                os.environ.get("SIGN_LANGUAGE_ALLOWED_START", "09:00"),
                "SIGN_LANGUAGE_ALLOWED_START",
            ),
            allowed_end=_parse_time(
                os.environ.get("SIGN_LANGUAGE_ALLOWED_END", "17:00"),
                "SIGN_LANGUAGE_ALLOWED_END",
            ),
            timezone_name=os.environ.get(
                "SIGN_LANGUAGE_TIMEZONE", "America/New_York"
            ),
        )

    def is_authorized(self, now: datetime | None = None) -> bool:
        """Return whether a timezone-aware instant is within [start, end)."""
        if now is None:
            now = datetime.now(ZoneInfo(self.timezone_name))
        elif now.tzinfo is None or now.utcoffset() is None:
            raise ConfigurationError("Authorization checks require a timezone-aware datetime.")
        local_now = now.astimezone(ZoneInfo(self.timezone_name))
        return self.allowed_start <= local_now.timetz().replace(tzinfo=None) < self.allowed_end

    def require_authorized(self, now: datetime | None = None) -> None:
        """Raise before model loading or inference when the operating window is closed."""
        if not self.is_authorized(now):
            raise AuthorizationError(
                "ASL recognition is available only from "
                f"{self.allowed_start.strftime('%H:%M')} to "
                f"{self.allowed_end.strftime('%H:%M')} "
                f"({self.timezone_name}, start-inclusive/end-exclusive)."
            )
