"""Crop-based apparent car-colour estimation using HSV colour features."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import ColourEstimate, ColourLabel


@dataclass(frozen=True)
class ColourClassifier:
    """Estimate apparent body colour from a BGR car crop.

    This lightweight HSV-feature implementation is isolated from detection so
    it can later be replaced by a trained crop classifier.
    """

    minimum_confidence: float = 0.15
    minimum_pixels: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("Colour confidence threshold must be between 0 and 1.")
        if self.minimum_pixels < 1:
            raise ValueError("Minimum colour sample size must be at least one pixel.")

    def classify(self, car_crop_bgr: np.ndarray) -> ColourEstimate:
        """Return the dominant apparent colour and its pixel-share score."""
        if (
            not isinstance(car_crop_bgr, np.ndarray)
            or car_crop_bgr.ndim != 3
            or car_crop_bgr.shape[2] != 3
            or car_crop_bgr.size == 0
        ):
            raise ValueError("Car colour classification requires a non-empty BGR image.")

        height, width = car_crop_bgr.shape[:2]
        left, right = int(width * 0.15), max(int(width * 0.85), 1)
        top, bottom = int(height * 0.30), max(int(height * 0.90), 1)
        body_region = car_crop_bgr[top:bottom, left:right]
        if body_region.size == 0:
            return ColourEstimate("unknown", 0.0, "car crop is too small")

        hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.int16)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        chromatic = (saturation >= 55) & (value >= 35)
        neutral = ~chromatic
        labels: dict[ColourLabel, np.ndarray] = {
            "red": chromatic & ((hue <= 10) | (hue >= 170)),
            "orange": chromatic & (hue > 10) & (hue <= 23) & (value > 125),
            "brown": chromatic & (hue > 10) & (hue <= 23) & (value <= 125),
            "yellow": chromatic & (hue > 23) & (hue <= 35),
            "green": chromatic & (hue > 35) & (hue <= 85),
            "blue": chromatic & (hue > 85) & (hue <= 135),
            "other": chromatic & ((hue > 135) & (hue < 170)),
            "black": neutral & (value <= 70),
            "white": neutral & (value >= 205),
            "gray": neutral & (value > 70) & (value < 205),
            "silver": np.zeros_like(neutral),
        }

        neutral_values = value[neutral]
        if neutral_values.size:
            neutral_median = float(np.median(neutral_values))
            neutral_spread = float(np.std(neutral_values))
            if neutral_median >= 135 and neutral_spread >= 28:
                labels["silver"] = neutral & (value > 70) & (value < 205)
                labels["gray"] = np.zeros_like(neutral)

        pixel_count = int(value.size)
        if pixel_count < self.minimum_pixels:
            return ColourEstimate("unknown", 0.0, "car crop is too small")

        scores = {label: float(mask.sum()) / pixel_count for label, mask in labels.items()}
        best_label = max(scores, key=scores.__getitem__)
        best_score = scores[best_label]
        if best_score < self.minimum_confidence:
            return ColourEstimate("unknown", best_score, "colour evidence is below threshold")
        return ColourEstimate(best_label, best_score)