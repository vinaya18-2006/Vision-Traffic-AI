from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modules.car_detection.colour_classifier import ColourClassifier
from modules.car_detection.contracts import (
    BoundingBox,
    CarDetection,
    ColourEstimate,
    ColourLabel,
    ObjectDetection,
    PersonDetection,
)
from modules.car_detection.detector import CarDetectionPipeline, YoloDetector
from modules.car_detection.errors import MediaInputError, ModelLoadError, VideoOutputError
from modules.car_detection.utils import (
    BLUE_CAR_BOX_COLOUR,
    OTHER_KNOWN_CAR_BOX_COLOUR,
    PERSON_BOX_COLOUR,
    UNKNOWN_CAR_BOX_COLOUR,
    annotate_detections,
    car_box_colour,
    clip_box,
    count_detections,
    crop_box,
    decode_image,
)
from modules.car_detection.video import infer_video


def _car(colour: ColourLabel, box: BoundingBox | None = None) -> CarDetection:
    return CarDetection(
        confidence=0.9,
        bounding_box=box or BoundingBox(10, 10, 30, 30),
        estimated_colour=ColourEstimate(colour, 0.8),
    )


def test_clip_box_clips_coordinates_and_rejects_empty_boxes() -> None:
    assert clip_box(BoundingBox(-4.2, 1.1, 12.4, 20.2), 10, 15) == BoundingBox(0, 1, 10, 15)
    assert clip_box(BoundingBox(12, 2, 20, 8), 10, 10) is None
    assert clip_box(BoundingBox(float("nan"), 0, 1, 1), 10, 10) is None


def test_crop_box_is_clipped_and_small_crops_are_rejected() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    crop = crop_box(image, BoundingBox(-10, -10, 5, 6))
    assert crop is not None
    assert crop.shape == (6, 5, 3)
    assert crop_box(image, BoundingBox(1, 1, 2, 5), minimum_size=2) is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("blue", BLUE_CAR_BOX_COLOUR),
        ("red", OTHER_KNOWN_CAR_BOX_COLOUR),
        ("white", OTHER_KNOWN_CAR_BOX_COLOUR),
        ("other", OTHER_KNOWN_CAR_BOX_COLOUR),
        ("unknown", UNKNOWN_CAR_BOX_COLOUR),
    ],
)
def test_car_box_colour_mapping(label: ColourLabel, expected: tuple[int, int, int]) -> None:
    assert car_box_colour(label) == expected


def test_empty_detections_and_count_calculation() -> None:
    counts = count_detections((), ())
    assert counts.car_count == 0
    assert counts.person_count == 0

    cars = (_car("blue"), _car("white"))
    people = (PersonDetection(0.75, BoundingBox(2, 2, 7, 8)),)
    assert count_detections(cars, people).car_count == 2
    assert count_detections(cars, people).person_count == 1


def test_annotation_uses_distinct_car_and_person_rules() -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    cars = (
        _car("blue", BoundingBox(2, 2, 18, 18)),
        _car("white", BoundingBox(20, 2, 36, 18)),
        _car("unknown", BoundingBox(2, 24, 18, 44)),
    )
    people = (PersonDetection(0.8, BoundingBox(20, 24, 44, 44)),)
    annotated = annotate_detections(image, cars, people)
    assert tuple(annotated[2, 10]) == BLUE_CAR_BOX_COLOUR
    assert tuple(annotated[2, 28]) == OTHER_KNOWN_CAR_BOX_COLOUR
    assert tuple(annotated[24, 10]) == UNKNOWN_CAR_BOX_COLOUR
    assert tuple(annotated[24, 30]) == PERSON_BOX_COLOUR
    assert np.array_equal(image, np.zeros_like(image))


def test_colour_classifier_returns_palette_label_or_unknown() -> None:
    blue_crop = np.zeros((60, 100, 3), dtype=np.uint8)
    blue_crop[:, :] = (200, 30, 20)
    estimate = ColourClassifier(minimum_confidence=0.1).classify(blue_crop)
    assert estimate.label == "blue"
    assert estimate.confidence > 0.1

    tiny_crop = np.zeros((2, 2, 3), dtype=np.uint8)
    assert ColourClassifier().classify(tiny_crop).label == "unknown"


def test_invalid_image_is_rejected() -> None:
    with pytest.raises(MediaInputError, match="malformed or cannot be decoded"):
        decode_image(b"not an image")

    with pytest.raises(MediaInputError, match="Unsupported image format"):
        decode_image(Path("unsupported.bmp"))


def test_missing_model_is_reported_without_downloading(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.pt"
    with pytest.raises(ModelLoadError, match="No model will be downloaded"):
        CarDetectionPipeline(model_path=missing_model)


def test_corrupt_model_is_reported_clearly(tmp_path: Path) -> None:
    corrupt_model = tmp_path / "corrupt.pt"
    corrupt_model.write_bytes(b"not a pytorch model")
    with pytest.raises(ModelLoadError, match="Could not load YOLO11n weights"):
        YoloDetector(model_path=corrupt_model)


class _NoInferencePipeline:
    class _Detector:
        metadata = {"model_name": "test-only"}

    class _ColourClassifier:
        minimum_confidence = 0.15

    detector = _Detector()
    confidence_threshold = 0.25
    colour_classifier = _ColourClassifier()

    def detect_objects(
        self, image_bgr: np.ndarray
    ) -> tuple[tuple[CarDetection, ...], tuple[PersonDetection, ...]]:
        raise AssertionError("Malformed videos must fail before inference.")


def test_unsupported_and_malformed_video_are_rejected(tmp_path: Path) -> None:
    unsupported = tmp_path / "video.mkv"
    unsupported.write_bytes(b"")
    with pytest.raises(MediaInputError, match="Unsupported video container"):
        infer_video(_NoInferencePipeline(), unsupported, tmp_path / "annotated.mp4")

    malformed = tmp_path / "malformed.mp4"
    malformed.write_bytes(b"not a video")
    with pytest.raises(MediaInputError, match="could not open video"):
        infer_video(_NoInferencePipeline(), malformed, tmp_path / "annotated.mp4")


def test_video_output_will_not_overwrite_existing_file(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a video")
    output = tmp_path / "annotated.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(VideoOutputError, match="will not be overwritten"):
        infer_video(_NoInferencePipeline(), video, output)


def test_detection_contracts_retain_class_names_and_confidences() -> None:
    detection = ObjectDetection("car", 0.86, BoundingBox(1, 2, 20, 30))
    assert detection.class_name == "car"
    assert detection.confidence == pytest.approx(0.86)
