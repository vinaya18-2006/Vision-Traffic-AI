from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pytest
from PIL import Image

from modules.wildlife.contracts import BoundingBox, ModelDetection, WildlifeDetection
from modules.wildlife.detector import WildlifePipeline, WildlifeYoloDetector
from modules.wildlife.errors import (
    MediaInputError,
    ModelLoadError,
    TaxonomyConfigError,
    VideoOutputError,
)
from modules.wildlife.taxonomy import load_taxonomy
from modules.wildlife.utils import (
    CARNIVORE_BOX_COLOUR,
    NON_CARNIVORE_BOX_COLOUR,
    UNKNOWN_BOX_COLOUR,
    annotate_detections,
    box_colour,
    summarize_detections,
)
from modules.wildlife.video import infer_video


class _FakeDetector:
    metadata: Mapping[str, object] = {
        "model_name": "test-detector",
        "class_map": {0: "cat", 1: "cow", 2: "bear"},
    }
    class_names = ("cat", "cow", "bear")

    def __init__(self, detections: tuple[ModelDetection, ...]) -> None:
        self.detections = detections
        self.thresholds: list[float] = []

    def detect(
        self, image_bgr: np.ndarray, confidence_threshold: float
    ) -> tuple[ModelDetection, ...]:
        self.thresholds.append(confidence_threshold)
        return self.detections


def _model_detection(
    species: str,
    confidence: float,
    box: BoundingBox | None = None,
) -> ModelDetection:
    return ModelDetection(
        class_name=species,
        confidence=confidence,
        bounding_box=box or BoundingBox(4, 4, 30, 30),
    )


def _pipeline(
    detections: tuple[ModelDetection, ...],
    confidence_threshold: float = 0.25,
) -> WildlifePipeline:
    return WildlifePipeline(
        confidence_threshold=confidence_threshold,
        detector=_FakeDetector(detections),
    )


def _image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _make_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_taxonomy_maps_supported_classes_and_leaves_contextual_labels_unknown() -> None:
    taxonomy = load_taxonomy()

    assert taxonomy.version == "1.0.0"
    assert taxonomy.classification_for("Cat") == "carnivore"
    assert taxonomy.classification_for("HORSE") == "non_carnivore"
    assert taxonomy.classification_for("dog") == "unknown"
    assert taxonomy.classification_for("bear") == "unknown"
    assert taxonomy.classification_for("unconfigured class") == "unknown"
    assert len(taxonomy.classifications) == len(taxonomy.animal_classes)


def test_invalid_taxonomy_mapping_is_reported(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    config = json.loads(
        Path(__file__).resolve().parents[1]
        .joinpath("modules", "wildlife", "taxonomy.json")
        .read_text(encoding="utf-8")
    )
    config["classifications"]["cat"] = "sometimes_carnivore"
    taxonomy_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TaxonomyConfigError, match="must be one of"):
        load_taxonomy(taxonomy_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["classifications"].__setitem__("cat", []), "must be one of"),
        (
            lambda config: config["classifications"].__setitem__("Cat", "unknown"),
            "mapped more than once",
        ),
    ],
)
def test_malformed_taxonomy_entries_are_user_facing_errors(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    config = json.loads(
        Path(__file__).resolve().parents[1]
        .joinpath("modules", "wildlife", "taxonomy.json")
        .read_text(encoding="utf-8")
    )
    mutate(config)
    taxonomy_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TaxonomyConfigError, match=message):
        load_taxonomy(taxonomy_path)


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("carnivore", CARNIVORE_BOX_COLOUR),
        ("non_carnivore", NON_CARNIVORE_BOX_COLOUR),
        ("unknown", UNKNOWN_BOX_COLOUR),
    ],
)
def test_box_colour_mapping(
    classification: str,
    expected: tuple[int, int, int],
) -> None:
    assert box_colour(classification) == expected


def test_annotation_draws_red_blue_and_neutral_boxes_without_mutating_source() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = (
        WildlifeDetection("cat", 0.91, BoundingBox(3, 3, 25, 25), "carnivore"),
        WildlifeDetection("cow", 0.83, BoundingBox(30, 3, 52, 25), "non_carnivore"),
        WildlifeDetection("bear", 0.7, BoundingBox(3, 32, 25, 54), "unknown"),
    )

    annotated = annotate_detections(image, detections)

    assert tuple(annotated[3, 10]) == CARNIVORE_BOX_COLOUR
    assert tuple(annotated[3, 40]) == NON_CARNIVORE_BOX_COLOUR
    assert tuple(annotated[32, 10]) == UNKNOWN_BOX_COLOUR
    assert np.array_equal(image, np.zeros_like(image))


def test_detection_counts_species_breakdown_and_average_confidence() -> None:
    detections = (
        WildlifeDetection("cat", 0.9, BoundingBox(1, 1, 8, 8), "carnivore"),
        WildlifeDetection("cat", 0.7, BoundingBox(9, 1, 16, 8), "carnivore"),
        WildlifeDetection("cow", 0.8, BoundingBox(17, 1, 24, 8), "non_carnivore"),
        WildlifeDetection("bear", 0.6, BoundingBox(25, 1, 32, 8), "unknown"),
    )

    summary = summarize_detections(detections)

    assert summary.total_animals == 4
    assert summary.carnivores == 2
    assert summary.non_carnivores == 1
    assert summary.unknown == 1
    assert summary.species_counts == {"bear": 1, "cat": 2, "cow": 1}
    assert summary.carnivore_species_counts == {"cat": 2}
    assert summary.average_confidence == pytest.approx(0.75)


def test_confidence_filtering_and_empty_detections() -> None:
    detector = _FakeDetector(
        (
            _model_detection("cat", 0.81),
            _model_detection("cow", 0.49),
            _model_detection("bear", 0.12),
        )
    )
    pipeline = WildlifePipeline(confidence_threshold=0.5, detector=detector)

    result = pipeline.infer_image(_image_bytes())

    assert detector.thresholds == [0.5]
    assert [(item.species, item.confidence) for item in result.detections] == [("cat", 0.81)]
    assert result.summary.total_animals == 1
    assert result.summary.carnivores == 1

    empty = _pipeline(()).infer_image(_image_bytes())
    assert empty.detections == ()
    assert empty.summary.total_animals == 0
    assert empty.summary.carnivores == 0
    assert empty.summary.average_confidence is None
    assert empty.summary.species_counts == {}


def test_invalid_image_is_rejected() -> None:
    pipeline = _pipeline(())

    with pytest.raises(MediaInputError, match="malformed or cannot be decoded"):
        pipeline.infer_image(b"not an image")
    with pytest.raises(MediaInputError, match="empty"):
        pipeline.infer_image(b"")


def test_missing_and_corrupt_model_weights_are_reported(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ModelLoadError, match="No model will be downloaded"):
        WildlifeYoloDetector(missing)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a pytorch model")
    import ultralytics

    def fail_load(_: str) -> None:
        raise ValueError("invalid weights")

    monkeypatch.setattr(ultralytics, "YOLO", fail_load)
    with pytest.raises(ModelLoadError, match="Could not load wildlife model weights"):
        WildlifeYoloDetector(corrupt)


def test_model_with_no_configured_animal_labels_is_rejected(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "animals.pt"
    model_path.write_bytes(b"weights")

    class _IncompatibleModel:
        names = {0: "person"}

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: _IncompatibleModel())
    with pytest.raises(ModelLoadError, match="no labels matching"):
        WildlifeYoloDetector(model_path)


def test_unsupported_and_malformed_video_are_rejected(tmp_path: Path) -> None:
    pipeline = _pipeline(())
    unsupported = tmp_path / "wildlife.mkv"
    unsupported.write_bytes(b"")
    with pytest.raises(MediaInputError, match="Unsupported video container"):
        infer_video(pipeline, unsupported, tmp_path / "annotated.mp4")

    malformed = tmp_path / "malformed.avi"
    malformed.write_bytes(b"not a video")
    with pytest.raises(MediaInputError, match="could not open video"):
        infer_video(pipeline, malformed, tmp_path / "annotated.avi")


def test_empty_video_is_rejected_without_running_inference(tmp_path: Path) -> None:
    empty_video = tmp_path / "empty.avi"
    empty_video.write_bytes(b"")

    with pytest.raises(MediaInputError, match="could not open video"):
        infer_video(_pipeline(()), empty_video, tmp_path / "annotated.avi")


def test_video_processing_reports_frame_summaries_and_not_unique_animals(tmp_path: Path) -> None:
    source = tmp_path / "input.avi"
    _make_video(source)
    output = tmp_path / "annotated.avi"
    pipeline = _pipeline((_model_detection("cat", 0.9), _model_detection("cow", 0.8)))

    result = infer_video(pipeline, source, output, max_frames=2)

    assert result.frame_count == 2
    assert result.source_fps == pytest.approx(12.0)
    assert result.processing_fps > 0
    assert result.frame_limit_reached
    assert result.unique_animal_count is None
    assert result.total_per_frame_detections == 4
    assert result.summary.carnivores == 2
    assert [frame.animal_count for frame in result.frames] == [2, 2]
    assert result.output_video_path.is_file()
    assert result.original_preview_frame.shape == (48, 64, 3)
    assert result.annotated_preview_frame.shape == (48, 64, 3)


def test_video_codec_failure_is_reported_and_partial_output_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.avi"
    _make_video(source, frame_count=1)
    output = tmp_path / "annotated.avi"

    class _UnavailableWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: _UnavailableWriter())
    with pytest.raises(VideoOutputError, match="selected output codec"):
        infer_video(_pipeline(()), source, output)
    assert not output.exists()


def test_video_output_decoder_initialization_failure_is_reported_and_cleaned(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.avi"
    _make_video(source, frame_count=1)
    output = tmp_path / "annotated.avi"
    real_video_capture = cv2.VideoCapture
    capture_count = 0

    def fail_output_verification(path: str):
        nonlocal capture_count
        capture_count += 1
        if capture_count == 2:
            raise cv2.error("output codec unavailable")
        return real_video_capture(path)

    monkeypatch.setattr(cv2, "VideoCapture", fail_output_verification)
    with pytest.raises(VideoOutputError, match="output decoder"):
        infer_video(_pipeline(()), source, output)
    assert not output.exists()


def test_video_output_does_not_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "input.avi"
    _make_video(source, frame_count=1)

    with pytest.raises(VideoOutputError, match="must not overwrite"):
        infer_video(_pipeline(()), source, source)
