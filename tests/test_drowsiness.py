from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from modules.drowsiness.config import DrowsinessConfig
from modules.drowsiness.contracts import ImageAnalysisResult, PersonAssessment
from modules.drowsiness.detector import DrowsinessPipeline
from modules.drowsiness.errors import DrowsinessMediaError
from modules.drowsiness.preprocessing import (
    RED_BOX_COLOUR,
    UNKNOWN_BOX_COLOUR,
    annotate_people,
    annotation_colour,
    decode_image,
)
from modules.drowsiness.video import GreedyIoUTracker, infer_video
from modules.person_analysis.contracts import BoundingBox, PersonDetection


class _FakeDetector:
    metadata = {"model_name": "test-person-detector"}

    def __init__(self, detections: tuple[PersonDetection, ...] = ()) -> None:
        self.detections = detections

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[PersonDetection, ...]:
        return self.detections


def _person(box: BoundingBox) -> PersonDetection:
    return PersonDetection(0.9, box)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((80, 100, 3), 120, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_pipeline_counts_people_but_does_not_claim_drowsiness_or_age() -> None:
    detections = (
        _person(BoundingBox(10, 10, 35, 60)),
        _person(BoundingBox(50, 10, 80, 65)),
    )
    pipeline = DrowsinessPipeline(person_detector=_FakeDetector(detections))

    result = pipeline.infer_image(_png_bytes())

    assert result.person_count == 2
    assert result.sleeping_drowsy_count is None
    assert result.drowsiness_model_available is False
    assert all(person.status == "unknown" for person in result.people)
    assert result.inference_metadata["age_model_available"] is False


def test_very_dark_image_gets_a_quality_warning_without_a_state_guess() -> None:
    pipeline = DrowsinessPipeline(person_detector=_FakeDetector())
    dark_image = np.zeros((80, 100, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", dark_image)
    assert encoded

    result = pipeline.infer_image(buffer.tobytes())

    assert result.quality_warnings
    assert result.drowsiness_model_available is False
    assert result.sleeping_drowsy_count is None


@pytest.mark.parametrize("source", [b"", b"not an image"])
def test_image_decoder_rejects_empty_and_malformed_inputs(source: bytes) -> None:
    with pytest.raises(DrowsinessMediaError):
        decode_image(source)


def test_image_decoder_rejects_unsupported_path_extension(tmp_path: Path) -> None:
    media = tmp_path / "frame.gif"
    media.write_bytes(_png_bytes())

    with pytest.raises(DrowsinessMediaError, match="Unsupported image format"):
        decode_image(media)


def test_unknown_people_use_non_red_annotation() -> None:
    pipeline = DrowsinessPipeline(
        person_detector=_FakeDetector((_person(BoundingBox(5, 5, 40, 50)),))
    )
    result = pipeline.infer_image(_png_bytes())

    assert annotation_colour(result.people[0]) == UNKNOWN_BOX_COLOUR
    assert not np.array_equal(result.annotated_image, np.zeros_like(result.annotated_image))


def test_red_annotation_requires_real_model_metadata_and_temporal_window() -> None:
    detection = _person(BoundingBox(10, 10, 40, 60))
    with pytest.raises(ValueError, match="model ID"):
        PersonAssessment(detection, status="drowsy")
    with pytest.raises(ValueError, match="temporal evidence"):
        PersonAssessment(
            detection,
            status="drowsy",
            status_confidence=0.8,
            status_model_id="test-model",
        )

    evidence = PersonAssessment(
        detection,
        status="drowsy",
        status_confidence=0.8,
        evidence_window=(1.0, 11.0),
        status_model_id="test-model",
    )
    assert annotation_colour(evidence) == RED_BOX_COLOUR
    annotated = annotate_people(np.zeros((80, 100, 3), dtype=np.uint8), (evidence,))
    assert tuple(annotated[10, 10]) == RED_BOX_COLOUR


def test_state_count_aggregation_counts_only_model_supported_sleep_states() -> None:
    from modules.drowsiness.contracts import VideoFrameResult

    people = (
        PersonAssessment(
            _person(BoundingBox(1, 1, 20, 30)),
            status="alert",
            status_confidence=0.95,
            status_model_id="test-model",
        ),
        PersonAssessment(
            _person(BoundingBox(21, 1, 40, 30)),
            status="drowsy",
            status_confidence=0.8,
            evidence_window=(0.0, 10.0),
            status_model_id="test-model",
        ),
        PersonAssessment(
            _person(BoundingBox(41, 1, 60, 30)),
            status="sleeping",
            status_confidence=0.9,
            evidence_window=(0.0, 10.0),
            status_model_id="test-model",
        ),
    )
    image = ImageAnalysisResult(
        people=people,
        source_image=np.zeros((80, 100, 3), dtype=np.uint8),
        annotated_image=np.zeros((80, 100, 3), dtype=np.uint8),
        inference_metadata={},
        drowsiness_model_available=True,
    )
    frame = VideoFrameResult(0, 0.0, people, drowsiness_model_available=True)

    assert image.sleeping_drowsy_count == 2
    assert frame.sleeping_drowsy_count == 2
    assert VideoFrameResult(0, 0.0, people).sleeping_drowsy_count is None


def test_temporal_window_and_person_configuration_boundaries() -> None:
    assert DrowsinessConfig(temporal_window_seconds=1.0).temporal_window_seconds == 1.0
    assert DrowsinessConfig(temporal_window_seconds=60.0).temporal_window_seconds == 60.0
    for invalid in (0.0, 60.1, float("nan")):
        with pytest.raises(ValueError, match="Temporal window"):
            DrowsinessConfig(temporal_window_seconds=invalid)
    with pytest.raises(ValueError, match="confidence threshold"):
        DrowsinessConfig(confidence_threshold=1.1)


def test_corrupt_person_weights_raise_model_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    from modules.person_analysis.errors import ModelLoadError

    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a YOLO checkpoint")
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda _: (_ for _ in ()).throw(ValueError("invalid checkpoint")),
    )

    with pytest.raises(ModelLoadError, match="Could not load person detector weights"):
        DrowsinessPipeline(DrowsinessConfig(person_model_path=corrupt_path))


def test_tracker_associates_overlapping_persons_and_keeps_multiple_occupants() -> None:
    tracker = GreedyIoUTracker(minimum_iou=0.3, max_gap_frames=1)
    first = (
        _person(BoundingBox(0, 0, 20, 40)),
        _person(BoundingBox(60, 0, 80, 40)),
    )
    second = (
        _person(BoundingBox(2, 0, 22, 40)),
        _person(BoundingBox(58, 0, 78, 40)),
    )

    ids_first = tracker.update(first, 0)
    ids_second = tracker.update(second, 1)

    assert ids_first == (1, 2)
    assert ids_second == ids_first
    assert tracker.created_track_count == 2


def test_tracker_starts_a_new_tracklet_after_occlusion_gap() -> None:
    tracker = GreedyIoUTracker(minimum_iou=0.3, max_gap_frames=1)
    detection = (_person(BoundingBox(0, 0, 20, 40)),)

    assert tracker.update(detection, 0) == (1,)
    assert tracker.update((), 1) == ()
    assert tracker.update((), 2) == ()
    assert tracker.update(detection, 3) == (2,)


def test_short_low_fps_video_reports_temporal_limitations(tmp_path: Path) -> None:
    source = tmp_path / "short.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (80, 64),
    )
    assert writer.isOpened()
    try:
        writer.write(np.zeros((64, 80, 3), dtype=np.uint8))
        writer.write(np.zeros((64, 80, 3), dtype=np.uint8))
    finally:
        writer.release()

    pipeline = DrowsinessPipeline(person_detector=_FakeDetector())
    result = infer_video(pipeline, source, tmp_path / "annotated.avi")

    assert result.frame_count == 2
    assert any("too low" in warning for warning in result.warnings)
    assert any("shorter than the configured temporal window" in warning for warning in result.warnings)
    assert any("dark lighting" in warning for warning in result.warnings)


def test_corrupt_video_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.avi"
    source.write_bytes(b"not a video")

    with pytest.raises(DrowsinessMediaError, match="could not open|No decodable"):
        infer_video(
            DrowsinessPipeline(person_detector=_FakeDetector()),
            source,
            tmp_path / "annotated.avi",
        )


def test_pipeline_rejects_invalid_track_id_mapping() -> None:
    pipeline = DrowsinessPipeline(
        person_detector=_FakeDetector((_person(BoundingBox(1, 1, 20, 30)),))
    )
    frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "yolo_test.jpg"))
    assert frame is not None
    with pytest.raises(ValueError, match="One track ID"):
        pipeline.assess_frame(frame, track_ids=())
