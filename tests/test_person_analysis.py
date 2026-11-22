from __future__ import annotations

from importlib import import_module
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from modules.person_analysis.contracts import BoundingBox, PersonDetection
from modules.person_analysis.detector import (
    PERSON_CLASS_ID,
    PersonAnalysisPipeline,
    PersonYoloDetector,
    default_model_path,
)
from modules.person_analysis.errors import (
    InferenceError,
    MediaInputError,
    ModelLoadError,
    VideoOutputError,
)
from modules.person_analysis.utils import (
    PERSON_BOX_COLOUR,
    annotate_people,
    decode_image,
)
from modules.person_analysis.video import infer_video


def test_person_analysis_package_and_video_module_import() -> None:
    package = import_module("modules.person_analysis")
    video_module = import_module("modules.person_analysis.video")

    assert package.infer_video is video_module.infer_video


class _FakeDetector:
    metadata = {"model_name": "unit-test-detector"}

    def __init__(self, detections: tuple[PersonDetection, ...] = ()) -> None:
        self.detections = detections
        self.thresholds: list[float] = []

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float,
    ) -> tuple[PersonDetection, ...]:
        self.thresholds.append(confidence_threshold)
        return self.detections


def _image_bytes(format_name: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8)).save(stream, format=format_name)
    return stream.getvalue()


def _person(
    confidence: float = 0.9,
    box: BoundingBox | None = None,
) -> PersonDetection:
    return PersonDetection(confidence, box or BoundingBox(10, 10, 40, 50))


def _make_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (80, 64),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            writer.write(np.full((64, 80, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def test_detector_requests_person_class_only_and_applies_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    model_path = tmp_path / "weights.pt"
    model_path.write_bytes(b"unit-test weights")
    confidence = [0.9, 0.2]
    classes = [0.0, 2.0]
    coordinates = [[1.0, 2.0, 30.0, 40.0], [4.0, 5.0, 20.0, 25.0]]
    calls = []

    class _Tensor:
        def __init__(self, values) -> None:
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class _Boxes:
        xyxy = _Tensor(coordinates)
        conf = _Tensor(confidence)
        cls = _Tensor(classes)

    class _Model:
        names = {0: "person", 2: "car"}

        def predict(self, **kwargs):
            calls.append(kwargs)
            return [type("Result", (), {"boxes": _Boxes()})()]

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: _Model())
    detector = PersonYoloDetector(model_path)

    detections = detector.detect(np.zeros((64, 80, 3), dtype=np.uint8), 0.5)

    assert calls[0]["classes"] == [PERSON_CLASS_ID]
    assert calls[0]["conf"] == 0.5
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].bounding_box == BoundingBox(1.0, 2.0, 30.0, 40.0)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_pipeline_rejects_invalid_confidence_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        PersonAnalysisPipeline(confidence_threshold=threshold, detector=_FakeDetector())


def test_detector_rejects_missing_corrupt_and_incompatible_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ModelLoadError, match="No model will be downloaded"):
        PersonYoloDetector(missing)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a model")
    import ultralytics

    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda _: (_ for _ in ()).throw(ValueError("invalid checkpoint")),
    )
    with pytest.raises(ModelLoadError, match="Could not load person detector weights"):
        PersonYoloDetector(corrupt)

    class _WrongClassModel:
        names = {0: "car"}

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: _WrongClassModel())
    with pytest.raises(ModelLoadError, match="class 0=person"):
        PersonYoloDetector(corrupt)


def test_default_model_path_reuses_existing_car_detector_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "person.pt"
    monkeypatch.setenv("PERSON_ANALYSIS_MODEL_PATH", str(configured))
    assert default_model_path() == configured.resolve()

    monkeypatch.delenv("PERSON_ANALYSIS_MODEL_PATH")
    monkeypatch.setenv("CAR_DETECTION_MODEL_PATH", str(configured))
    assert default_model_path() == configured.resolve()


def test_image_inference_returns_detections_and_clipped_annotation() -> None:
    detector = _FakeDetector(
        (
            _person(0.91, BoundingBox(-4, 3, 50, 80)),
            _person(0.73, BoundingBox(52, 5, 100, 40)),
        )
    )
    result = PersonAnalysisPipeline(detector=detector).infer_image(_image_bytes())

    assert result.person_count == 2
    assert result.source_shape == (64, 80, 3)
    assert result.annotated_image.shape == result.source_shape
    assert result.inference_metadata["confidence_threshold"] == 0.25
    assert result.inference_metadata["score_is_calibrated_probability"] is False
    assert detector.thresholds == [0.25]
    assert tuple(result.annotated_image[3, 5]) == PERSON_BOX_COLOUR


def test_image_decoder_rejects_empty_malformed_and_unsupported_inputs(tmp_path: Path) -> None:
    with pytest.raises(MediaInputError, match="empty") as empty_error:
        decode_image(b"")
    assert "empty" in str(empty_error.value).lower()

    with pytest.raises(MediaInputError, match="malformed or cannot be decoded") as malformed_error:
        decode_image(b"not an image")
    assert "malformed" in str(malformed_error.value).lower()

    with pytest.raises(MediaInputError, match="Unsupported image format") as unsupported_error:
        decode_image(tmp_path / "image.bmp")
    assert "unsupported image format" in str(unsupported_error.value).lower()


def test_empty_detections_and_inference_failures_are_not_substituted() -> None:
    result = PersonAnalysisPipeline(detector=_FakeDetector()).infer_image(_image_bytes())
    assert result.detections == ()
    assert result.person_count == 0

    class _FailingDetector(_FakeDetector):
        def detect(self, image_bgr: np.ndarray, confidence_threshold: float):
            raise InferenceError("model execution failed")

    with pytest.raises(InferenceError, match="model execution failed"):
        PersonAnalysisPipeline(detector=_FailingDetector()).infer_image(_image_bytes())


def test_annotation_clips_boxes_and_does_not_mutate_source() -> None:
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    annotated = annotate_people(
        image,
        (
            _person(box=BoundingBox(-5, -5, 20, 20)),
            _person(box=BoundingBox(100, 1, 120, 20)),
            _person(box=BoundingBox(float("nan"), 1, 10, 10)),
        ),
    )
    assert tuple(annotated[0, 5]) == PERSON_BOX_COLOUR
    assert np.array_equal(image, np.zeros_like(image))


def test_malformed_and_unsupported_video_fail_before_inference(tmp_path: Path) -> None:
    detector = _FakeDetector()
    pipeline = PersonAnalysisPipeline(detector=detector)
    unsupported = tmp_path / "video.mkv"
    unsupported.write_bytes(b"")
    with pytest.raises(MediaInputError, match="Unsupported video container"):
        infer_video(pipeline, unsupported, tmp_path / "annotated.mkv")

    malformed = tmp_path / "malformed.avi"
    malformed.write_bytes(b"not a video")
    with pytest.raises(MediaInputError, match="could not open video"):
        infer_video(pipeline, malformed, tmp_path / "annotated.avi")
    assert detector.thresholds == []


def test_video_reports_counts_and_detections_per_frame_with_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.avi"
    _make_video(source)
    output = tmp_path / "annotated.avi"
    detections = (_person(), _person(0.8, BoundingBox(45, 10, 70, 50)))
    pipeline = PersonAnalysisPipeline(detector=_FakeDetector(detections))
    updates: list[tuple[int, int]] = []

    result = infer_video(
        pipeline,
        source,
        output,
        max_frames=2,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert result.frame_count == 2
    assert [frame.person_count for frame in result.frames] == [2, 2]
    assert result.frames[0].detections == detections
    assert result.frame_limit_reached
    assert updates == [(1, 2), (2, 2)]
    assert result.inference_metadata["count_policy"].startswith("person detections reported per frame")
    assert result.output_video_path.is_file()
    assert result.original_preview_frame.shape == (64, 80, 3)
    assert result.annotated_preview_frame.shape == (64, 80, 3)


def test_video_writer_codec_failure_releases_capture_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.avi"
    _make_video(source, frame_count=1)
    output = tmp_path / "annotated.avi"
    real_capture = cv2.VideoCapture
    captures = []

    def capture_factory(path: str):
        capture = real_capture(path)
        captures.append(capture)
        return capture

    class _UnavailableWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", capture_factory)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: _UnavailableWriter())
    with pytest.raises(VideoOutputError, match="selected output codec"):
        infer_video(PersonAnalysisPipeline(detector=_FakeDetector()), source, output)
    assert not output.exists()
    assert captures[0].isOpened() is False


def test_app_navigation_exposes_person_analysis_and_unavailable_features() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    assert app.exception == []
    app.selectbox[0].select("Person Attribute & Emotion Analysis").run(timeout=20)
    assert app.exception == []
    assert any(
        "Age estimation and facial-expression analysis are unavailable" in element.value
        for element in app.info
    )
