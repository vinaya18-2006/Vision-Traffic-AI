from __future__ import annotations

import io
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest
from PIL import Image

from modules.sign_language.config import SignLanguageConfig
from modules.sign_language.contracts import BoundingBox, HandObservation
from modules.sign_language.detector import KerasLandmarkClassifier, SignLanguagePipeline
from modules.sign_language.errors import (
    AuthorizationError,
    LowQualityInputError,
    MediaInputError,
    ModelLoadError,
)
from modules.sign_language.preprocessing import (
    decode_image,
    landmark_vector,
    load_vocabulary,
)
from modules.sign_language.video import infer_video


_NOW = datetime(2026, 11, 7, 12, 0, tzinfo=ZoneInfo("America/New_York"))


def _image_bytes(width: int = 128, height: int = 128, solid: bool = False) -> bytes:
    if solid:
        rgb = np.full((height, width, 3), 30, dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        rgb = rng.integers(10, 245, size=(height, width, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def _observation(x: float = 0.2) -> HandObservation:
    landmarks = np.tile(np.asarray([x, 0.4, 0.0], dtype=np.float32), (21, 1))
    return HandObservation(landmarks, BoundingBox(10, 10, 80, 90))


class _FakeExtractor:
    def __init__(self, hands: tuple[HandObservation, ...] = ()) -> None:
        self.hands = hands
        self.calls = 0

    def detect_hands(self, image_bgr: np.ndarray) -> tuple[HandObservation, ...]:
        self.calls += 1
        return self.hands


class _SequenceExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def detect_hands(self, image_bgr: np.ndarray) -> tuple[HandObservation, ...]:
        self.calls += 1
        return (_observation(0.1 * self.calls),)


class _FakeClassifier:
    def __init__(self, label: str = "A", score: float = 0.9) -> None:
        self.label = label
        self.score = score
        self.metadata = {
            "model_id": "unit-test-only",
            "vocabulary": load_vocabulary(),
            "score_is_calibrated_probability": False,
        }
        self.features: list[np.ndarray] = []

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        self.features.append(features.copy())
        return self.label, self.score


def _pipeline(
    tmp_path: Path,
    hands: tuple[HandObservation, ...] = (),
    classifier: _FakeClassifier | None = None,
    extractor=None,
) -> SignLanguagePipeline:
    config = SignLanguageConfig(model_path=tmp_path / "not-loaded.keras")
    return SignLanguagePipeline(
        config=config,
        extractor=extractor or _FakeExtractor(hands),
        classifier=classifier or _FakeClassifier(),
        now_provider=lambda: _NOW,
    )


def test_authorization_window_uses_local_timezone_and_half_open_boundaries() -> None:
    config = SignLanguageConfig(timezone_name="America/New_York")

    assert config.is_authorized(datetime(2026, 11, 7, 9, 0, tzinfo=ZoneInfo("America/New_York")))
    assert config.is_authorized(datetime(2026, 11, 7, 16, 59, tzinfo=ZoneInfo("America/New_York")))
    assert not config.is_authorized(
        datetime(2026, 11, 7, 8, 59, tzinfo=ZoneInfo("America/New_York"))
    )
    assert not config.is_authorized(
        datetime(2026, 11, 7, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    )


def test_authorization_converts_aware_instants_and_rejects_naive_values() -> None:
    config = SignLanguageConfig(timezone_name="America/New_York")

    assert config.is_authorized(datetime(2026, 11, 7, 14, 0, tzinfo=ZoneInfo("UTC")))
    from modules.sign_language.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="timezone-aware"):
        config.is_authorized(datetime(2026, 11, 7, 12, 0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timezone_name", "Not/A_Timezone"),
        ("allowed_start", time(17, 0)),
        ("minimum_score", float("nan")),
        ("minimum_score", 1.1),
    ],
)
def test_config_rejects_invalid_values(field: str, value) -> None:
    from modules.sign_language.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        SignLanguageConfig(**{field: value})


def test_vocabulary_is_the_pinned_24_letter_static_asl_scope() -> None:
    labels = load_vocabulary()

    assert len(labels) == 24
    assert labels == tuple("ABCDEFGHI" + "KLMNOPQRSTUVWXY")
    assert "J" not in labels
    assert "Z" not in labels


def test_decoder_rejects_empty_malformed_unsupported_and_low_quality_images(
    tmp_path: Path,
) -> None:
    with pytest.raises(MediaInputError, match="empty"):
        decode_image(b"")
    with pytest.raises(MediaInputError, match="malformed or cannot be decoded"):
        decode_image(b"not an image")
    with pytest.raises(MediaInputError, match="Unsupported image format"):
        decode_image(tmp_path / "frame.bmp")
    with pytest.raises(LowQualityInputError, match="too small"):
        decode_image(_image_bytes(32, 32))
    with pytest.raises(LowQualityInputError, match="too little visual detail"):
        decode_image(_image_bytes(solid=True))


def test_landmark_preprocessing_validates_shape_and_finite_values() -> None:
    values = np.arange(63, dtype=np.float32).reshape(21, 3)

    assert np.array_equal(landmark_vector(values), values.reshape(63))
    with pytest.raises(MediaInputError, match="21 finite"):
        landmark_vector(np.zeros((20, 3), dtype=np.float32))
    values[0, 0] = np.nan
    with pytest.raises(MediaInputError, match="21 finite"):
        landmark_vector(values)


def test_image_pipeline_maps_supported_letter_and_returns_annotated_copy(
    tmp_path: Path,
) -> None:
    classifier = _FakeClassifier(label="M")
    image = decode_image(_image_bytes())
    pipeline = _pipeline(tmp_path, (_observation(),), classifier)

    result = pipeline.infer_image(_image_bytes())

    assert result.prediction.label == "M"
    assert result.prediction.status == "recognized"
    assert result.prediction.confidence == pytest.approx(0.9)
    assert result.annotated_image.shape == result.source_shape
    assert np.array_equal(image, decode_image(_image_bytes()))
    assert classifier.features[0].shape == (63,)
    assert result.inference_metadata["video_temporal_inference"] is False
    assert [features[0] for features in classifier.features] == pytest.approx([0.1, 0.2])


def test_no_hand_and_multiple_hands_are_explicit_unknown_states(tmp_path: Path) -> None:
    no_hand = _pipeline(tmp_path).infer_image(_image_bytes())
    multiple_hands = _pipeline(
        tmp_path,
        (_observation(), _observation(0.5)),
    ).infer_image(_image_bytes())

    assert no_hand.prediction.status == "no_hand"
    assert no_hand.prediction.label == "unknown"
    assert no_hand.prediction.confidence is None
    assert multiple_hands.prediction.status == "unknown"
    assert "multiple hands" in multiple_hands.prediction.warning.lower()


def test_low_score_is_withheld_as_unknown_not_replaced_with_another_letter(
    tmp_path: Path,
) -> None:
    classifier = _FakeClassifier(label="Q", score=0.2)
    result = _pipeline(tmp_path, (_observation(),), classifier).infer_image(_image_bytes())

    assert result.prediction.status == "unknown"
    assert result.prediction.label == "unknown"
    assert result.prediction.confidence == pytest.approx(0.2)
    assert "below" in result.prediction.warning


def test_authorization_precedes_model_loading_and_media_decoding(tmp_path: Path) -> None:
    outside_window = datetime(2026, 11, 7, 8, 59, tzinfo=ZoneInfo("America/New_York"))
    pipeline = SignLanguagePipeline(
        config=SignLanguageConfig(model_path=tmp_path / "missing.keras"),
        now_provider=lambda: outside_window,
    )

    with pytest.raises(AuthorizationError, match="available only"):
        pipeline.infer_image(b"not even decoded")
    assert not (tmp_path / "missing.keras").exists()


def test_missing_and_corrupt_models_fail_explicitly_before_inference(
    tmp_path: Path,
) -> None:
    observation = (_observation(),)
    missing_pipeline = SignLanguagePipeline(
        config=SignLanguageConfig(model_path=tmp_path / "missing.keras"),
        extractor=_FakeExtractor(observation),
        now_provider=lambda: _NOW,
    )
    with pytest.raises(ModelLoadError, match="not found"):
        missing_pipeline.infer_image(_image_bytes())

    corrupt_path = tmp_path / "corrupt.keras"
    corrupt_path.write_bytes(b"not a Keras checkpoint")
    corrupt_pipeline = SignLanguagePipeline(
        config=SignLanguageConfig(model_path=corrupt_path),
        extractor=_FakeExtractor(observation),
        now_provider=lambda: _NOW,
    )
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        corrupt_pipeline.infer_image(_image_bytes())


def test_video_rejects_unsupported_and_malformed_inputs_before_inference(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path)
    unsupported = tmp_path / "clip.mkv"
    unsupported.write_bytes(b"")
    with pytest.raises(MediaInputError, match="Unsupported video container"):
        infer_video(pipeline, unsupported, tmp_path / "out.mkv")

    malformed = tmp_path / "broken.avi"
    malformed.write_bytes(b"not a video")
    with pytest.raises(MediaInputError, match="could not open video"):
        infer_video(pipeline, malformed, tmp_path / "out.avi")


def test_video_preserves_frame_order_and_reports_independent_static_letters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (96, 96),
    )
    assert writer.isOpened()
    try:
        for value in (50, 100, 150):
            writer.write(np.full((96, 96, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    classifier = _FakeClassifier()
    extractor = _SequenceExtractor()
    pipeline = _pipeline(tmp_path, classifier=classifier, extractor=extractor)
    progress: list[tuple[int, int]] = []
    result = infer_video(
        pipeline,
        source,
        tmp_path / "annotated.avi",
        max_frames=2,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert [frame.frame_index for frame in result.frames] == [0, 1]
    assert [frame.timestamp_seconds for frame in result.frames] == [0.0, 0.1]
    assert [frame.prediction.label for frame in result.frames] == ["A", "A"]
    assert result.frame_count == 2
    assert result.frame_limit_reached
    assert result.inference_metadata["video_temporal_inference"] is False
    assert progress == [(1, 2), (2, 2)]
    assert result.output_video_path.is_file()
    assert result.original_preview_frame.shape == (96, 96, 3)
    assert result.annotated_preview_frame.shape == (96, 96, 3)
