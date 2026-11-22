from __future__ import annotations

import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import pytest

from modules.sign_language.config import SignLanguageConfig
from modules.sign_language.detector import SignLanguagePipeline
from modules.sign_language.preprocessing import load_vocabulary
from modules.sign_language.video import infer_video


def _integration_pipeline() -> SignLanguagePipeline:
    config = SignLanguageConfig(
        allowed_start=time(0, 0),
        allowed_end=time(23, 59),
        timezone_name="UTC",
    )
    return SignLanguagePipeline(
        config=config,
        now_provider=lambda: datetime.now(ZoneInfo("UTC")),
    )


@pytest.mark.integration
def test_real_checkpoint_and_mediapipe_run_on_existing_non_sign_scene(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    image_path = root / "yolo_test.jpg"
    model_path = SignLanguageConfig().model_path
    if not model_path.is_file():
        pytest.skip("The explicitly provisioned pinned ASL checkpoint is not available.")
    if not image_path.is_file():
        pytest.skip("The existing non-sign scene yolo_test.jpg is not available.")

    pipeline = _integration_pipeline()
    image_result = pipeline.infer_image(image_path)

    assert image_result.inference_metadata["model_sha256"]
    assert image_result.inference_metadata["vocabulary"] == load_vocabulary()
    assert image_result.prediction.status in {"recognized", "unknown", "no_hand"}
    assert image_result.annotated_image.shape == image_result.source_shape

    frame = cv2.imread(str(image_path))
    assert frame is not None
    video_path = tmp_path / "non_sign_scene.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (frame.shape[1], frame.shape[0]),
    )
    assert writer.isOpened()
    try:
        writer.write(frame)
        writer.write(frame)
    finally:
        writer.release()

    video_result = infer_video(
        pipeline,
        video_path,
        tmp_path / "annotated.avi",
        max_frames=2,
    )
    assert video_result.frame_count == 2
    assert [item.frame_index for item in video_result.frames] == [0, 1]
    assert all(
        item.prediction.status in {"recognized", "unknown", "no_hand"}
        for item in video_result.frames
    )
    assert video_result.output_video_path.is_file()


@pytest.mark.integration
def test_real_sign_acceptance_requires_a_genuine_labeled_sample() -> None:
    image_path = os.environ.get("SIGN_LANGUAGE_TEST_IMAGE")
    expected_label = os.environ.get("SIGN_LANGUAGE_TEST_LABEL")
    if not image_path and not expected_label:
        pytest.skip(
            "No genuine labeled ASL test image is provisioned; real sign accuracy is unverified."
        )
    if not image_path or not expected_label:
        pytest.fail(
            "Set both SIGN_LANGUAGE_TEST_IMAGE and SIGN_LANGUAGE_TEST_LABEL for labeled "
            "real-sign acceptance."
        )
    model_path = SignLanguageConfig().model_path
    if not model_path.is_file():
        pytest.skip("The explicitly provisioned pinned ASL checkpoint is not available.")
    assert expected_label in load_vocabulary()

    result = _integration_pipeline().infer_image(image_path)

    assert result.prediction.status == "recognized"
    assert result.prediction.label == expected_label
