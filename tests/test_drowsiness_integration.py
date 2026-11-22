from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.drowsiness.config import DrowsinessConfig
from modules.drowsiness.detector import DrowsinessPipeline
from modules.drowsiness.video import infer_video
from modules.person_analysis.detector import default_model_path


@pytest.mark.integration
def test_existing_yolo_weights_run_real_person_image_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = default_model_path()
    image_path = root / "yolo_test.jpg"
    if not model_path.is_file() or not image_path.is_file():
        pytest.skip("Existing yolo11n.pt or yolo_test.jpg is not available.")

    result = DrowsinessPipeline(
        DrowsinessConfig(confidence_threshold=0.1)
    ).infer_image(image_path)

    assert result.person_count == 4
    assert result.inference_metadata["model_name"] == "YOLO11n"
    assert result.inference_metadata["drowsiness_model_available"] is False
    assert result.sleeping_drowsy_count is None
    assert all(person.status == "unknown" for person in result.people)


@pytest.mark.integration
def test_real_yolo_runs_ordered_short_video_and_keeps_drowsiness_unavailable(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = default_model_path()
    image_path = root / "yolo_test.jpg"
    if not model_path.is_file() or not image_path.is_file():
        pytest.skip("Existing yolo11n.pt or yolo_test.jpg is not available.")
    frame = cv2.imread(str(image_path))
    assert frame is not None
    source_path = tmp_path / "two_frames.avi"
    writer = cv2.VideoWriter(
        str(source_path),
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

    pipeline = DrowsinessPipeline(DrowsinessConfig(confidence_threshold=0.1))
    result = infer_video(pipeline, source_path, tmp_path / "annotated.avi", max_frames=2)

    assert result.frame_count == 2
    assert [item.frame_index for item in result.frames] == [0, 1]
    assert [item.timestamp_seconds for item in result.frames] == pytest.approx([0.0, 1 / 12])
    assert [item.person_count for item in result.frames] == [4, 4]
    assert result.unique_tracklet_count == 4
    assert result.sleeping_drowsy_count is None
    assert result.drowsiness_model_available is False
    assert result.output_video_path.is_file()
    assert any("no validated temporal model" in warning for warning in result.warnings)
    assert any("shorter than the configured temporal window" in warning for warning in result.warnings)


@pytest.mark.integration
def test_missing_person_weights_are_reported_without_downloading(tmp_path: Path) -> None:
    from modules.person_analysis.errors import ModelLoadError

    with pytest.raises(ModelLoadError, match="No model will be downloaded"):
        DrowsinessPipeline(DrowsinessConfig(person_model_path=tmp_path / "missing.pt"))
