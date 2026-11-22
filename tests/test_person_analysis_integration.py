from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.person_analysis.detector import PersonAnalysisPipeline, default_model_path
from modules.person_analysis.video import infer_video


@pytest.mark.integration
def test_local_yolo11n_runs_real_person_image_and_video_inference(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = default_model_path()
    sample_image = root / "yolo_test.jpg"
    if not model_path.is_file() or not sample_image.is_file():
        pytest.skip("The documented local yolo11n.pt or yolo_test.jpg is not provisioned.")

    pipeline = PersonAnalysisPipeline(model_path=model_path, confidence_threshold=0.1)
    image_result = pipeline.infer_image(sample_image)

    assert image_result.inference_metadata["model_name"] == "YOLO11n"
    assert image_result.inference_metadata["model_sha256"]
    assert image_result.inference_metadata["class_map"] == {0: "person"}
    assert image_result.annotated_image.shape == image_result.source_shape
    assert all(0.1 <= item.confidence <= 1 for item in image_result.detections)
    assert image_result.person_count > 0

    frame = cv2.cvtColor(cv2.imread(str(sample_image)), cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_path = tmp_path / "sample.avi"
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
    assert len(video_result.frames) == 2
    assert all(frame_result.person_count == image_result.person_count for frame_result in video_result.frames)
    assert video_result.source_fps == pytest.approx(12.0)
    assert video_result.unique_person_count is None if hasattr(video_result, "unique_person_count") else True
    assert video_result.output_video_path.is_file()

