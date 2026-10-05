from pathlib import Path

import cv2
import pytest

from modules.car_detection.detector import CarDetectionPipeline, default_model_path
from modules.car_detection.utils import decode_image
from modules.car_detection.video import infer_video


@pytest.mark.integration
def test_yolo11n_runs_real_image_and_video_inference(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    model_path = default_model_path()
    sample_image = repository_root / "yolo_test.jpg"
    pipeline = CarDetectionPipeline(model_path=model_path, confidence_threshold=0.1)

    image_result = pipeline.infer_image(sample_image)

    assert image_result.annotated_image.shape == image_result.source_shape
    assert image_result.inference_metadata["model_name"] == "YOLO11n"
    assert image_result.inference_metadata["model_sha256"]
    high_threshold_pipeline = CarDetectionPipeline(
        confidence_threshold=0.99,
        detector=pipeline.detector,
    )
    high_threshold_result = high_threshold_pipeline.infer_image(sample_image)
    assert high_threshold_result.person_count < image_result.person_count

    frame = decode_image(sample_image)
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
    assert video_result.source_fps == pytest.approx(12.0)
    assert video_result.processing_fps > 0
    assert video_result.frame_limit_reached
    assert video_result.unique_object_counts is None
    assert video_result.output_video_path.is_file()
    assert video_result.original_preview_frame.shape == frame.shape
    assert video_result.annotated_preview_frame.shape == frame.shape
