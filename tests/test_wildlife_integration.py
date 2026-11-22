from pathlib import Path

import pytest

from modules.wildlife.detector import WildlifePipeline, default_model_path


@pytest.mark.integration
def test_local_yolo11n_wildlife_smoke_test() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    model_path = default_model_path()
    if not model_path.is_file():
        pytest.skip("The documented local yolo11n.pt weights are not provisioned.")
    image_path = repository_root / "yolo_test.jpg"

    result = WildlifePipeline(model_path=model_path).infer_image(image_path)

    assert result.inference_metadata["model_name"] == "yolo11n"
    assert result.inference_metadata["model_sha256"]
    assert {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }.issubset(set(result.inference_metadata["animal_classes"]))
    assert result.summary.total_animals == len(result.detections)
