# Task 1 — Car colour detection

The module detects people and cars with Ultralytics YOLO11n pretrained detection weights and
estimates each car's **apparent** colour from a separate crop-based HSV colour-feature stage.
The Streamlit view is reusable from `app.py`; importing and calling the inference module does not
require Streamlit.

## Model and reproducibility

- Detector: Ultralytics YOLO11n, COCO-pretrained `yolo11n.pt`; COCO class `0` is `person`, class
  `2` is `car`. Inference filters to those two classes and uses Ultralytics' standard prediction
  preprocessing and non-maximum suppression.
- Weights format: Ultralytics/PyTorch `.pt`. The default path is the repository-root
  `yolo11n.pt`; configure another local file with `CAR_DETECTION_MODEL_PATH` or the Task 1 UI.
  Missing or incompatible weights raise a user-facing error; this code never downloads or
  substitutes a model. To provision another machine, manually obtain the `yolo11n.pt` asset from
  the official [Ultralytics YOLO11 model page](https://docs.ultralytics.com/models/yolo11/), place
  it at the repository root or set `CAR_DETECTION_MODEL_PATH`, then verify its hash. Do not call
  `YOLO("yolo11n.pt")` before placing the file; Ultralytics may download missing named weights.
  Review current [license and model terms](https://github.com/ultralytics/ultralytics) before use
  or redistribution (Ultralytics software is AGPL-3.0; commercial licensing may be required for
  some uses).
- Local validated artifact SHA-256:
  `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`.
  Verify a replacement file with PowerShell:
  `Get-FileHash .\yolo11n.pt -Algorithm SHA256`.
- The separate colour estimator is a non-trained HSV crop-feature heuristic; it has no model
  weights or independent model license. It examines the interior portion of each clipped car crop,
  groups HSV evidence into the supported palette, and returns `unknown` if the strongest colour
  evidence is below the configured threshold. Its evidence score is a pixel-share heuristic, not a
  calibrated probability.

## Use

```python
from modules.car_detection import CarDetectionPipeline, infer_video

pipeline = CarDetectionPipeline(
    confidence_threshold=0.25,
    colour_confidence_threshold=0.15,
)
image_result = pipeline.infer_image("path/to/image.jpg")

# For video, supply a distinct output path. The caller owns and cleans up this file.
video_result = infer_video(pipeline, "path/to/input.mp4", "path/to/annotated.mp4")
```

The structured image result contains car/person counts, per-car class, confidence, XYXY box,
estimated colour and colour-evidence score, an annotated BGR image, and model/inference metadata.
`infer_video` returns per-frame detection details, total processed frame count, source/output and
measured processing FPS. Video frame counts are detections per frame, **not** unique-object counts;
tracking is not implemented, and the result intentionally has no unique count. Video processing
defaults to a 3,000-frame maximum; set `max_frames` to a positive integer to adjust it. The result
states whether additional frames remained when the cap was reached.

Run the integrated UI from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run Task 1 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_car_detection.py
```

## Limitations

- Crop colour is apparent colour under the image's lighting and white balance, not verified vehicle
  paint colour. Reflections, shadows, windows, wheels, background pixels, small boxes, and occlusion
  can change the estimate. Gray versus silver and nearby hues may be ambiguous; low evidence is
  explicitly reported as `unknown`.
- YOLO11n is a general COCO detector, not a traffic-specific detector. Small, distant, occluded, or
  unusual vehicles and people can be missed or misclassified. Detection confidence is not a
  calibrated probability.
- Video processing is offline and may be slower than real time. The available container formats
  depend on the installed OpenCV backends and codecs; an advertised container extension does not
  guarantee that every encoded clip can be decoded or played by the browser. The UI displays the
  original and annotated first frames as still previews as well as native video players and a
  download; browser playback depends on browser-supported codecs.
- Video FPS is reported as source FPS when available, the output FPS used to write the result, and
  measured processing FPS. If source FPS metadata is absent or invalid, the annotated copy uses
  25 FPS. No unique vehicle/person counts are produced without tracking.
