# Task 4 — Person Attribute & Emotion Analysis

This module currently performs **person detection only**. It reports detected person boxes and
confidence scores for images and per-frame video detections. It does not estimate age or facial
expression because no suitable validated model components for those features are included.
Detection confidence is not an age or emotion score. Nationality and ethnicity are never inferred
from appearance; this task has no nationality/region input because no documented downstream branch
requires one.

## Detector and local weights

- **Model:** Ultralytics YOLO11n, a general object detector. The person class is COCO class ID
  `0`; the configured model is checked for that label at load time, and inference requests only
  that class.
- **Local artifact:** The project uses the existing repository-root `yolo11n.pt` file by default.
  The code and other task documentation identify this artifact as YOLO11n and list the official
  [YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/) as the acquisition
  reference. The exact origin/revision of the local file cannot be established from the file
  itself. The SHA-256 is recorded in result metadata at runtime.
- **Library:** `ultralytics==8.4.173` in the current project requirements.
- **License:** Ultralytics software is distributed under AGPL-3.0; commercial use may require a
  separate license. The local weights' independent license/redistribution terms are not recorded
  in the project and must be reviewed before redistribution or commercial use.
- **Loading:** `PERSON_ANALYSIS_MODEL_PATH` may point to another existing local `.pt` file.
  Otherwise the module uses `CAR_DETECTION_MODEL_PATH` when configured, then the existing Task 1
  default path at the repository root. Missing or incompatible local weights are explicit errors.
  The module passes an existing path to Ultralytics and does not download weights.
- **Preprocessing:** Ultralytics performs its standard image resize/normalization and detection
  post-processing. Returned boxes are in source-frame pixel coordinates. Valid boxes are clipped
  to source image bounds before presentation and annotation.

To prepare an environment, follow the repository's Windows setup and pinned dependencies in
`requirements.txt`. No additional dependency or model provisioning step is required when the
existing local weights are available:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest tests\test_person_analysis.py
.\.venv\Scripts\python.exe -m pytest tests\test_person_analysis_integration.py
```

Python usage:

```python
from modules.person_analysis import PersonAnalysisPipeline

pipeline = PersonAnalysisPipeline(confidence_threshold=0.25)
result = pipeline.infer_image("path/to/image.jpg")
print(result.person_count, result.detections)
```

## Media and result semantics

Image formats are those supported by the existing Pillow-based image decoder (JPEG, PNG, and
WebP when decoder support is available). Video containers depend on the installed OpenCV backend;
the current shared utility exposes MP4, AVI, and MOV candidates when a video backend is present.
An extension does not guarantee that a particular codec can be decoded. Corrupt/empty images,
unsupported formats, invalid videos, and encoder/decoder failures are reported rather than treated
as empty detections.

The confidence threshold excludes lower-scoring detections; the reported values are model scores,
not calibrated probabilities. A still image count is the number of retained person boxes in that
image. Video results show detections and counts separately for each processed frame. No detections
are summed and represented as unique people; tracking is not implemented. Video processing is
offline, capped at 3,000 frames by default, and may be slower than real time.

## Limitations and unavailable analysis

YOLO11n is a general detector, not an identity or attribute model. Small, distant, occluded, or
unusual poses may be missed, and confidence scores are uncertain. No face crops are retained for
additional processing. Age estimation and facial-expression analysis remain disabled until
appropriate models, their provenance, licenses, evaluation, and limitations are documented and
validated. The UI explicitly identifies these features as unavailable rather than substituting
guesses.

