# Task 2 — Wildlife Risk Intelligence

The module performs real local object detection through Ultralytics YOLO11n, maps supported model
class labels through a separate versioned dietary taxonomy, and reports image- or frame-level
results. Inference and taxonomy logic are independent of the Streamlit UI.

## Model and label coverage

- **Baseline:** Ultralytics YOLO11n, using the existing COCO-pretrained `yolo11n.pt` weights.
  The artifact is loaded from the repository root by default; its SHA-256 is
  `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`.
- **Library:** Ultralytics `8.4.173` in the project's pinned environment. Standard Ultralytics
  preprocessing and detection post-processing are used; no custom training or extra classifier
  is claimed.
- **Animal labels present in this model:** `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`,
  `elephant`, `bear`, `zebra`, and `giraffe`. These are the actual supported animal classes from
  its COCO label map. This is not a fine-grained species model: for example, “bird” is not a bird
  species, and this model does not label deer, foxes, wolves, lions, or tigers.
- Only those configured animal labels are requested from the model. Non-animal COCO classes are
  excluded from wildlife counts.
- Missing or corrupt local weights are reported as errors. There is no automatic download,
  fallback model, fabricated output, or claim of custom training.

Use `WILDLIFE_MODEL_PATH` to configure a different local Ultralytics `.pt` model. A replacement
model must use labels that match `animal_classes` in the taxonomy file. Extend that list and add an
explicit classification for every label; use `unknown` when diet is not established. This makes an
animal-specific model configurable without pretending the COCO baseline can recognize its classes.
Review Ultralytics' current [license and model terms](https://github.com/ultralytics/ultralytics)
before use or redistribution.

## Dietary taxonomy

`taxonomy.json` is independently versioned (`1.0.0`), records its source, references, and
context-dependent cases. Configure an alternate file with `WILDLIFE_TAXONOMY_PATH`. Every
configured animal class needs one explicit `carnivore`, `non_carnivore`, or `unknown` entry.
Unmapped categories are never silently treated as non-carnivores.

The baseline's broad labels constrain dietary certainty: `cat` is mapped to carnivore;
`horse`, `sheep`, `cow`, `elephant`, `zebra`, and `giraffe` are mapped to non-carnivore.
`bird`, `dog`, and `bear` are unknown because each model label spans varied/context-dependent
diets. Unknown detections are neutral grey and do not increase the carnivore count. Mappings are
based on the model label, not the animal's appearance or a presumed species not identified by it.

## API

```python
from modules.wildlife import WildlifePipeline, infer_video

pipeline = WildlifePipeline(confidence_threshold=0.25)
image_result = pipeline.infer_image("path/to/image.jpg")
video_result = infer_video(
    pipeline,
    "path/to/input.mp4",
    "path/to/annotated.mp4",
    max_frames=3000,
)
```

Each detection contains the model's animal class label, confidence, XYXY pixel box, and dietary
classification. Confidence is the detector score, not a calibrated probability. Carnivore boxes
are red (BGR `(0, 0, 255)`), non-carnivores are blue (BGR `(255, 0, 0)`), and unknown classes are
neutral grey.

Video is processed frame by frame with a 3,000-frame default safety cap. Results report processed
frames, source FPS when available, processing FPS, counts for each processed frame, and summaries
over **detection instances across frames**. They do not report unique animals; tracking is not
implemented. Encoded output is reopened and checked before success is returned. Container and
codec availability depends on the installed OpenCV backends.

## Run

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest tests\test_wildlife.py
.\.venv\Scripts\python.exe -m pytest tests\test_wildlife_integration.py -m integration
```

The integration smoke test uses the already-provisioned local `yolo11n.pt`; ordinary wildlife
unit tests mock the detector and never download model weights.

## Limitations

This baseline is a general COCO object detector, not an animal-specific wildlife detector. Its
labels are broad object classes rather than species identities, and small, distant, occluded, or
out-of-domain animals may be missed or mislabeled. Scores are not calibrated probabilities.
Additional wildlife classes require suitable animal-specific pretrained weights or appropriately
licensed training data and verified training; no such training is included here. Dietary categories
are deterministic taxonomy lookups, not model predictions, and broad/context-dependent labels stay
unclassified. Video results count frame detections, not distinct animals.
