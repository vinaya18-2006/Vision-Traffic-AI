# Task 6 - Driver drowsiness detection (incomplete scaffold)

This module currently performs **real person detection only**. It uses the repository's existing
YOLO11n weights to count and annotate detected people in images and videos. It does **not** infer
alertness, drowsiness, sleeping, age, or facial/eye state. Drowsiness status and sleeping/drowsy
counts are explicitly unavailable, not zero. No red boxes are emitted because no validated
drowsiness model is configured.

The application is an experimental computer-vision aid, not a replacement for safe driving,
vehicle safety systems, rest, or medical assessment.

## Existing detector

- **Architecture:** Ultralytics YOLO11n, using COCO class `0` (`person`) only.
- **Weights:** repository-root `yolo11n.pt`, SHA-256
  `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`.
- **Runtime:** `ultralytics==8.4.173`, already in the project requirements.
- **License/provenance:** the local checkpoint's precise source revision and redistribution
  terms are not established in the repository. Review current Ultralytics and weight terms before
  redistribution or commercial use. The module reuses the already provisioned local file and
  never downloads weights automatically.
- Set `PERSON_ANALYSIS_MODEL_PATH` or `CAR_DETECTION_MODEL_PATH` as documented by Task 4 to use
  another local person detector.

Detection confidence is not a calibrated probability. Video tracklets use greedy frame-to-frame
intersection-over-union matching. They are approximate associations, not verified identities or
unique-person counts, and may split or merge under occlusion.

## Drowsiness model and data blocker

No temporal drowsiness classifier, face/eye landmark weights, drowsiness dataset, or representative
vehicle-interior video is provisioned in this repository. Therefore:

- Image results show person boxes only. A still frame cannot establish sustained drowsiness.
- Video results preserve frame order and source timestamps when FPS metadata is available, but
  perform no eye-state or temporal-state inference. The configured temporal window is informational
  only; short clips and low/unknown frame rates are called out.
- `alert`, `drowsy`, and `sleeping` are not currently supported runtime predictions. Positive red
  annotations require a model ID, score, and temporal evidence window; no production path supplies
  those without a future validated model.
- No age model is available, and no age is estimated.

### Investigated external candidates (not provisioned)

- [`chbh7051/vit-driver-drowsiness-detection`](https://huggingface.co/chbh7051/vit-driver-drowsiness-detection)
  is a static ViT image classifier tagged Apache-2.0 and associated with UTA-RLDD. Its card reports
  99.3% accuracy, but does not sufficiently describe the evaluation split, subject independence,
  intended use, or data provenance; the metric is unverified. It cannot provide temporal evidence
  and is not used.
- [`Simuletic/Driver_Drowsiness_Monitoring_Detection_Dataset`](https://huggingface.co/datasets/Simuletic/Driver_Drowsiness_Monitoring_Detection_Dataset)
  is tagged CC-BY-4.0 and describes a 168-frame synthetic sample across six short sequences. Its
  synthetic sample is not sufficient to validate generalization to real drivers and is not used.
- The Hugging Face derivative [`chbh7051/UTA-RLDD_images`](https://huggingface.co/datasets/chbh7051/UTA-RLDD_images)
  does not expose a dataset license in its metadata. Its terms and relationship to the original
  source must be verified before use.

No model or dataset was downloaded for Task 6. A usable completion requires authorized, consented,
real in-cabin video with temporal labels and documented license, plus subject-independent
evaluation across lighting, eyewear, pose, and occlusion. Train or fine-tune a temporal classifier
on the licensed sequences, define alert/drowsy/sleeping/unknown labels and fixed temporal windows,
then evaluate held-out subjects and report per-class precision/recall, false-alarm rate, missed
drowsiness rate, and calibration/uncertainty. Any optional age model needs a separate dataset,
license, and evaluation; it should remain disabled otherwise.

## Use and tests

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest tests\test_drowsiness.py tests\test_drowsiness_integration.py tests\test_drowsiness_ui.py -v
```

Video supports only the containers and codecs exposed by the local OpenCV build. Processing is
offline and capped at 3,000 frames by default. Per-frame person counts and heuristic tracklet
counts are shown separately. No sleeping/drowsy totals or persistent state are claimed.
