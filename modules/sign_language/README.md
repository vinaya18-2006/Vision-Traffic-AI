# Task 5 — Static ASL fingerspelling pilot

This module recognizes one static ASL fingerspelled letter from one visible hand. It is an
experimental, finite-vocabulary classifier, not ASL translation. It does not recognize J or Z,
dynamic signs, words, sentences, grammar, or multiple-hand signs.

## Model, vocabulary, and provenance

- Language and scope: American Sign Language (ASL), static letters A–Y except J and Z.
- Landmark extractor: MediaPipe Hands, returning 21 normalized image-space `(x, y, z)` points.
- Classifier: publisher-provided Keras checkpoint from
  [`AmimulBmeIU/asl-alphabet-classifier`](https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier),
  pinned at revision `9e5cde38db66ae889322aeea3c547848676fa96e`.
- Checkpoint size: 245,970 bytes. SHA-256:
  `9A73CD144A41C2CA92BD4D07ECEC2FE32F5F4DDE5E28C8BCC0B0ADA013502F11`.
- The publisher declares MIT and reports 0.9415 validation accuracy. Its card does not identify
  training data, signer split, or evaluation protocol. The metric is not independently verified;
  underlying training-data licensing and signer diversity therefore remain unverified. Do not
  treat this pilot as validated for accessibility, production, or high-stakes use.
- Scores are uncalibrated model outputs. The configurable cutoff withholds low-score letters, but
  the model was not trained with an out-of-vocabulary class; a high score on an unsupported sign
  can still be wrong. No robust OOV performance is claimed.
- Keras uses the project's existing PyTorch runtime. MediaPipe/Keras extras are pinned in
  `requirements-sign-language.txt`; no TensorFlow runtime is required.

The vocabulary and model provenance are recorded in `vocabulary.json` and `model_manifest.json`.
No model weights or dataset are committed.

## Setup and execution

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-sign-language.txt
.\.venv\Scripts\python.exe -m modules.sign_language.provision_model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Provisioning downloads only the pinned 245,970-byte checkpoint and verifies its SHA-256. It never
runs automatically during application import. The default local path is
`models/sign_language/asl_landmarks.keras`; set `SIGN_LANGUAGE_MODEL_PATH` to use another local
copy of the same pinned checkpoint.

The daily authorization window defaults to 09:00–17:00 in `America/New_York`, including the start
and excluding the end. Configure it with:

- `SIGN_LANGUAGE_ALLOWED_START` (`HH:MM`)
- `SIGN_LANGUAGE_ALLOWED_END` (`HH:MM`)
- `SIGN_LANGUAGE_TIMEZONE` (IANA timezone)
- `SIGN_LANGUAGE_MINIMUM_SCORE` (number from 0 to 1)

Authorization is checked before model loading and again for every video frame. Closed or invalid
windows fail explicitly.

## Input and result semantics

Images are validated and converted through the existing Pillow/OpenCV image decoder. Tiny or
visually degenerate images are rejected. The classifier expects exactly one detected hand. Missing
hands return `no_hand`; multiple hands return `unknown`. A below-threshold model output is shown
as `unknown` with its raw model score. An annotation is produced as a copy; uploads are not
overwritten.

Uploaded video is decoded in frame order, annotated, and capped at 3,000 frames by default.
Every frame is classified independently as a static letter. Results include frame index and
timestamp; there is no temporal model, smoothing, event segmentation, letter aggregation, or word
formation. This is not dynamic-sign recognition.

The configured image/video formats are determined by the installed Pillow and OpenCV backends;
file extensions do not guarantee codec support. Real-time webcam video is not implemented.

## Tests and verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sign_language.py -v
.\.venv\Scripts\python.exe -m pytest tests\test_sign_language_integration.py -v
.\.venv\Scripts\python.exe -m pytest tests\test_sign_language_ui.py -v
```

Unit tests use test doubles only for plumbing, validation, and error cases. The real-model smoke
test can run on the repository's existing non-sign scene as a no-fabricated-output check; it does
not measure recognition accuracy. To run the labeled acceptance test, set
`SIGN_LANGUAGE_TEST_IMAGE` to a genuine, authorized sign image and
`SIGN_LANGUAGE_TEST_LABEL` to its verified letter. A suitable labeled test image is not currently
provisioned. Until a signer-diverse, independently labeled held-out set and model training
provenance are available, real sign accuracy and generalization remain unverified.

## Limitations

The classifier's training dataset and signer demographics are unknown. The declared 24-letter
vocabulary is fingerspelling, not a complete sign language. J/Z require motion; video is only a
series of isolated static-frame guesses. The threshold is not calibrated, and unknown handling is
not a validated OOV detector. MediaPipe can miss small, occluded, poorly lit, or non-frontal hands.
Capture a clear single hand against a contrasting background; do not use results as a substitute
for an interpreter.
