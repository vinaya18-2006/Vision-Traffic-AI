# Task 3 — Voice Emotion Detection

This module is prepared to run the pretrained `superb/wav2vec2-base-superb-er` utterance-level
speech-emotion model from a **local directory only**. It does not train a model, download weights,
use a random classifier, or substitute handcrafted audio features for emotion predictions. Until
the user explicitly provisions the model and approves its dependencies, inference reports a
clear missing-model error.

## Candidate model and provenance

- **Model:** SUPERB Wav2Vec2-Base emotion-recognition port,
  [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er)
- **Source:** [Hugging Face model repository](https://huggingface.co/superb/wav2vec2-base-superb-er);
  model card describes it as a port of the S3PRL SUPERB emotion-recognition model.
- **Supported labels:** neutral (`neu`), happy (`hap`), angry (`ang`), and sad (`sad`).
- **Input:** mono 16 kHz speech. This module currently accepts PCM WAV only; uploaded data is
  decoded with Python's built-in `wave` module and is not retained beyond the Streamlit session.
- **Weights size:** Hugging Face lists `pytorch_model.bin` as **378,361,105 bytes** (about 378 MB
  or 361 MiB), plus small config files.
- **License:** The model card declares Apache-2.0. It was trained on IEMOCAP through the SUPERB
  benchmark; the model-card license does not grant rights to the underlying IEMOCAP recordings.
  Review IEMOCAP's separate access/use terms before any commercial or redistribution use.
- **Evaluation caveat:** The model card reports 0.6258 accuracy for its Transformers port on one
  SUPERB/IEMOCAP session. This is reported external benchmark context, not a result measured by
  this application and not a guarantee for other languages, speakers, or recording conditions.

## Environment and explicit provisioning

The validated project environment is Windows, Python 3.11.9, with PyTorch 2.14.1+cpu. The Task 3
inference dependencies are Transformers 4.49.0 and SciPy 1.15.3; the selected Transformers version
supports Python 3.9+ and requires PyTorch 1.11+, so this does not upgrade the project's existing
PyTorch. Transformers' Hugging Face Hub, Tokenizers, and Safetensors dependencies are installed as
package dependencies. SciPy supplies polyphase resampling for valid WAV uploads whose sample rate
is not 16 kHz. No torchaudio, librosa, or FFmpeg package is needed for the PCM WAV input path.

The default directory is `models/voice_emotion/`; set `VOICE_EMOTION_MODEL_DIR` to another local
folder if needed. Install the pinned inference dependencies, then explicitly provision only the
model files used for inference:

```powershell
.\.venv\Scripts\python.exe -m pip install transformers==4.49.0 scipy==1.15.3
.\.venv\Scripts\python.exe -m modules.voice_emotion.provision_model
```

The provisioning command downloads and verifies only `config.json`,
`preprocessor_config.json`, and `pytorch_model.bin` from pinned revision
`441a7599c3b22107314dcbd9166621c5c83f2cc5`. It does not download sample audio, datasets, or
training assets. Existing files are re-verified and skipped; interrupted downloads are resumed
when the Hub supports byte ranges, and the checkpoint is installed only after its pinned size
and SHA-256 match. Inference calls Transformers with `local_files_only=True`, so startup and
predictions cannot fetch anything from the network. The model directory is git-ignored.

```python
from modules.voice_emotion import VoiceEmotionPipeline

pipeline = VoiceEmotionPipeline()
result = pipeline.infer_audio("path/to/16khz-mono-speech.wav")
print(result.label, result.confidence)
```

The returned score is a model output, not a calibrated probability. A low-score threshold adds a
warning; it does not rewrite one emotion as another. The class describes a model prediction and is
not a definitive statement about a speaker's internal state. No gender, identity, nationality, or
other unrelated attribute is inferred.

## Validation and tests

The built-in decoder accepts only 8/16/24/32-bit PCM WAV, one or two channels, sample rates from
8–96 kHz, files no larger than 25 MiB, durations up to 30 seconds, and non-silent audio. Stereo is
downmixed; other valid rates are resampled to 16 kHz with SciPy's polyphase filter. Source and
model sample rates are reported separately.

Run module tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_voice_emotion.py
```

Unit tests validate media decoding, resampling, size/duration/silence checks, and the explicit
missing-model error. Real-weight inference requires a user-provided speech WAV; tests must not
describe synthetic tones or synthetic training data as emotion predictions.

## Limitations

This English-language model was evaluated on the IEMOCAP/SUPERB benchmark; audio outside that
domain may behave differently. Emotion labels and scores can be affected by language, prosody,
recording quality, and dataset limitations. The task estimates a broad utterance-level class and
does not identify speakers, infer gender, or establish a person's actual emotional state.
