from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from modules.voice_emotion.detector import (
    MODEL_LABELS,
    MODEL_SAMPLE_RATE,
    VoiceEmotionPipeline,
    default_model_path,
)
from modules.voice_emotion.contracts import EmotionPrediction
from modules.voice_emotion.errors import MediaInputError, ModelLoadError
from modules.voice_emotion.utils import (
    MAX_AUDIO_DURATION_SECONDS,
    MAX_AUDIO_SIZE_BYTES,
    decode_audio,
    resample_audio,
    supported_audio_extensions,
)


def _wav_bytes(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1 and channels > 1:
        samples = np.repeat(samples[:, np.newaxis], channels, axis=1)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return stream.getvalue()


def _speech_samples(duration_seconds: float = 0.25, sample_rate: int = 16_000) -> np.ndarray:
    sample_count = int(duration_seconds * sample_rate)
    times = np.arange(sample_count, dtype=np.float32) / sample_rate
    return 0.4 * np.sin(2 * np.pi * 220 * times)


def test_decoder_handles_uploaded_wav_bytes_and_normalizes_pcm() -> None:
    waveform, sample_rate = decode_audio(_wav_bytes(_speech_samples()))

    assert sample_rate == MODEL_SAMPLE_RATE
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size == int(0.25 * sample_rate)
    assert np.max(np.abs(waveform)) <= 1.0
    assert np.max(np.abs(waveform)) > 0.3


def test_decoder_reads_portable_temporary_file_and_downmixes_stereo(tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(_wav_bytes(_speech_samples(), channels=2))

    waveform, sample_rate = decode_audio(wav_path)

    assert sample_rate == MODEL_SAMPLE_RATE
    assert waveform.ndim == 1
    assert waveform.size == int(0.25 * sample_rate)


def test_resample_audio_converts_to_model_rate_and_preserves_duration() -> None:
    source_rate = 44_100
    waveform = _speech_samples(duration_seconds=0.25, sample_rate=source_rate)

    converted = resample_audio(waveform, source_rate, MODEL_SAMPLE_RATE)

    assert converted.dtype == np.float32
    assert converted.size == int(0.25 * MODEL_SAMPLE_RATE)
    assert np.isfinite(converted).all()


@pytest.mark.parametrize(
    ("audio_bytes", "message"),
    [
        (b"", "empty"),
        (b"not a WAV file", "malformed or cannot be decoded"),
        (_wav_bytes(np.zeros(1600, dtype=np.float32)), "silent or too quiet"),
    ],
)
def test_decoder_rejects_empty_malformed_and_silent_audio(
    audio_bytes: bytes,
    message: str,
) -> None:
    with pytest.raises(MediaInputError, match=message):
        decode_audio(audio_bytes)


def test_decoder_rejects_unsupported_source_and_file_format(tmp_path: Path) -> None:
    with pytest.raises(MediaInputError, match="Unsupported audio source type"):
        decode_audio(None)  # type: ignore[arg-type]

    mp3_path = tmp_path / "voice.mp3"
    mp3_path.write_bytes(b"audio")
    with pytest.raises(MediaInputError, match="Unsupported audio format"):
        decode_audio(mp3_path)

    assert supported_audio_extensions() == (".wav",)


def test_decoder_enforces_sample_rate_duration_and_size_limits(tmp_path: Path) -> None:
    low_rate = _wav_bytes(_speech_samples(sample_rate=4_000), sample_rate=4_000)
    with pytest.raises(MediaInputError, match="sample rate"):
        decode_audio(low_rate)

    long_audio = _wav_bytes(
        _speech_samples(MAX_AUDIO_DURATION_SECONDS + 0.1),
        sample_rate=MODEL_SAMPLE_RATE,
    )
    with pytest.raises(MediaInputError, match="duration"):
        decode_audio(long_audio)

    oversized = tmp_path / "oversized.wav"
    oversized.write_bytes(b"x" * (MAX_AUDIO_SIZE_BYTES + 1))
    with pytest.raises(MediaInputError, match="size limit"):
        decode_audio(oversized)


def test_missing_pretrained_model_is_reported_without_network_fallback(
    tmp_path: Path,
) -> None:
    missing_model_dir = tmp_path / "model-not-provisioned"

    with pytest.raises(ModelLoadError, match="Missing files"):
        VoiceEmotionPipeline(model_path=missing_model_dir)


def test_default_model_directory_is_local_and_labels_match_candidate() -> None:
    assert default_model_path().name == "voice_emotion"
    assert set(MODEL_LABELS.values()) == {"angry", "happy", "neutral", "sad"}


def test_pipeline_does_not_substitute_a_label_for_low_model_score() -> None:
    class _LowScoreClassifier:
        metadata = {"model_id": "unit-test-only"}

        def predict(
            self,
            waveform: np.ndarray,
            sample_rate: int,
        ) -> EmotionPrediction:
            return EmotionPrediction("sad", 0.1, 0.1)

    pipeline = VoiceEmotionPipeline(
        classifier=_LowScoreClassifier(),
        confidence_threshold=0.5,
    )
    result = pipeline.infer_audio(
        _wav_bytes(_speech_samples(sample_rate=44_100), sample_rate=44_100)
    )

    assert result.label == "sad"
    assert result.confidence == pytest.approx(0.1)
    assert result.warning is not None
    assert result.sample_rate == MODEL_SAMPLE_RATE
    assert result.inference_metadata["source_sample_rate"] == 44_100
