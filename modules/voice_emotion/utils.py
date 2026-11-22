"""Portable WAV decoding and validation for voice-emotion inference."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from .errors import MediaInputError

SUPPORTED_AUDIO_EXTENSIONS = (".wav",)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 30.0
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 96_000


def supported_audio_extensions() -> tuple[str, ...]:
    """Return formats supported by the built-in decoder."""
    return SUPPORTED_AUDIO_EXTENSIONS


def _decode_pcm_frames(
    frames: bytes,
    sample_width: int,
    frame_count: int,
    channels: int,
) -> np.ndarray:
    """Decode uncompressed integer PCM frames to normalized mono float32."""
    expected_bytes = frame_count * channels * sample_width
    if len(frames) != expected_bytes:
        raise MediaInputError("WAV audio data is truncated or has an invalid frame count.")

    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8)
        if packed.size % 3:
            raise MediaInputError("WAV audio contains an incomplete 24-bit PCM sample.")
        packed = packed.reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8_388_608.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        raise MediaInputError(
            f"Unsupported WAV PCM sample width: {sample_width * 8} bits."
        )

    expected_samples = frame_count * channels
    if samples.size != expected_samples:
        raise MediaInputError("WAV audio data is truncated or has an invalid frame count.")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.asarray(samples, dtype=np.float32)


def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV payload into normalized mono audio."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise MediaInputError("Only uncompressed PCM WAV files are supported.")

            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            duration = frame_count / sample_rate if sample_rate else 0.0
            if channels not in (1, 2):
                raise MediaInputError("WAV audio must have one or two channels.")
            if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
                raise MediaInputError(
                    f"WAV sample rate must be between {MIN_SAMPLE_RATE} and "
                    f"{MAX_SAMPLE_RATE} Hz."
                )
            if not 0.0 < duration <= MAX_AUDIO_DURATION_SECONDS:
                raise MediaInputError(
                    f"Audio duration must be greater than zero and no more than "
                    f"{MAX_AUDIO_DURATION_SECONDS:g} seconds."
                )
            frames = wav_file.readframes(frame_count)
    except MediaInputError:
        raise
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        raise MediaInputError(f"Audio file is malformed or cannot be decoded: {exc}") from exc

    waveform = _decode_pcm_frames(frames, sample_width, frame_count, channels)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise MediaInputError("Audio file contains no usable finite samples.")
    if float(np.sqrt(np.mean(np.square(waveform)))) < 1e-4:
        raise MediaInputError("Audio is silent or too quiet to analyze.")
    return waveform, sample_rate


def decode_audio(
    audio_source: Path | str | bytes | bytearray | memoryview,
) -> tuple[np.ndarray, int]:
    """Decode a bounded WAV file or WAV bytes to normalized mono float32 samples."""
    if isinstance(audio_source, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(audio_source)
        suffix = ".wav"
    elif isinstance(audio_source, (str, Path)):
        path = Path(audio_source).expanduser()
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise MediaInputError(
                f"Unsupported audio format '{suffix or 'no extension'}'. "
                f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}."
            )
        try:
            if not path.is_file():
                raise MediaInputError(f"Audio file does not exist or is not a file: {path}")
            if path.stat().st_size > MAX_AUDIO_SIZE_BYTES:
                raise MediaInputError(
                    f"Audio file exceeds the {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MiB size limit."
                )
            audio_bytes = path.read_bytes()
        except MediaInputError:
            raise
        except OSError as exc:
            raise MediaInputError(f"Could not read audio file '{path}': {exc}") from exc
    else:
        raise MediaInputError("Unsupported audio source type; provide a WAV path or WAV bytes.")

    if not audio_bytes:
        raise MediaInputError("Audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise MediaInputError(
            f"Audio file exceeds the {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MiB size limit."
        )
    if suffix != ".wav":
        raise MediaInputError(f"Unsupported audio format '{suffix}'.")
    return _decode_wav_bytes(audio_bytes)


def resample_audio(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample validated mono audio with a polyphase anti-aliasing filter."""
    if source_rate <= 0 or target_rate <= 0:
        raise MediaInputError("Audio sample rates must be positive.")
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly
    except ImportError as exc:
        raise MediaInputError(
            "Resampling audio requires the scipy dependency; install the pinned "
            "Task 3 requirements."
        ) from exc

    divisor = gcd(source_rate, target_rate)
    result = resample_poly(
        np.asarray(waveform, dtype=np.float32),
        target_rate // divisor,
        source_rate // divisor,
    )
    return np.asarray(result, dtype=np.float32)
