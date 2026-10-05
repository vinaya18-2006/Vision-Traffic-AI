"""Frame-by-frame video inference and annotated video writing."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping, Protocol

import cv2
import numpy as np

from .contracts import CarDetection, PersonDetection, VideoFrameResult, VideoInferenceResult
from .errors import MediaInputError, VideoOutputError
from .utils import annotate_detections, supported_video_extensions

DEFAULT_OUTPUT_FPS = 25.0
DEFAULT_MAX_VIDEO_FRAMES = 3000


class _DetectorMetadata(Protocol):
    metadata: Mapping[str, object]


class _ColourThreshold(Protocol):
    minimum_confidence: float


class _FramePipeline(Protocol):
    detector: _DetectorMetadata
    confidence_threshold: float
    colour_classifier: _ColourThreshold

    def detect_objects(
        self, image_bgr: np.ndarray
    ) -> tuple[tuple[CarDetection, ...], tuple[PersonDetection, ...]]: ...


def infer_video(
    pipeline: _FramePipeline,
    video_path: Path | str,
    output_video_path: Path | str,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> VideoInferenceResult:
    """Annotate every decodable frame and report per-frame counts and FPS.

    The caller owns the output file and is responsible for its later cleanup.
    This pipeline does not track identities; it never reports unique-object
    totals.
    """
    source = Path(video_path).expanduser().resolve()
    output = Path(output_video_path).expanduser().resolve()
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError("Maximum video frame count must be a positive integer.")
    supported_extensions, backends = supported_video_extensions()
    if source.suffix.lower() not in supported_extensions:
        supported_text = ", ".join(supported_extensions) or "none detected"
        raise MediaInputError(
            f"Unsupported video container '{source.suffix or 'no extension'}'. "
            f"Available OpenCV backends ({', '.join(backends) or 'none'}) expose: "
            f"{supported_text}."
        )
    if not source.is_file():
        raise MediaInputError(f"Video file was not found: '{source}'.")
    if source == output:
        raise VideoOutputError("Output video path must not overwrite the input video.")
    if output.exists():
        raise VideoOutputError(f"Output video already exists and will not be overwritten: '{output}'.")
    if not output.parent.is_dir():
        raise VideoOutputError(f"Output directory does not exist: '{output.parent}'.")

    try:
        capture = cv2.VideoCapture(str(source))
    except cv2.error as exc:
        raise MediaInputError(f"OpenCV could not initialize video decoding for '{source}': {exc}") from exc
    if not capture.isOpened():
        capture.release()
        raise MediaInputError(
            f"OpenCV could not open video '{source}'. Check that the file is valid "
            "and its codec is supported."
        )

    source_fps_value = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = (
        source_fps_value
        if math.isfinite(source_fps_value) and source_fps_value > 0
        else None
    )
    output_fps = source_fps or DEFAULT_OUTPUT_FPS
    writer: cv2.VideoWriter | None = None
    frame_dimensions: tuple[int, int] | None = None
    frame_results: list[VideoFrameResult] = []
    preview_frames: tuple[np.ndarray, np.ndarray] | None = None
    output_created = False
    frame_limit_reached = False
    started = time.perf_counter()
    try:
        frame_index = 0
        while True:
            if frame_index >= max_frames:
                try:
                    frame_limit_reached = capture.read()[0]
                except cv2.error as exc:
                    raise MediaInputError(
                        f"OpenCV failed while checking the frame limit for '{source}': {exc}"
                    ) from exc
                break
            try:
                success, frame = capture.read()
            except cv2.error as exc:
                raise MediaInputError(
                    f"OpenCV failed while decoding frame {frame_index} of '{source}': {exc}"
                ) from exc
            if not success:
                if frame_index == 0:
                    raise MediaInputError(f"No decodable video frames were found in '{source}'.")
                break
            if frame is None or frame.size == 0:
                raise MediaInputError(f"Video frame {frame_index} is empty or malformed.")

            height, width = frame.shape[:2]
            if writer is None:
                frame_dimensions = (width, height)
                fourcc = _video_fourcc(output.suffix)
                try:
                    writer = cv2.VideoWriter(
                        str(output),
                        fourcc,
                        output_fps,
                        frame_dimensions,
                    )
                except cv2.error as exc:
                    raise VideoOutputError(f"Could not initialize video writer: {exc}") from exc
                output_created = True
                if not writer.isOpened():
                    raise VideoOutputError(
                        f"OpenCV could not create '{output}'. The selected output codec "
                        "may not be available."
                    )
            elif (width, height) != frame_dimensions:
                raise MediaInputError(
                    f"Video frame {frame_index} dimensions differ from the first frame."
                )

            cars, people = pipeline.detect_objects(frame)
            annotated = annotate_detections(frame, cars, people)
            if frame_index == 0:
                preview_frames = (frame.copy(), annotated.copy())
            try:
                writer.write(annotated)
            except cv2.error as exc:
                raise VideoOutputError(f"Could not write video frame {frame_index}: {exc}") from exc
            frame_results.append(
                VideoFrameResult(frame_index=frame_index, cars=cars, people=people)
            )
            frame_index += 1
    except Exception:
        if writer is not None:
            writer.release()
        capture.release()
        if output_created:
            output.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.release()
        capture.release()

    processing_seconds = time.perf_counter() - started
    if not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise VideoOutputError(f"The annotated video was not written successfully to '{output}'.")
    if preview_frames is None:
        output.unlink(missing_ok=True)
        raise VideoOutputError("No annotated preview frame was produced.")
    metadata = dict(pipeline.detector.metadata)
    metadata.update(
        {
            "confidence_threshold": pipeline.confidence_threshold,
            "colour_classifier": "HSV crop colour features",
            "colour_confidence_threshold": pipeline.colour_classifier.minimum_confidence,
            "count_policy": "per-frame retained detections; object tracking is not implemented",
            "max_frames": max_frames,
        }
    )
    frame_count = len(frame_results)
    return VideoInferenceResult(
        output_video_path=output,
        frame_count=frame_count,
        source_fps=source_fps,
        output_fps=output_fps,
        processing_fps=frame_count / processing_seconds if processing_seconds > 0 else 0.0,
        frames=tuple(frame_results),
        inference_metadata=metadata,
        original_preview_frame=preview_frames[0],
        annotated_preview_frame=preview_frames[1],
        max_frames=max_frames,
        frame_limit_reached=frame_limit_reached,
    )


def _video_fourcc(extension: str) -> int:
    normalized = extension.lower()
    if normalized == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    if normalized in {".mp4", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    raise VideoOutputError(f"Unsupported annotated video output extension: {extension}.")
