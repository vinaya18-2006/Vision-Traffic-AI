"""Per-frame static-letter inference for uploaded video."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import cv2

from .contracts import VideoFramePrediction, VideoRecognitionResult
from .detector import SignLanguagePipeline
from .errors import MediaInputError, VideoOutputError
from .preprocessing import annotate_prediction, supported_video_extensions

DEFAULT_MAX_VIDEO_FRAMES = 3000
DEFAULT_OUTPUT_FPS = 25.0
ProgressCallback = Callable[[int, int], None]


def infer_video(
    pipeline: SignLanguagePipeline,
    video_path: Path | str,
    output_video_path: Path | str,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    progress_callback: ProgressCallback | None = None,
) -> VideoRecognitionResult:
    """Classify frames independently; never aggregate letters into signs or words."""
    source = Path(video_path).expanduser().resolve()
    output = Path(output_video_path).expanduser().resolve()
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("Maximum video frame count must be a positive integer.")
    supported, backends = supported_video_extensions()
    if source.suffix.lower() not in supported:
        raise MediaInputError(
            f"Unsupported video container '{source.suffix or 'no extension'}'. "
            f"OpenCV backends ({', '.join(backends) or 'none'}) expose: "
            f"{', '.join(supported) or 'none'}."
        )
    if not source.is_file():
        raise MediaInputError(f"Video file was not found: '{source}'.")
    if source == output:
        raise VideoOutputError("Annotated output must not overwrite the source video.")
    if output.exists():
        raise VideoOutputError(
            f"Annotated output already exists and will not be overwritten: '{output}'."
        )
    if not output.parent.is_dir():
        raise VideoOutputError(f"Output directory does not exist: '{output.parent}'.")

    pipeline.prepare()
    try:
        capture = cv2.VideoCapture(str(source))
    except cv2.error as exc:
        raise MediaInputError(f"Could not initialize video decoding: {exc}") from exc
    try:
        if not capture.isOpened():
            raise MediaInputError(
                f"OpenCV could not open video '{source}'. Check the file and codec."
            )
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_total = (
            min(reported_frames, max_frames) if reported_frames > 0 else max_frames
        )
    except cv2.error as exc:
        capture.release()
        raise MediaInputError(f"Could not inspect video metadata: {exc}") from exc

    output_fps = source_fps or DEFAULT_OUTPUT_FPS
    writer: cv2.VideoWriter | None = None
    output_created = False
    dimensions: tuple[int, int] | None = None
    frame_results: list[VideoFramePrediction] = []
    previews = None
    frame_limit_reached = False
    started = time.perf_counter()
    try:
        frame_index = 0
        while frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                if frame_index == 0:
                    raise MediaInputError(f"No decodable video frames were found in '{source}'.")
                break
            if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                raise MediaInputError(f"Video frame {frame_index} is empty or malformed.")

            height, width = frame.shape[:2]
            if writer is None:
                dimensions = (width, height)
                try:
                    writer = cv2.VideoWriter(
                        str(output),
                        _video_fourcc(output.suffix),
                        output_fps,
                        dimensions,
                    )
                except cv2.error as exc:
                    raise VideoOutputError(f"Could not initialize video writer: {exc}") from exc
                output_created = True
                if not writer.isOpened():
                    raise VideoOutputError(
                        f"OpenCV could not create '{output}' with the selected codec."
                    )
            elif dimensions != (width, height):
                raise MediaInputError(
                    f"Video frame {frame_index} dimensions differ from the first frame."
                )

            timestamp = frame_index / (source_fps or output_fps)
            prediction = pipeline.infer_frame(
                frame,
                frame_index=frame_index,
                timestamp_seconds=timestamp,
            )
            annotated = annotate_prediction(frame, prediction)
            if frame_index == 0:
                previews = (frame.copy(), annotated.copy())
            writer.write(annotated)
            frame_results.append(
                VideoFramePrediction(frame_index, timestamp, prediction)
            )
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, max(progress_total, 1))

        if frame_index == max_frames:
            frame_limit_reached = bool(capture.read()[0])
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

    if not output.is_file() or output.stat().st_size == 0 or previews is None:
        output.unlink(missing_ok=True)
        raise VideoOutputError(f"Annotated video was not written successfully to '{output}'.")
    _verify_output_video(output)
    elapsed = time.perf_counter() - started
    metadata = {
        **pipeline.model_metadata,
        "language": "American Sign Language (ASL)",
        "recognition_scope": "independent static letters per frame; no temporal aggregation",
        "video_temporal_inference": False,
        "operation_window": (
            f"{pipeline.config.allowed_start:%H:%M}-{pipeline.config.allowed_end:%H:%M} "
            f"{pipeline.config.timezone_name} (start-inclusive/end-exclusive)"
        ),
    }
    return VideoRecognitionResult(
        output_video_path=output,
        frame_count=len(frame_results),
        source_fps=source_fps,
        output_fps=output_fps,
        frames=tuple(frame_results),
        inference_metadata=metadata,
        original_preview_frame=previews[0],
        annotated_preview_frame=previews[1],
        max_frames=max_frames,
        frame_limit_reached=frame_limit_reached,
    )


def _video_fourcc(extension: str) -> int:
    if extension.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    if extension.lower() in {".mp4", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    raise VideoOutputError(f"Unsupported annotated video extension: '{extension}'.")


def _verify_output_video(path: Path) -> None:
    verification = cv2.VideoCapture(str(path))
    try:
        if not verification.isOpened():
            raise VideoOutputError(f"Annotated video '{path}' cannot be reopened.")
        success, frame = verification.read()
        if not success or frame is None or frame.size == 0:
            raise VideoOutputError(f"Annotated video '{path}' contains no readable frames.")
    finally:
        verification.release()
