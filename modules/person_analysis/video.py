"""Frame-by-frame person inference and annotated video writing."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import cv2

from .contracts import VideoAnalysisResult, VideoFrameResult
from .detector import PersonAnalysisPipeline, PersonDetector
from .errors import InferenceError, MediaInputError, VideoOutputError
from .utils import annotate_people, supported_video_extensions

DEFAULT_MAX_VIDEO_FRAMES = 3000
DEFAULT_OUTPUT_FPS = 25.0
ProgressCallback = Callable[[int, int], None]


def infer_video(
    pipeline: PersonAnalysisPipeline,
    video_path: Path | str,
    output_video_path: Path | str,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    progress_callback: ProgressCallback | None = None,
) -> VideoAnalysisResult:
    """Write an annotated copy and retain per-frame counts without tracking people."""
    source = Path(video_path).expanduser().resolve()
    output = Path(output_video_path).expanduser().resolve()
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("Maximum video frame count must be a positive integer.")
    supported, backends = supported_video_extensions()
    if source.suffix.lower() not in supported:
        raise MediaInputError(
            f"Unsupported video container '{source.suffix or 'no extension'}'. "
            f"Available OpenCV backends ({', '.join(backends) or 'none'}) expose: "
            f"{', '.join(supported) or 'none'}."
        )
    if not source.is_file():
        raise MediaInputError(f"Video file was not found: '{source}'.")
    if source == output:
        raise VideoOutputError("Output video path must not overwrite the input video.")
    if output.exists():
        raise VideoOutputError(
            f"Output video already exists and will not be overwritten: '{output}'."
        )
    if not output.parent.is_dir():
        raise VideoOutputError(f"Output directory does not exist: '{output.parent}'.")

    try:
        capture = cv2.VideoCapture(str(source))
    except cv2.error as exc:
        raise MediaInputError(f"Could not initialize video decoding for '{source}': {exc}") from exc
    try:
        if not capture.isOpened():
            raise MediaInputError(
                f"OpenCV could not open video '{source}'. Check that the file is valid "
                "and its codec is supported."
            )
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_total = max(1, min(reported_frames, max_frames)) if reported_frames > 0 else max_frames
    except cv2.error as exc:
        capture.release()
        raise MediaInputError(f"Could not inspect video metadata for '{source}': {exc}") from exc

    output_fps = source_fps or DEFAULT_OUTPUT_FPS
    writer: cv2.VideoWriter | None = None
    output_created = False
    dimensions: tuple[int, int] | None = None
    frames: list[VideoFrameResult] = []
    previews = None
    frame_limit_reached = False
    started = time.perf_counter()
    try:
        frame_index = 0
        while frame_index < max_frames:
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
                        f"OpenCV could not create '{output}'. The selected output codec "
                        "may not be available."
                    )
            elif dimensions != (width, height):
                raise MediaInputError(
                    f"Video frame {frame_index} dimensions differ from the first frame."
                )

            detections = pipeline.detect_people(frame)
            annotated = annotate_people(frame, detections)
            if frame_index == 0:
                previews = (frame.copy(), annotated.copy())
            try:
                writer.write(annotated)
            except cv2.error as exc:
                raise VideoOutputError(
                    f"Could not write annotated video frame {frame_index}: {exc}"
                ) from exc
            frames.append(VideoFrameResult(frame_index, detections))
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, progress_total)

        if frame_index == max_frames:
            try:
                frame_limit_reached = bool(capture.read()[0])
            except cv2.error as exc:
                raise MediaInputError(
                    f"OpenCV failed while checking the frame limit for '{source}': {exc}"
                ) from exc
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
    if previews is None:
        output.unlink(missing_ok=True)
        raise VideoOutputError("No annotated preview frame was produced.")
    _verify_output_video(output)

    metadata = dict(pipeline.detector.metadata)
    metadata.update(
        {
            "confidence_threshold": pipeline.confidence_threshold,
            "count_policy": "person detections reported per frame; tracking is not implemented",
            "max_frames": max_frames,
        }
    )
    frame_count = len(frames)
    return VideoAnalysisResult(
        output_video_path=output,
        frame_count=frame_count,
        source_fps=source_fps,
        output_fps=output_fps,
        processing_fps=frame_count / processing_seconds if processing_seconds > 0 else 0.0,
        frames=tuple(frames),
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
    try:
        verification = cv2.VideoCapture(str(path))
        try:
            if not verification.isOpened():
                raise VideoOutputError(
                    f"The annotated output video '{path}' cannot be decoded by OpenCV."
                )
            success, frame = verification.read()
            if not success or frame is None or frame.size == 0:
                raise VideoOutputError(
                    f"The annotated output video '{path}' contains no decodable frames."
                )
        finally:
            verification.release()
    except cv2.error as exc:
        raise VideoOutputError(f"Could not verify annotated output video '{path}': {exc}") from exc
    except VideoOutputError:
        path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise VideoOutputError(f"Could not read annotated output video '{path}': {exc}") from exc
