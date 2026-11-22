"""Safe, frame-by-frame wildlife video inference and annotation."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .contracts import (
    VideoAnalysisResult,
    VideoFrameResult,
)
from .detector import WildlifePipeline
from .errors import MediaInputError, VideoOutputError
from .utils import annotate_detections, summarize_detections, supported_video_extensions

DEFAULT_MAX_VIDEO_FRAMES = 3000
DEFAULT_OUTPUT_FPS = 25.0


def infer_video(
    pipeline: WildlifePipeline,
    video_path: Path | str,
    output_video_path: Path | str,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> VideoAnalysisResult:
    """Process frames up to the safety cap and report frame-level detection totals.

    Detection counts are summed across processed frames. This routine does not
    track object identities and therefore never claims unique-animal totals.
    The caller owns and cleans up the output video.
    """
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
        is_opened = capture.isOpened()
    except cv2.error as exc:
        capture.release()
        raise MediaInputError(f"Could not inspect video decoder for '{source}': {exc}") from exc
    if not is_opened:
        capture.release()
        raise MediaInputError(
            f"OpenCV could not open video '{source}'. Check that the file is valid "
            "and its codec is supported."
        )

    try:
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    except cv2.error as exc:
        capture.release()
        raise MediaInputError(f"Could not read video metadata for '{source}': {exc}") from exc
    source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
    output_fps = source_fps or DEFAULT_OUTPUT_FPS
    writer: cv2.VideoWriter | None = None
    output_created = False
    dimensions: tuple[int, int] | None = None
    frame_results: list[VideoFrameResult] = []
    previews: tuple[np.ndarray, np.ndarray] | None = None
    frame_limit_reached = False
    started = time.perf_counter()
    try:
        frame_index = 0
        while True:
            if frame_index >= max_frames:
                try:
                    frame_limit_reached = bool(capture.read()[0])
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
            if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                raise MediaInputError(f"Video frame {frame_index} is empty or malformed.")

            height, width = frame.shape[:2]
            if writer is None:
                dimensions = (width, height)
                fourcc = _video_fourcc(output.suffix)
                try:
                    writer = cv2.VideoWriter(str(output), fourcc, output_fps, dimensions)
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

            detections = pipeline.detect_objects(frame)
            annotated = annotate_detections(frame, detections)
            if frame_index == 0:
                previews = frame.copy(), annotated.copy()
            try:
                writer.write(annotated)
            except cv2.error as exc:
                raise VideoOutputError(
                    f"Could not write annotated video frame {frame_index}: {exc}"
                ) from exc
            frame_results.append(
                VideoFrameResult(frame_index=frame_index, detections=detections)
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

    elapsed = time.perf_counter() - started
    if not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise VideoOutputError(f"The annotated video was not written successfully to '{output}'.")
    if previews is None:
        output.unlink(missing_ok=True)
        raise VideoOutputError("No annotated video preview frame was produced.")
    _verify_encoded_video(output)

    detections = tuple(
        detection for frame in frame_results for detection in frame.detections
    )
    metadata = dict(pipeline.detector.metadata)
    metadata.update(
        pipeline._analysis_metadata(elapsed * 1000.0)
    )
    metadata.update(
        {
            "count_policy": "sum of retained detections over processed frames; no tracking",
            "max_frames": max_frames,
        }
    )
    frame_count = len(frame_results)
    return VideoAnalysisResult(
        output_video_path=output,
        frame_count=frame_count,
        source_fps=source_fps,
        processing_fps=frame_count / elapsed if elapsed > 0 else 0.0,
        frames=tuple(frame_results),
        summary=summarize_detections(detections),
        inference_metadata=metadata,
        original_preview_frame=previews[0],
        annotated_preview_frame=previews[1],
        max_frames=max_frames,
        frame_limit_reached=frame_limit_reached,
    )


def _verify_encoded_video(path: Path) -> None:
    try:
        capture = cv2.VideoCapture(str(path))
    except cv2.error as exc:
        path.unlink(missing_ok=True)
        raise VideoOutputError(
            f"Could not initialize the output decoder for encoded video '{path}': {exc}"
        ) from exc
    try:
        is_opened = capture.isOpened()
    except cv2.error as exc:
        capture.release()
        path.unlink(missing_ok=True)
        raise VideoOutputError(
            f"Could not inspect the encoded video codec for '{path}': {exc}"
        ) from exc
    if not is_opened:
        capture.release()
        path.unlink(missing_ok=True)
        raise VideoOutputError(
            f"The output video codec could not be reopened for verification: '{path}'."
        )
    try:
        success, frame = capture.read()
    except cv2.error as exc:
        path.unlink(missing_ok=True)
        raise VideoOutputError(f"Could not verify encoded video '{path}': {exc}") from exc
    finally:
        capture.release()
    if not success or frame is None or frame.size == 0:
        path.unlink(missing_ok=True)
        raise VideoOutputError(f"The encoded video contains no decodable frames: '{path}'.")


def _video_fourcc(extension: str) -> int:
    normalized = extension.lower()
    if normalized == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    if normalized in {".mp4", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    raise VideoOutputError(f"Unsupported annotated video output extension: {extension}.")
