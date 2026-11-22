"""Frame-ordered video person detection and IoU tracklet association."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2

from modules.person_analysis.contracts import BoundingBox, PersonDetection
from modules.person_analysis.utils import supported_video_extensions

from .contracts import PersonAssessment, VideoAnalysisResult, VideoFrameResult
from .detector import DrowsinessPipeline
from .errors import DrowsinessMediaError, DrowsinessVideoError
from .preprocessing import annotate_people, image_quality_warnings

DEFAULT_MAX_VIDEO_FRAMES = 3000
DEFAULT_OUTPUT_FPS = 25.0
MIN_USABLE_VIDEO_FPS = 5.0
ProgressCallback = Callable[[int, int], None]


@dataclass
class _Tracklet:
    track_id: int
    box: BoundingBox
    last_frame: int


class GreedyIoUTracker:
    """Associate nearby frame detections by IoU; output is not identity tracking."""

    def __init__(self, minimum_iou: float = 0.3, max_gap_frames: int = 5) -> None:
        if (
            isinstance(minimum_iou, bool)
            or not isinstance(minimum_iou, (int, float))
            or not math.isfinite(minimum_iou)
            or not 0 <= minimum_iou <= 1
        ):
            raise ValueError("Tracker IoU threshold must be between 0 and 1.")
        if (
            isinstance(max_gap_frames, bool)
            or not isinstance(max_gap_frames, int)
            or max_gap_frames < 0
        ):
            raise ValueError("Tracker maximum gap must be non-negative.")
        self.minimum_iou = minimum_iou
        self.max_gap_frames = max_gap_frames
        self._tracks: dict[int, _Tracklet] = {}
        self._next_track_id = 1
        self.created_track_count = 0

    def update(
        self,
        detections: Sequence[PersonDetection],
        frame_index: int,
    ) -> tuple[int, ...]:
        """Return one deterministic tracklet ID per detection in input order."""
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("Frame index must be non-negative.")
        eligible = [
            track
            for track in self._tracks.values()
            if frame_index - track.last_frame <= self.max_gap_frames + 1
        ]
        candidates = sorted(
            (
                (_intersection_over_union(track.box, detection.bounding_box), track.track_id, index)
                for track in eligible
                for index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        assigned_tracks: set[int] = set()
        assigned_detections: dict[int, int] = {}
        for overlap, track_id, detection_index in candidates:
            if overlap < self.minimum_iou:
                break
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections[detection_index] = track_id

        track_ids: list[int] = []
        for index, detection in enumerate(detections):
            track_id = assigned_detections.get(index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self.created_track_count += 1
            self._tracks[track_id] = _Tracklet(
                track_id=track_id,
                box=detection.bounding_box,
                last_frame=frame_index,
            )
            track_ids.append(track_id)
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= self.max_gap_frames + 1
        }
        return tuple(track_ids)


def infer_video(
    pipeline: DrowsinessPipeline,
    video_path: Path | str,
    output_video_path: Path | str,
    max_frames: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VideoAnalysisResult:
    """Write annotated frames in order and report per-frame counts/approximate tracklets."""
    source = Path(video_path).expanduser().resolve()
    output = Path(output_video_path).expanduser().resolve()
    frame_limit = max_frames if max_frames is not None else pipeline.config.max_video_frames
    if isinstance(frame_limit, bool) or not isinstance(frame_limit, int) or frame_limit < 1:
        raise ValueError("Maximum video frame count must be a positive integer.")
    supported, backends = supported_video_extensions()
    if source.suffix.lower() not in supported:
        raise DrowsinessMediaError(
            f"Unsupported video container '{source.suffix or 'no extension'}'. "
            f"Available backends ({', '.join(backends) or 'none'}) expose: "
            f"{', '.join(supported) or 'none'}."
        )
    if not source.is_file():
        raise DrowsinessMediaError(f"Video file was not found: '{source}'.")
    if source == output:
        raise DrowsinessVideoError("Output video path must not overwrite the input.")
    if output.exists():
        raise DrowsinessVideoError(f"Output video already exists: '{output}'.")
    if not output.parent.is_dir():
        raise DrowsinessVideoError(f"Output directory does not exist: '{output.parent}'.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise DrowsinessMediaError(
            f"OpenCV could not open '{source}'; verify the video and codec."
        )
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = (
        max(1, min(reported_frames, frame_limit))
        if reported_frames > 0
        else frame_limit
    )
    output_fps = source_fps or DEFAULT_OUTPUT_FPS
    tracker = GreedyIoUTracker(
        pipeline.config.tracker_iou_threshold,
        pipeline.config.tracker_max_gap_frames,
    )
    writer: cv2.VideoWriter | None = None
    frames: list[VideoFrameResult] = []
    output_created = False
    frame_size: tuple[int, int] | None = None
    frame_limit_reached = False
    low_light_frame_count = 0
    started = time.perf_counter()
    try:
        for frame_index in range(frame_limit):
            success, image_bgr = capture.read()
            if not success:
                if frame_index == 0:
                    raise DrowsinessMediaError(f"No decodable frames were found in '{source}'.")
                if reported_frames > 0 and frame_index < reported_frames:
                    raise DrowsinessMediaError(
                        f"Video decoding stopped at frame {frame_index} of "
                        f"{reported_frames} reported frames."
                    )
                break
            if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
                raise DrowsinessMediaError(f"Video frame {frame_index} is malformed.")
            height, width = image_bgr.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(
                    str(output),
                    _video_fourcc(output.suffix),
                    output_fps,
                    frame_size,
                )
                output_created = True
                if not writer.isOpened():
                    raise DrowsinessVideoError(
                        f"OpenCV could not create '{output}' with the selected codec."
                    )
            elif frame_size != (width, height):
                raise DrowsinessMediaError(
                    f"Video frame {frame_index} dimensions differ from the first frame."
                )

            detections = pipeline.detect_people(image_bgr)
            track_ids = tracker.update(detections, frame_index)
            people = tuple(
                PersonAssessment(detection=detection, track_id=track_ids[index])
                for index, detection in enumerate(detections)
            )
            if image_quality_warnings(image_bgr):
                low_light_frame_count += 1
            timestamp = frame_index / source_fps if source_fps is not None else None
            writer.write(annotate_people(image_bgr, people))
            frames.append(VideoFrameResult(frame_index, timestamp, people))
            if progress_callback is not None:
                progress_callback(frame_index + 1, progress_total)

        if len(frames) == frame_limit:
            frame_limit_reached = bool(capture.read()[0])
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        capture.release()
        if output_created:
            output.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.release()
        capture.release()

    if not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise DrowsinessVideoError(f"Annotated video was not written to '{output}'.")
    verification = cv2.VideoCapture(str(output))
    output_is_decodable = False
    try:
        output_is_decodable = verification.isOpened() and verification.read()[0]
    finally:
        verification.release()
    if not output_is_decodable:
        output.unlink(missing_ok=True)
        raise DrowsinessVideoError(f"Annotated output '{output}' cannot be decoded.")

    elapsed = time.perf_counter() - started
    video_duration = len(frames) / source_fps if source_fps is not None else None
    warnings = [
        "Drowsiness state and sleeping/drowsy counts are unavailable: no validated temporal model is configured."
    ]
    warnings.append(
        "Occlusion can split or merge person tracklets; person identities are not verified."
    )
    if low_light_frame_count:
        warnings.append(
            f"Very dark lighting was detected in {low_light_frame_count} processed frame(s)."
        )
    if source_fps is None:
        warnings.append("Source frame rate is unavailable; frame timestamps are unknown.")
    elif source_fps < MIN_USABLE_VIDEO_FPS:
        warnings.append(
            f"Source frame rate ({source_fps:.2f} FPS) is too low for reliable temporal analysis."
        )
    if (
        video_duration is not None
        and video_duration < pipeline.config.temporal_window_seconds
    ):
        warnings.append(
            "Clip is shorter than the configured temporal window; no window-level state is available."
        )

    metadata = dict(pipeline.metadata)
    metadata.update(
        {
            "count_policy": "per-frame person detections; tracklets are heuristic associations",
            "tracker": "greedy intersection-over-union",
            "temporal_inference": "unavailable; no validated temporal model is configured",
            "processing_fps": len(frames) / elapsed if elapsed > 0 else 0.0,
            "max_video_frames": frame_limit,
        }
    )
    return VideoAnalysisResult(
        output_video_path=output,
        frame_count=len(frames),
        source_fps=source_fps,
        output_fps=output_fps,
        frames=tuple(frames),
        unique_tracklet_count=tracker.created_track_count,
        temporal_window_seconds=pipeline.config.temporal_window_seconds,
        drowsiness_model_available=False,
        warnings=tuple(warnings),
        inference_metadata=metadata,
        frame_limit_reached=frame_limit_reached,
    )


def _intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _video_fourcc(extension: str) -> int:
    if extension.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    if extension.lower() in {".mp4", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    raise DrowsinessVideoError(f"Unsupported annotated video extension: '{extension}'.")
