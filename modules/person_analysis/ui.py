"""Streamlit presentation for person detection and optional model status."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Protocol

import pandas as pd
import streamlit as st

from .contracts import ImageAnalysisResult, VideoAnalysisResult
from .detector import PersonAnalysisPipeline, PersonYoloDetector, default_model_path
from .errors import (
    InferenceError,
    MediaInputError,
    ModelLoadError,
    PersonAnalysisError,
    VideoOutputError,
)
from .utils import decode_image, supported_image_extensions, supported_video_extensions
from .video import DEFAULT_MAX_VIDEO_FRAMES, infer_video

LOGGER = logging.getLogger(__name__)
_RESULT_KEY = "person_analysis_result"
_RESULT_MODE_KEY = "person_analysis_result_mode"
_SOURCE_BYTES_KEY = "person_analysis_source_bytes"
_SOURCE_NAME_KEY = "person_analysis_source_name"
_VIDEO_BYTES_KEY = "person_analysis_video_bytes"
_ERROR_KEY = "person_analysis_error"


class _UploadedMedia(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


@st.cache_resource(max_entries=4)
def _cached_detector(model_path: str) -> PersonYoloDetector:
    """Cache one loaded detector per explicitly configured local weight path."""
    return PersonYoloDetector(model_path)


def render_person_analysis() -> None:
    """Render the person-only analysis UI and explain unsupported optional stages."""
    st.subheader("Person Attribute & Emotion Analysis")
    st.caption(
        "Detects people and reports visible person boxes. Detection scores are model scores, "
        "not calibrated probabilities."
    )
    st.info(
        "Age estimation and facial-expression analysis are unavailable: this project does not "
        "include suitable validated model components for those features. No age, emotion, "
        "nationality, or ethnicity is inferred."
    )

    mode = st.radio(
        "Input type",
        options=["Image", "Video"],
        horizontal=True,
        key="person_analysis_mode",
    )
    image_extensions = supported_image_extensions()
    video_extensions, backends = supported_video_extensions()
    accepted_extensions = image_extensions if mode == "Image" else video_extensions
    upload = st.file_uploader(
        f"Upload a {mode.lower()}",
        type=[extension.lstrip(".") for extension in accepted_extensions],
        key="person_analysis_upload",
        help="Only formats exposed by the installed image/video decoders are accepted.",
    )

    with st.sidebar:
        st.subheader("Person analysis settings")
        confidence = st.slider(
            "Detection confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Detections below this model-score threshold are excluded.",
            key="person_analysis_confidence",
        )
        model_path_text = st.text_input(
            "Local YOLO11n weights",
            value=str(default_model_path()),
            help="Select an existing local yolo11n.pt file; weights are never downloaded automatically.",
            key="person_analysis_model_path",
        )
        max_frames = st.number_input(
            "Maximum video frames",
            min_value=1,
            max_value=100000,
            value=DEFAULT_MAX_VIDEO_FRAMES,
            step=250,
            key="person_analysis_max_frames",
        )
        st.caption(
            f"Image formats: {', '.join(image_extensions) or 'none detected'}. "
            f"Video containers: {', '.join(video_extensions) or 'none detected'}; "
            f"available backends: {', '.join(backends) or 'none'}."
        )

    analyze = st.button(
        "Analyze media",
        type="primary",
        icon=":material/person_search:",
        width="stretch",
        key="person_analysis_submit",
    )
    if analyze:
        progress_bar = st.progress(0.0, text="Preparing person analysis...")
        status = st.empty()
        _run_analysis(
            mode,
            upload,
            model_path_text,
            confidence,
            int(max_frames),
            progress_bar,
            status,
        )
        progress_bar.empty()
        status.empty()

    error = st.session_state.get(_ERROR_KEY)
    if error:
        st.error(error, icon=":material/error:")
    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        if upload is None:
            st.warning(
                "Choose an image or video to run person detection. Optional age and expression "
                "models are not installed."
            )
        return

    result_mode = st.session_state.get(_RESULT_MODE_KEY, mode)
    source_name = st.session_state.get(_SOURCE_NAME_KEY, "previous input")
    if result_mode != mode or (upload is not None and upload.name != source_name):
        st.caption(
            f"Showing the last completed {result_mode.lower()} analysis for **{source_name}**. "
            "Analyze the selected input to refresh these results."
        )
    if isinstance(result, ImageAnalysisResult):
        _render_image_result(result, st.session_state.get(_SOURCE_BYTES_KEY), source_name)
    else:
        _render_video_result(result, st.session_state.get(_VIDEO_BYTES_KEY))


def _run_analysis(
    mode: str,
    upload: _UploadedMedia | None,
    model_path_text: str,
    confidence_threshold: float,
    max_frames: int,
    progress_bar,
    status,
) -> None:
    for key in (
        _RESULT_KEY,
        _RESULT_MODE_KEY,
        _SOURCE_BYTES_KEY,
        _SOURCE_NAME_KEY,
        _VIDEO_BYTES_KEY,
        _ERROR_KEY,
    ):
        st.session_state.pop(key, None)
    if upload is None:
        st.session_state[_ERROR_KEY] = "Choose an image or video before starting analysis."
        return

    source_name = Path(upload.name).name
    source_bytes = upload.getvalue()
    suffix = Path(source_name).suffix.lower()
    supported = (
        supported_image_extensions() if mode == "Image" else supported_video_extensions()[0]
    )
    if suffix not in supported:
        st.session_state[_ERROR_KEY] = (
            f"Unsupported {mode.lower()} format '{suffix or 'no extension'}'. "
            f"Supported formats: {', '.join(supported) or 'none detected'}."
        )
        return

    try:
        detector = _cached_detector(str(Path(model_path_text).expanduser().resolve()))
        pipeline = PersonAnalysisPipeline(
            confidence_threshold=confidence_threshold,
            detector=detector,
        )
        if mode == "Image":
            result = pipeline.infer_image(source_bytes)
            video_bytes = None
            progress_bar.progress(1.0, text="Person analysis complete.")
        else:
            with tempfile.TemporaryDirectory(prefix="roadsight-person-") as directory:
                source_path = Path(directory) / source_name
                output_path = Path(directory) / f"annotated{suffix}"
                source_path.write_bytes(source_bytes)

                def update_progress(completed: int, total: int) -> None:
                    fraction = min(1.0, completed / max(total, 1))
                    progress_bar.progress(
                        fraction,
                        text=f"Processed frame {completed} of up to {total}.",
                    )
                    status.caption(
                        f"{completed} frames processed. Counts are per frame; "
                        "people are not tracked across frames."
                    )

                result = infer_video(
                    pipeline,
                    source_path,
                    output_path,
                    max_frames=max_frames,
                    progress_callback=update_progress,
                )
                video_bytes = output_path.read_bytes()
        st.session_state[_RESULT_KEY] = result
        st.session_state[_RESULT_MODE_KEY] = mode
        st.session_state[_SOURCE_BYTES_KEY] = source_bytes
        st.session_state[_SOURCE_NAME_KEY] = source_name
        if video_bytes is not None:
            st.session_state[_VIDEO_BYTES_KEY] = video_bytes
    except (ModelLoadError, MediaInputError, VideoOutputError, InferenceError) as exc:
        LOGGER.exception("Person analysis could not complete")
        st.session_state[_ERROR_KEY] = f"Person analysis unavailable: {exc}"
    except (OSError, ValueError) as exc:
        LOGGER.exception("Person analysis input or processing failed")
        st.session_state[_ERROR_KEY] = f"Could not process this input: {exc}"
    except PersonAnalysisError as exc:
        LOGGER.exception("Person analysis failed")
        st.session_state[_ERROR_KEY] = f"Person analysis failed: {exc}"
    except Exception:
        LOGGER.exception("Unexpected person-analysis dashboard error")
        st.session_state[_ERROR_KEY] = (
            "Person analysis could not be completed. Check the input and local model "
            "configuration, then try again."
        )


def _render_image_result(
    result: ImageAnalysisResult,
    source_bytes: bytes | None,
    source_name: str,
) -> None:
    st.metric("People detected in this image", result.person_count)
    source_column, annotated_column = st.columns(2)
    with source_column:
        if source_bytes:
            original = decode_image(source_bytes)
            st.image(original, caption=f"Original: {source_name}", channels="BGR", alt="Original uploaded image")
    with annotated_column:
        st.image(
            result.annotated_image,
            caption="Annotated person detections",
            channels="BGR",
            alt="Uploaded image with person bounding boxes",
        )
    st.subheader("Person detections")
    st.dataframe(
        _detection_rows(result.detections),
        hide_index=True,
        width="stretch",
    )
    st.caption("Confidence values are detector scores, not calibrated probabilities.")


def _render_video_result(
    result: VideoAnalysisResult,
    video_bytes: bytes | None,
) -> None:
    st.metric("Processed video frames", result.frame_count)
    st.caption(
        "Counts below are per frame. Repeated detections are not summed as unique people; "
        "person tracking is not implemented."
    )
    if result.frame_limit_reached:
        st.warning(
            f"Stopped at the configured limit of {result.max_frames} frames; additional frames "
            "were present in the source."
        )
    source_column, annotated_column = st.columns(2)
    with source_column:
        st.image(
            result.original_preview_frame,
            caption="Original first frame",
            channels="BGR",
            alt="Original first frame of the uploaded video",
        )
    with annotated_column:
        st.image(
            result.annotated_preview_frame,
            caption="Annotated first frame",
            channels="BGR",
            alt="First video frame with person bounding boxes",
        )
    if video_bytes:
        st.video(video_bytes, format=f"video/{'x-msvideo' if result.output_video_path.suffix.lower() == '.avi' else 'mp4'}", end_time=None)
        st.download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=f"annotated{result.output_video_path.suffix.lower()}",
            mime="video/x-msvideo" if result.output_video_path.suffix.lower() == ".avi" else "video/mp4",
        )
    st.subheader("Person detections by frame")
    st.dataframe(
        _video_detection_rows(result.frames),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Processed at {result.processing_fps:.2f} frames/second. "
        "Video inference is offline and does not imply real-time performance."
    )


def _detection_rows(detections) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "person": index + 1,
                "confidence score": round(detection.confidence, 4),
                "x1": round(detection.bounding_box.x1, 2),
                "y1": round(detection.bounding_box.y1, 2),
                "x2": round(detection.bounding_box.x2, 2),
                "y2": round(detection.bounding_box.y2, 2),
            }
            for index, detection in enumerate(detections)
        ],
        columns=["person", "confidence score", "x1", "y1", "x2", "y2"],
    )


def _video_detection_rows(frames) -> pd.DataFrame:
    rows = []
    for frame in frames:
        for person_index, detection in enumerate(frame.detections, start=1):
            rows.append(
                {
                    "frame": frame.frame_index,
                    "person in frame": person_index,
                    "people in frame": frame.person_count,
                    "confidence score": round(detection.confidence, 4),
                    "x1": round(detection.bounding_box.x1, 2),
                    "y1": round(detection.bounding_box.y1, 2),
                    "x2": round(detection.bounding_box.x2, 2),
                    "y2": round(detection.bounding_box.y2, 2),
                }
            )
        if not frame.detections:
            rows.append(
                {
                    "frame": frame.frame_index,
                    "person in frame": None,
                    "people in frame": 0,
                    "confidence score": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "frame",
            "person in frame",
            "people in frame",
            "confidence score",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )

