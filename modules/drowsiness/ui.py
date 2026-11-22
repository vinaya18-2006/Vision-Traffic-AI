"""Streamlit interface for person detection and explicit model unavailability."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from modules.person_analysis.detector import PersonYoloDetector
from modules.person_analysis.errors import PersonAnalysisError
from modules.person_analysis.utils import supported_image_extensions, supported_video_extensions

from .config import DrowsinessConfig
from .contracts import ImageAnalysisResult, VideoAnalysisResult
from .detector import DrowsinessPipeline
from .preprocessing import decode_image
from .video import infer_video

LOGGER = logging.getLogger(__name__)
_RESULT_KEY = "drowsiness_result"
_MODE_KEY = "drowsiness_result_mode"
_VIDEO_BYTES_KEY = "drowsiness_video_bytes"
_SOURCE_VIDEO_BYTES_KEY = "drowsiness_source_video_bytes"


@st.cache_resource(max_entries=3)
def _cached_detector(model_path: str) -> PersonYoloDetector:
    return PersonYoloDetector(model_path)


def render_drowsiness() -> None:
    """Render safe person detection; drowsiness and age remain unavailable."""
    st.subheader("Driver Drowsiness Detection")
    st.caption(
        "Experimental computer-vision aid only. It is not a replacement for driver-safety "
        "systems, rest, or medical assessment."
    )
    st.error(
        "Drowsiness inference is unavailable: no validated temporal drowsiness model or "
        "licensed, representative evaluation video is configured. Person boxes are not "
        "sleepiness evidence; sleeping/drowsy counts are unavailable, not zero."
    )
    st.info(
        "Still images cannot establish sustained drowsiness. No age model is configured, "
        "so age estimates are omitted. Red boxes are reserved for model-backed temporal "
        "drowsy/sleeping detections and will not be emitted by this build."
    )

    mode = st.radio(
        "Input type",
        options=["Image", "Video"],
        horizontal=True,
        key="drowsiness_mode",
    )
    image_extensions = supported_image_extensions()
    video_extensions, backends = supported_video_extensions()
    accepted = image_extensions if mode == "Image" else video_extensions
    upload = st.file_uploader(
        f"Upload a vehicle-interior {mode.lower()}",
        type=[extension.lstrip(".") for extension in accepted],
        key="drowsiness_upload",
    )
    with st.sidebar:
        st.subheader("Drowsiness settings")
        confidence = st.slider(
            "Person detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="drowsiness_confidence",
        )
        window = st.slider(
            "Intended temporal window (seconds)",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
            key="drowsiness_window",
            help="Displayed for configuration only; no temporal classifier is available.",
        )
        max_frames = st.number_input(
            "Maximum video frames",
            min_value=1,
            max_value=100000,
            value=3000,
            step=250,
            key="drowsiness_max_frames",
        )
        st.caption(
            f"Video containers: {', '.join(video_extensions) or 'none detected'}; "
            f"backends: {', '.join(backends) or 'none'}."
        )

    if st.button(
        "Analyze media",
        type="primary",
        icon=":material/visibility:",
        width="stretch",
        key="drowsiness_submit",
    ):
        st.session_state.pop(_RESULT_KEY, None)
        st.session_state.pop(_VIDEO_BYTES_KEY, None)
        st.session_state.pop(_SOURCE_VIDEO_BYTES_KEY, None)
        if upload is None:
            st.error("Choose an image or video before starting analysis.")
        else:
            _run_analysis(mode, upload, confidence, window, int(max_frames))

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        return
    result_mode = st.session_state.get(_MODE_KEY, mode)
    if result_mode == "Image":
        _render_image_result(result)
    else:
        _render_video_result(
            result,
            st.session_state.get(_SOURCE_VIDEO_BYTES_KEY),
            st.session_state.get(_VIDEO_BYTES_KEY),
        )


def _run_analysis(mode: str, upload, confidence: float, window: float, max_frames: int) -> None:
    suffix = Path(upload.name).suffix.lower()
    extensions = (
        supported_image_extensions()
        if mode == "Image"
        else supported_video_extensions()[0]
    )
    if suffix not in extensions:
        st.error(f"Unsupported {mode.lower()} format '{suffix or 'no extension'}'.")
        return

    config = DrowsinessConfig(
        confidence_threshold=confidence,
        temporal_window_seconds=window,
        max_video_frames=max_frames,
    )
    try:
        detector = _cached_detector(str(config.person_model_path))
        pipeline = DrowsinessPipeline(config, detector)
        with st.spinner("Running local YOLO11n person detection..."):
            if mode == "Image":
                result = pipeline.infer_image(upload.getvalue())
                st.session_state[_RESULT_KEY] = result
            else:
                with tempfile.TemporaryDirectory(prefix="roadsight-drowsiness-") as directory:
                    source_path = Path(directory) / Path(upload.name).name
                    output_path = Path(directory) / f"annotated{suffix}"
                    source_bytes = upload.getvalue()
                    source_path.write_bytes(source_bytes)
                    progress = st.progress(0.0, text="Preparing video inference...")

                    def update_progress(completed: int, total: int) -> None:
                        progress.progress(
                            min(1.0, completed / max(total, 1)),
                            text=f"Processed frame {completed} of up to {total}.",
                        )

                    result = infer_video(
                        pipeline,
                        source_path,
                        output_path,
                        max_frames=max_frames,
                        progress_callback=update_progress,
                    )
                    progress.progress(1.0, text="Video person detection complete.")
                    st.session_state[_SOURCE_VIDEO_BYTES_KEY] = source_bytes
                    st.session_state[_VIDEO_BYTES_KEY] = output_path.read_bytes()
                    progress.empty()
                    st.session_state[_RESULT_KEY] = result
            st.session_state[_MODE_KEY] = mode
    except (PersonAnalysisError, ValueError, OSError) as exc:
        LOGGER.warning("Drowsiness page analysis failed: %s", exc)
        st.error(str(exc))


def _render_image_result(result: ImageAnalysisResult) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            result.source_image[:, :, ::-1],
            caption="Uploaded image",
            alt="Uploaded vehicle-interior image",
        )
    with col2:
        st.image(
            result.annotated_image[:, :, ::-1],
            caption="Annotated image",
            alt="Person detections",
        )
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Detected people", result.person_count)
    with col4:
        st.metric(
            "Sleeping/drowsy people",
            str(result.sleeping_drowsy_count)
            if result.sleeping_drowsy_count is not None
            else "Unavailable",
        )
        if result.quality_warnings:
            for warning in result.quality_warnings:
                st.warning(warning)
    if not result.people:
        st.info("No people were detected in this image.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Person": index,
                        "Detection score": person.detection.confidence,
                        "Drowsiness status": "Unknown (model unavailable)",
                        "Drowsiness score": None,
                        "Age estimate": "Unavailable",
                    }
                    for index, person in enumerate(result.people, start=1)
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _render_video_result(
    result: VideoAnalysisResult,
    source_video: bytes | None,
    annotated_video: bytes | None,
) -> None:
    st.metric("Processed frames", result.frame_count)
    st.metric("Heuristic person tracklets", result.unique_tracklet_count)
    st.metric("Sleeping/drowsy people", "Unavailable")
    st.caption(
        f"Configured temporal window: {result.temporal_window_seconds:.0f} seconds. "
        "No state is inferred or aggregated over that window."
    )
    for warning in result.warnings:
        st.warning(warning)
    video_columns = st.columns(2)
    if source_video:
        with video_columns[0]:
            st.caption("Uploaded video")
            st.video(source_video)
    if annotated_video:
        with video_columns[1]:
            st.caption("Annotated video")
            st.video(annotated_video)
    if not annotated_video:
        st.warning("The annotated video could not be retained for display.")
    rows = []
    for frame in result.frames:
        if not frame.people:
            rows.append(
                {
                    "Frame": frame.frame_index,
                    "Timestamp (s)": frame.timestamp_seconds,
                    "People in frame": frame.person_count,
                    "Person tracklet": "None",
                    "Person detection score": None,
                    "Drowsiness status": "Unavailable",
                    "Drowsiness score": None,
                    "Sleeping/drowsy count": "Unavailable",
                }
            )
        for person in frame.people:
            rows.append(
                {
                    "Frame": frame.frame_index,
                    "Timestamp (s)": frame.timestamp_seconds,
                    "People in frame": frame.person_count,
                    "Person tracklet": person.track_id,
                    "Person detection score": person.detection.confidence,
                    "Drowsiness status": "Unknown (model unavailable)",
                    "Drowsiness score": person.status_confidence,
                    "Sleeping/drowsy count": "Unavailable",
                }
            )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
    )
