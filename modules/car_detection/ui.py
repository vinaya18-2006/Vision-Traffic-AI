"""Professional Streamlit dashboard for Task 1 vehicle analytics."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Protocol, TypeAlias

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .contracts import (
    CarDetection,
    ColourLabel,
    ImageInferenceResult,
    PersonDetection,
    VideoFrameResult,
    VideoInferenceResult,
)
from .detector import CarDetectionPipeline, YoloDetector, default_model_path
from .errors import CarDetectionError, MediaInputError, ModelLoadError
from .utils import (
    KNOWN_COLOUR_LABELS,
    decode_image,
    supported_image_extensions,
    supported_video_extensions,
)
from .video import DEFAULT_MAX_VIDEO_FRAMES, infer_video

LOGGER = logging.getLogger(__name__)
COLOUR_LABELS: tuple[ColourLabel, ...] = (
    "blue",
    "red",
    "white",
    "black",
    "gray",
    "silver",
    "green",
    "yellow",
    "orange",
    "brown",
    "other",
    "unknown",
)
TaskResult: TypeAlias = ImageInferenceResult | VideoInferenceResult


class _UploadedMedia(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


_RESET_KEY = "task1_reset_generation"
_RESULT_KEY = "task1_dashboard_result"
_MODE_KEY = "task1_result_mode"
_SOURCE_KEY = "task1_source_bytes"
_SOURCE_NAME_KEY = "task1_source_name"
_VIDEO_KEY = "task1_video_bytes"
_ERROR_KEY = "task1_dashboard_error"


@st.cache_resource(max_entries=4)
def _cached_detector(model_path: str) -> YoloDetector:
    """Load and retain one local YOLO11n instance per configured weights path."""
    return YoloDetector(model_path)


def _reset_dashboard() -> None:
    """Clear Task 1 results and rotate widget keys before the next script run."""
    for key in (_RESULT_KEY, _MODE_KEY, _SOURCE_KEY, _SOURCE_NAME_KEY, _VIDEO_KEY, _ERROR_KEY):
        st.session_state.pop(key, None)
    st.session_state[_RESET_KEY] = st.session_state.get(_RESET_KEY, 0) + 1


def _run_analysis(
    mode: str,
    uploaded_file: _UploadedMedia | None,
    model_path_text: str,
    confidence_threshold: float,
    colour_confidence_threshold: float,
    max_video_frames: int,
) -> None:
    """Run inference only on explicit form submission and keep its result in session state."""
    for key in (_RESULT_KEY, _MODE_KEY, _SOURCE_KEY, _SOURCE_NAME_KEY, _VIDEO_KEY, _ERROR_KEY):
        st.session_state.pop(key, None)
    if uploaded_file is None:
        st.session_state[_ERROR_KEY] = "Choose an image or video before starting analysis."
        return

    source_name = Path(uploaded_file.name).name
    source_bytes = uploaded_file.getvalue()
    suffix = Path(source_name).suffix.lower()
    accepted_extensions = (
        supported_image_extensions() if mode == "Image" else supported_video_extensions()[0]
    )
    if suffix not in accepted_extensions:
        st.session_state[_ERROR_KEY] = (
            f"Unsupported {mode.lower()} format '{suffix or 'no extension'}'. "
            f"Supported formats: {', '.join(accepted_extensions) or 'none detected'}."
        )
        return

    try:
        detector = _cached_detector(str(Path(model_path_text).expanduser().resolve()))
        pipeline = CarDetectionPipeline(
            confidence_threshold=confidence_threshold,
            colour_confidence_threshold=colour_confidence_threshold,
            detector=detector,
        )
        if mode == "Image":
            result: TaskResult = pipeline.infer_image(source_bytes)
            video_bytes = None
        else:
            with tempfile.TemporaryDirectory(prefix="visiontraffic-") as directory:
                source_path = Path(directory) / source_name
                output_path = Path(directory) / f"annotated{suffix}"
                source_path.write_bytes(source_bytes)
                result = infer_video(
                    pipeline,
                    source_path,
                    output_path,
                    max_frames=max_video_frames,
                )
                video_bytes = output_path.read_bytes()
        st.session_state[_RESULT_KEY] = result
        st.session_state[_MODE_KEY] = mode
        st.session_state[_SOURCE_KEY] = source_bytes
        st.session_state[_SOURCE_NAME_KEY] = source_name
        if video_bytes is not None:
            st.session_state[_VIDEO_KEY] = video_bytes
    except ModelLoadError as exc:
        LOGGER.exception("Task 1 model loading failed")
        st.session_state[_ERROR_KEY] = f"Model unavailable: {exc}"
    except MediaInputError as exc:
        LOGGER.exception("Task 1 media could not be processed")
        st.session_state[_ERROR_KEY] = f"Input processing failed: {exc}"
    except CarDetectionError as exc:
        LOGGER.exception("Task 1 inference failed")
        st.session_state[_ERROR_KEY] = f"Analysis failed: {exc}"
    except (OSError, ValueError, cv2.error) as exc:
        LOGGER.exception("Task 1 encountered an input or processing error")
        st.session_state[_ERROR_KEY] = f"Could not process this input: {exc}"
    except Exception:
        LOGGER.exception("Unexpected Task 1 dashboard error")
        st.session_state[_ERROR_KEY] = (
            "Analysis could not be completed. Check the input and model configuration, "
            "then try again."
        )


def render_task1() -> None:
    """Render the interactive VisionTraffic AI image/video analytics dashboard."""
    _render_dashboard_styles()
    _render_product_header()
    mode, uploaded_file, model_path, confidence, colour_threshold, max_frames, submitted = (
        _render_sidebar_controls()
    )
    if submitted:
        with st.spinner("Running YOLO11n inference and preparing analytics..."):
            _run_analysis(
                mode,
                uploaded_file,
                model_path,
                confidence,
                colour_threshold,
                max_frames,
            )

    error = st.session_state.get(_ERROR_KEY)
    if error:
        st.error(error, icon=":material/error:")
    result: TaskResult | None = st.session_state.get(_RESULT_KEY)
    if result is None:
        _render_empty_state(mode, uploaded_file)
        return
    result_mode = st.session_state.get(
        _MODE_KEY,
        "Image" if isinstance(result, ImageInferenceResult) else "Video",
    )
    result_source = st.session_state.get(_SOURCE_NAME_KEY, "previous input")
    input_changed = mode != result_mode or (
        uploaded_file is not None and uploaded_file.name != result_source
    )
    if input_changed:
        st.caption(
            f"Showing the last completed {result_mode.lower()} analysis for **{result_source}**. "
            "Submit analysis to update the dashboard."
        )

    _render_kpis(result)
    overview_tab, details_tab, analytics_tab, technical_tab = st.tabs(
        ["Overview", "Detection Details", "Analytics", "Technical Info"]
    )
    with overview_tab:
        _render_detection_view(
            result,
            st.session_state.get(_SOURCE_KEY),
            st.session_state.get(_SOURCE_NAME_KEY, "source"),
            st.session_state.get(_VIDEO_KEY),
        )
    with details_tab:
        _render_detection_details(result)
    with analytics_tab:
        _render_analytics(result)
    with technical_tab:
        _render_processing_information(result)


def _render_product_header() -> None:
    model_status = "SYSTEM ONLINE"
    status_dot = "vt-status-dot"
    if not default_model_path().is_file():
        model_status = "WEIGHTS REQUIRED"
        status_dot = "vt-status-dot"
    st.markdown(
        """
        <div class="vt-header">
          <div>
            <div class="vt-eyebrow">VEHICLE &amp; TRAFFIC VISION ANALYTICS</div>
            <h1>VisionTraffic AI</h1>
            <p>Intelligent Vehicle &amp; Traffic Analytics</p>
          </div>
          <div class="vt-status-group">
            <span class="vt-status"><span class="STATUS_DOT"></span>MODEL_STATUS</span>
            <span class="vt-device">Model&nbsp; <strong>YOLO11n</strong></span>
            <span class="vt-device">Device&nbsp; <strong>CPU</strong></span>
          </div>
        </div>
        <div class="vt-task-label">Task 1&nbsp; &bull;&nbsp; Car Colour Detection</div>
        """.replace("STATUS_DOT", status_dot).replace("MODEL_STATUS", model_status),
        unsafe_allow_html=True,
    )


def _render_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --vt-bg: #07121a;
          --vt-surface: rgba(16, 27, 38, 0.9);
          --vt-surface-strong: rgba(18, 33, 47, 0.95);
          --vt-surface-soft: rgba(13, 25, 36, 0.72);
          --vt-border: rgba(148, 163, 184, 0.22);
          --vt-primary: #5dd7ff;
          --vt-primary-strong: #7ef0ff;
          --vt-accent: #7de2a1;
          --vt-warning: #f5b642;
          --vt-danger: #ff6f7d;
          --vt-text: #edf6ff;
          --vt-muted: rgba(237, 246, 255, 0.72);
          --vt-shadow: 0 16px 32px rgba(7, 18, 26, 0.38);
        }
        .block-container {
          padding-top: 1.35rem !important;
          padding-bottom: 2.25rem !important;
          max-width: 1440px !important;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(8, 17, 25, 0.94), rgba(11, 21, 31, 0.98));
          border-right: 1px solid var(--vt-border);
        }
        .vt-header {
          display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
          padding:1.25rem 1.5rem; border:1px solid var(--vt-border);
          border-radius:22px; background:linear-gradient(135deg,
            rgba(13, 37, 53, 0.96), rgba(11, 24, 34, 0.86));
          box-shadow: var(--vt-shadow);
          margin-bottom:.85rem;
          position:relative; overflow:hidden;
        }
        .vt-header::before {
          content:""; position:absolute; inset:-25% auto auto -12%; width:240px; height:240px;
          background: radial-gradient(circle, rgba(93, 215, 255, 0.34), transparent 60%);
          pointer-events:none;
        }
        .vt-eyebrow {
          font-size:.72rem; font-weight:700; letter-spacing:.16em; color:var(--vt-primary);
          margin-bottom:.36rem; text-transform:uppercase;
        }
        .vt-header h1 {
          font-size:2.15rem; line-height:1.06; margin:0; color:var(--vt-text);
          font-weight:700; letter-spacing:-0.03em;
        }
        .vt-header p {margin:.3rem 0 0; color:var(--vt-muted); font-size:1rem;}
        .vt-status-group {display:flex; flex-direction:column; align-items:flex-end; gap:.6rem; z-index:1;}
        .vt-status {
          display:inline-flex; align-items:center; gap:.45rem; border-radius:999px;
          padding:.42rem .8rem; border:1px solid rgba(108, 255, 180, 0.35);
          color:var(--vt-text); background:rgba(17, 63, 48, 0.42);
          font-size:.84rem; font-weight:650;
        }
        .vt-status-dot {
          width:.58rem; height:.58rem; border-radius:50%; background:#46d39a; box-shadow:0 0 0 0 rgba(70,211,154,.45);
          animation: vt-pulse 2.2s infinite ease-out;
        }
        @keyframes vt-pulse {
          0% { box-shadow: 0 0 0 0 rgba(70,211,154,.45); }
          70% { box-shadow: 0 0 0 10px rgba(70,211,154,0); }
          100% { box-shadow: 0 0 0 0 rgba(70,211,154,0); }
        }
        .vt-device {font-size:.84rem; color:var(--vt-muted);} .vt-device strong {color:var(--vt-text);}
        .vt-task-label {
          display:inline-block; margin:.1rem 0 1.05rem; padding:.4rem .7rem; border-radius:999px;
          background:rgba(15, 31, 41, 0.88); border:1px solid var(--vt-border); font-size:.82rem; font-weight:700;
          color:var(--vt-text); letter-spacing:.02em;
        }
        .vt-section-caption {color:var(--vt-muted); font-size:.9rem;} .vt-banner {
          padding:.72rem .9rem; border-radius:12px; border:1px solid rgba(93, 215, 255, 0.2);
          background:rgba(18, 51, 67, 0.55); font-size:.9rem; margin:.45rem 0 .85rem; color:var(--vt-text);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .vt-empty {
          padding:2rem 1.25rem; border:1px dashed rgba(148,163,184,0.35); border-radius:20px; text-align:center;
          background:linear-gradient(180deg, rgba(15, 27, 35, 0.72), rgba(12, 22, 30, 0.92));
          box-shadow: var(--vt-shadow);
        }
        .vt-empty h3 {margin:.15rem 0 .5rem; font-size:1.35rem; color:var(--vt-text);}
        .vt-empty p {margin:0; opacity:.72; color:var(--vt-muted);}
        .vt-kpi-card {
          background:linear-gradient(180deg, rgba(16, 28, 38, 0.9), rgba(12, 23, 32, 0.92));
          border:1px solid var(--vt-border); border-radius:16px; padding:1rem .9rem 0.9rem; min-height:120px;
          box-shadow: var(--vt-shadow); transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
          animation: vt-fade-in .45s ease both;
        }
        .vt-kpi-card:hover {
          transform: translateY(-2px); border-color: rgba(93, 215, 255, 0.44);
          box-shadow: 0 18px 32px rgba(7, 18, 26, 0.48);
        }
        .vt-kpi-label {
          font-size:.68rem; line-height:1.2; letter-spacing:.12em; text-transform:uppercase; color:var(--vt-muted);
          margin-bottom:.55rem; font-weight:700;
        }
        .vt-kpi-value {
          font-size:1.8rem; font-weight:700; line-height:1.1; color:var(--vt-text); letter-spacing:-0.04em;
          margin-bottom:.2rem;
        }
        .vt-kpi-context {
          font-size:.78rem; color:var(--vt-muted); line-height:1.35;
        }
        .vt-panel {
          background:linear-gradient(180deg, rgba(16, 28, 38, 0.9), rgba(12, 22, 31, 0.95));
          border:1px solid var(--vt-border); border-radius:18px; padding:1rem; box-shadow: var(--vt-shadow);
          margin-bottom:1rem; animation: vt-fade-in .5s ease both;
        }
        .vt-panel h3 {margin:0 0 .8rem; color:var(--vt-text); font-size:1.05rem; font-weight:700;}
        .vt-legend-item {display:inline-flex; align-items:center; gap:.55rem; margin-right:1rem; color:var(--vt-muted); font-size:.82rem;}
        .vt-swatch {width:.72rem; height:.72rem; border-radius:6px; display:inline-block;}
        .vt-ghost-button {
          display:inline-flex; align-items:center; justify-content:center; border-radius:12px; padding:.62rem 1rem;
          background:linear-gradient(135deg, rgba(17, 87, 113, 0.85), rgba(14, 62, 78, 0.9));
          border:1px solid rgba(93, 215, 255, 0.35); color:var(--vt-text); font-weight:600;
          transition: transform .18s ease, box-shadow .18s ease;
        }
        .vt-ghost-button:hover {transform: translateY(-1px); box-shadow:0 10px 18px rgba(18,70,90,.35);}
        .vt-metric-strip {display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:.9rem; margin:0 0 1rem;}
        .vt-stats-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:.8rem; margin-top: .75rem;}
        .vt-stat-box {padding:.8rem .9rem; border-radius:12px; background:rgba(10,22,33,.75); border:1px solid var(--vt-border);}
        .vt-stat-box .label {font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; color:var(--vt-muted);}
        .vt-stat-box .value {font-size:1.15rem; font-weight:700; margin-top:.25rem; color:var(--vt-text);}
        .vt-empty-graphic {
          width:110px; height:110px; margin:0 auto 1rem; display:block; opacity:.8;
          filter: drop-shadow(0 12px 18px rgba(93, 215, 255, 0.18));
        }
        @keyframes vt-fade-in { from { opacity:0; transform: translateY(8px);} to { opacity:1; transform:none;} }
        @media (max-width: 720px) {
          .vt-header {align-items:flex-start; flex-direction:column; padding:1rem;}
          .vt-status-group {align-items:flex-start; width:100%; flex-direction:row; flex-wrap:wrap;}
          .vt-header h1 {font-size:1.8rem;}
        }
        @media (prefers-reduced-motion: reduce) {
          .vt-status-dot, .vt-kpi-card, .vt-panel, .vt-ghost-button {animation:none !important; transition:none !important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_controls() -> tuple[str, _UploadedMedia | None, str, float, float, int, bool]:
    with st.sidebar:
        st.subheader("Analysis controls")
        generation = st.session_state.setdefault(_RESET_KEY, 0)
        mode = st.segmented_control(
            "Input mode",
            options=["Image", "Video"],
            default="Image",
            key=f"task1_mode_{generation}",
            help="Choose the media type to analyze.",
            width="stretch",
        )
        selected_mode = mode or "Image"
        image_extensions = supported_image_extensions()
        video_extensions, backends = supported_video_extensions()
        accepted_extensions = image_extensions if selected_mode == "Image" else video_extensions
        uploaded_file = st.file_uploader(
            f"{selected_mode} file",
            type=[extension.lstrip(".") for extension in accepted_extensions],
            key=f"task1_upload_{generation}_{selected_mode.lower()}",
            help="Files are processed locally by the configured model.",
        )
        with st.form(f"task1_analysis_form_{generation}"):
            confidence = st.slider(
                "Detection confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Detections below this threshold are excluded.",
            )
            with st.expander("Advanced detection settings"):
                colour_threshold = st.slider(
                    "Minimum colour evidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    help="Lower evidence estimates are returned as unknown.",
                )
                max_frames = st.number_input(
                    "Maximum video frames",
                    min_value=1,
                    value=DEFAULT_MAX_VIDEO_FRAMES,
                    step=250,
                )
                model_path = st.text_input(
                    "YOLO11n weights path",
                    value=str(default_model_path()),
                    help="Select an existing local yolo11n.pt file. No download is attempted.",
                )
            submitted = st.form_submit_button(
                "Analyze media",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )
        st.button(
            "Reset analysis",
            icon=":material/refresh:",
            on_click=_reset_dashboard,
            width="stretch",
        )
        st.divider()
        st.markdown("**Model**  \nYOLO11n · COCO classes: person, car")
        st.markdown(
            "**Colour estimate**  \n"
            "Estimated from the visible crop; lighting, reflections, and occlusion can affect it."
        )
        st.caption(
            f"Image decoders: {', '.join(image_extensions) or 'none'}. "
            f"Video containers from {', '.join(backends) or 'no OpenCV backends'}: "
            f"{', '.join(video_extensions) or 'none'}."
        )
    return (
        selected_mode,
        uploaded_file,
        model_path,
        confidence,
        colour_threshold,
        int(max_frames),
        submitted,
    )


def _render_empty_state(mode: str, uploaded_file: _UploadedMedia | None) -> None:
    if st.session_state.get(_ERROR_KEY):
        return
    st.markdown(
        """
        <div class="vt-empty">
          <svg class="vt-empty-graphic" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <defs>
              <linearGradient id="panelGlow" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stop-color="#5dd7ff" stop-opacity="0.85"/>
                <stop offset="100%" stop-color="#7de2a1" stop-opacity="0.85"/>
              </linearGradient>
            </defs>
            <rect x="18" y="18" width="124" height="124" rx="26" fill="rgba(93, 215, 255, 0.08)" stroke="rgba(93, 215, 255, 0.4)"/>
            <path d="M42 90 L69 58 L83 74 L112 46 L118 90 Z" fill="url(#panelGlow)" opacity="0.85"/>
            <circle cx="52" cy="62" r="8" fill="#5dd7ff" opacity="0.82"/>
            <rect x="36" y="98" width="88" height="14" rx="7" fill="rgba(125, 226, 161, 0.2)"/>
            <path d="M82 27 L86 43 L102 43 L89 54 L94 72 L82 61 L70 72 L75 54 L62 43 L78 43 Z" fill="#f2f8ff" opacity="0.72"/>
          </svg>
          <h3>VisionTraffic AI</h3>
          <p>Upload traffic media to begin intelligent detection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if uploaded_file is None:
        return
    if mode == "Image":
        try:
            source_image = decode_image(uploaded_file.getvalue())
        except CarDetectionError as exc:
            st.error(f"Unsupported or unreadable image: {exc}", icon=":material/broken_image:")
        else:
            st.subheader("Source preview")
            st.image(
                source_image,
                channels="BGR",
                alt="Uploaded source image awaiting analysis",
                width="stretch",
            )
            st.info("Select **Analyze media** in the control panel to run real YOLO11n inference.")
    else:
        st.info(
            "Video selected. Run analysis to decode frames, preview the original and annotated "
            "result, and inspect frame-level analytics."
        )


def _cars_and_people(
    result: TaskResult,
) -> tuple[tuple[CarDetection, ...], tuple[PersonDetection, ...]]:
    if isinstance(result, ImageInferenceResult):
        return result.cars, result.people
    return (
        tuple(car for frame in result.frames for car in frame.cars),
        tuple(person for frame in result.frames for person in frame.people),
    )


def _render_kpis(result: TaskResult) -> None:
    cars, people = _cars_and_people(result)
    blue_count = sum(car.estimated_colour.label == "blue" for car in cars)
    other_known_count = sum(
        car.estimated_colour.label in KNOWN_COLOUR_LABELS - {"blue"} for car in cars
    )
    confidences = [detection.confidence for detection in (*cars, *people)]
    mean_confidence = float(np.mean(confidences)) if confidences else None
    still_image = isinstance(result, ImageInferenceResult)
    metrics = [
        ("Total cars", str(len(cars)), "Retained detections"),
        ("Blue cars", str(blue_count), "Blue body colour"),
        ("Other colour cars", str(other_known_count), "Known non-blue"),
        ("People", str(len(people)), "Retained detections"),
        ("Avg confidence", f"{mean_confidence:.1%}" if mean_confidence is not None else "—", "Detector score"),
        ("Processing FPS", f"{result.processing_fps:.2f}" if isinstance(result, VideoInferenceResult) else "—", "Measured throughput"),
    ]
    columns = st.columns(6)
    for column, (label, value, context) in zip(columns, metrics):
        with column:
            st.markdown(
                f'<div class="vt-kpi-card">'
                f'<div class="vt-kpi-label">{label}</div>'
                f'<div class="vt-kpi-value">{value}</div>'
                f'<div class="vt-kpi-context">{context}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    if isinstance(result, VideoInferenceResult):
        st.markdown(
            '<div class="vt-banner"><strong>Video totals are frame-level detections.</strong> '
            "Unique vehicle/person counts require object tracking.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Image counts represent retained detections in this still image; confidence values are "
            "detector scores, not calibrated probabilities."
        )


def _render_detection_view(
    result: TaskResult,
    source_bytes: bytes | None,
    source_name: str,
    video_bytes: bytes | None,
) -> None:
    st.markdown('<div class="vt-panel"><h3>Detection preview</h3></div>', unsafe_allow_html=True)
    if isinstance(result, ImageInferenceResult):
        if source_bytes is None:
            st.warning("The source image is no longer available in this session.")
            return
        original, annotated = st.columns(2)
        with original:
            st.markdown("**Original image**")
            st.image(
                decode_image(source_bytes),
                channels="BGR",
                alt=f"Original uploaded image: {source_name}",
                width="stretch",
            )
        with annotated:
            st.markdown("**AI detection**")
            st.image(
                result.annotated_image,
                channels="BGR",
                alt="Image annotated with car colours and person detections",
                width="stretch",
            )
        st.markdown(
            '<div class="vt-panel">'
            '<div class="vt-legend-item"><span class="vt-swatch" style="background:#ff3b30"></span>Blue car</div>'
            '<div class="vt-legend-item"><span class="vt-swatch" style="background:#1d9bf0"></span>Other known colour</div>'
            '<div class="vt-legend-item"><span class="vt-swatch" style="background:#8ea0ad"></span>Unknown colour</div>'
            '<div class="vt-legend-item"><span class="vt-swatch" style="background:#6ee7b7"></span>Person</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        encoded, image_buffer = cv2.imencode(".png", result.annotated_image)
        if not encoded:
            st.error("The annotated image could not be encoded for download.")
        else:
            st.download_button(
                "Download annotated image",
                image_buffer.tobytes(),
                file_name="visiontraffic_annotated.png",
                mime="image/png",
                icon=":material/download:",
                key="task1_dashboard_image_download",
            )
        if not result.cars and not result.people:
            st.info(
                "Inference completed successfully, but no cars or people passed the selected "
                "confidence threshold."
            )
        return

    if video_bytes is not None:
        st.markdown("**Annotated video**")
        st.video(
            video_bytes,
            format=_video_mime_type(Path(source_name).suffix),
            alt="YOLO11n-annotated video",
        )
        st.markdown(
            '<div class="vt-panel">'
            '<div class="vt-stats-grid">'
            f'<div class="vt-stat-box"><div class="label">Frames Processed</div><div class="value">{result.frame_count:,}</div></div>'
            f'<div class="vt-stat-box"><div class="label">Source FPS</div><div class="value">{result.source_fps if result.source_fps is not None else "Unknown"}</div></div>'
            f'<div class="vt-stat-box"><div class="label">Output FPS</div><div class="value">{result.output_fps:.2f}</div></div>'
            f'<div class="vt-stat-box"><div class="label">Processing FPS</div><div class="value">{result.processing_fps:.2f}</div></div>'
            f'<div class="vt-stat-box"><div class="label">Peak Cars / Frame</div><div class="value">{max(frame.car_count for frame in result.frames) if result.frames else 0}</div></div>'
            f'<div class="vt-stat-box"><div class="label">Peak People / Frame</div><div class="value">{max(frame.person_count for frame in result.frames) if result.frames else 0}</div></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download annotated video",
            video_bytes,
            file_name=f"visiontraffic_annotated{Path(source_name).suffix.lower()}",
            mime=_video_mime_type(Path(source_name).suffix),
            icon=":material/download:",
            key="task1_dashboard_video_download",
        )
    else:
        st.warning("The annotated video file is unavailable; run the analysis again to recreate it.")
    if result.frame_limit_reached:
        st.warning(
            f"Processing stopped at {result.max_frames:,} frames. Later frames were not included."
        )


def _render_detection_details(result: TaskResult) -> None:
    st.subheader("Detection details")
    cars, people = _cars_and_people(result)
    table = _detection_table(cars, people, result.frames if isinstance(result, VideoInferenceResult) else None)
    if table.empty:
        st.info("No detections are available for this result.")
        return
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "Bounding box": st.column_config.TextColumn("Bounding box"),
        },
        alt="Vehicle and person detections with estimated colours and confidence",
        width="stretch",
    )
    with st.expander("Advanced detection details"):
        st.caption(
            "Confidence is the detector score. Colour evidence is the HSV crop pixel-share score, "
            "not a calibrated probability."
        )
        advanced_rows = []
        for detection in (*cars, *people):
            if isinstance(detection, CarDetection):
                advanced_rows.append(
                    {
                        "Class": detection.class_name,
                        "Colour evidence": f"{detection.estimated_colour.confidence:.1%}",
                        "Colour note": detection.estimated_colour.reason or "Evidence available",
                    }
                )
        if advanced_rows:
            st.dataframe(
                pd.DataFrame(advanced_rows),
                hide_index=True,
                alt="Advanced car crop-colour evidence",
            )
        else:
            st.caption("No car crop-colour estimates were produced.")


def _detection_table(
    cars: tuple[CarDetection, ...],
    people: tuple[PersonDetection, ...],
    frames: tuple[VideoFrameResult, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frames is not None:
        for frame in frames:
            for detection in frame.cars:
                rows.append(_car_row(detection, len(rows) + 1, frame.frame_index))
            for detection in frame.people:
                rows.append(_person_row(detection, len(rows) + 1, frame.frame_index))
    else:
        for detection in cars:
            rows.append(_car_row(detection, len(rows) + 1))
        for detection in people:
            rows.append(_person_row(detection, len(rows) + 1))
    columns = ["ID", "Class", "Colour", "Confidence", "Bounding box", "Detection status"]
    if frames is not None:
        columns.insert(1, "Frame")
    return pd.DataFrame(rows, columns=columns)


def _car_row(car: CarDetection, number: int, frame_index: int | None = None) -> dict[str, object]:
    label = car.estimated_colour.label
    status = "BLUE VEHICLE" if label == "blue" else (
        "UNKNOWN COLOUR" if label == "unknown" else "OTHER VEHICLE"
    )
    return {
        "ID": f"CAR-{number:03d}",
        "Frame": frame_index,
        "Class": car.class_name.title(),
        "Colour": label.title(),
        "Confidence": car.confidence,
        "Bounding box": _format_box(car),
        "Detection status": status,
    }


def _person_row(
    person: PersonDetection,
    number: int,
    frame_index: int | None = None,
) -> dict[str, object]:
    return {
        "ID": f"PERSON-{number:03d}",
        "Frame": frame_index,
        "Class": person.class_name.title(),
        "Colour": "—",
        "Confidence": person.confidence,
        "Bounding box": _format_box(person),
        "Detection status": "PERSON",
    }


def _format_box(detection: CarDetection | PersonDetection) -> str:
    return ", ".join(f"{coordinate:.0f}" for coordinate in detection.bounding_box.as_xyxy())


def _render_analytics(result: TaskResult) -> None:
    st.subheader("Vehicle colour analytics")
    cars, people = _cars_and_people(result)
    counts = {label: 0 for label in COLOUR_LABELS}
    for car in cars:
        counts[car.estimated_colour.label] += 1
    total_cars = len(cars)
    distribution = pd.DataFrame(
        [
            {
                "Colour": label.title(),
                "Count": counts[label],
                "Percentage": counts[label] / total_cars if total_cars else 0.0,
            }
            for label in COLOUR_LABELS
        ]
    )
    nonzero = distribution[distribution["Count"] > 0]
    if nonzero.empty:
        st.info("No vehicle colours were observed in this result.")
    else:
        st.bar_chart(
            nonzero,
            x="Colour",
            y="Count",
            alt="Observed car detections by estimated apparent colour",
        )
    st.dataframe(
        distribution,
        hide_index=True,
        column_config={
            "Percentage": st.column_config.ProgressColumn(
                "Share of car detections",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
            )
        },
        alt="Counts and percentage share for every supported car colour",
    )

    if isinstance(result, ImageInferenceResult):
        _render_image_analytics(result, distribution, people)
    else:
        _render_video_analytics(result)


def _render_image_analytics(
    result: ImageInferenceResult,
    distribution: pd.DataFrame,
    people: tuple[PersonDetection, ...],
) -> None:
    chart_column, confidence_column = st.columns(2)
    with chart_column:
        with st.container(border=True):
            st.markdown("**Scene totals**")
            st.bar_chart(
                pd.DataFrame(
                    {"Detections": [result.car_count, len(people)]},
                    index=["Cars", "People"],
                ),
                alt="Car and person detection totals in the image",
            )
    confidences = [car.confidence for car in result.cars] + [
        person.confidence for person in people
    ]
    with confidence_column:
        with st.container(border=True):
            st.markdown("**Detection confidence distribution**")
            if len(confidences) < 2:
                st.caption("At least two detections are needed to show a confidence distribution.")
            else:
                histogram, edges = np.histogram(confidences, bins=np.linspace(0.0, 1.0, 6))
                confidence_frame = pd.DataFrame(
                    {
                        "Confidence range": [
                            f"{edges[index]:.0%}–{edges[index + 1]:.0%}"
                            for index in range(len(histogram))
                        ],
                        "Detections": histogram,
                    }
                )
                st.bar_chart(
                    confidence_frame,
                    x="Confidence range",
                    y="Detections",
                    alt="Actual detector scores grouped into confidence intervals",
                )
    if result.car_count == 0 and result.person_count == 0:
        st.caption("Charts reflect an empty but successfully processed detection result.")


def _render_video_analytics(result: VideoInferenceResult) -> None:
    st.markdown("**Frame-by-frame detections**")
    frame_data = pd.DataFrame(
        [
            {
                "Frame": frame.frame_index,
                "Cars": frame.car_count,
                "People": frame.person_count,
                "Blue cars": frame.blue_car_count,
                "Other known-colour cars": frame.other_known_colour_car_count,
            }
            for frame in result.frames
        ]
    )
    if frame_data.empty:
        st.info("No decodable frames were available for analytics.")
    else:
        st.line_chart(
            frame_data,
            x="Frame",
            y=["Cars", "People", "Blue cars", "Other known-colour cars"],
            alt="Actual car, person, blue-car, and other-known-colour detections by video frame",
        )
    fps_data = pd.DataFrame(
        {
            "FPS": [
                result.source_fps or 0.0,
                result.output_fps,
                result.processing_fps,
            ]
        },
        index=["Source", "Output", "Processing"],
    )
    with st.container(border=True):
        st.markdown("**Video rate comparison**")
        st.bar_chart(
            fps_data,
            alt="Source, output, and measured processing FPS for this video",
        )
    st.caption(
        "FPS values are distinct measurements: source metadata, encoded output rate, and measured "
        "processing throughput. A missing source rate is displayed as 0 in the chart and as unknown "
        "in processing information."
    )


def _render_processing_information(result: TaskResult) -> None:
    detector_metadata = result.inference_metadata
    if isinstance(result, VideoInferenceResult):
        metrics = [
            ("Source FPS", f"{result.source_fps:.2f}" if result.source_fps is not None else "Unknown"),
            ("Output FPS", f"{result.output_fps:.2f}"),
            ("Processing FPS", f"{result.processing_fps:.2f}"),
            ("Frames processed", f"{result.frame_count:,}"),
            (
                "Processing duration",
                f"{result.frame_count / result.processing_fps:.2f} s"
                if result.processing_fps > 0
                else "—",
            ),
        ]
    else:
        metrics = [
            ("Source FPS", "—"),
            ("Output FPS", "—"),
            ("Processing FPS", "—"),
            ("Frames processed", "1 image"),
            (
                "Processing duration",
                f"{float(detector_metadata.get('inference_duration_ms', 0.0)) / 1000:.2f} s",
            ),
        ]
    st.markdown('<div class="vt-panel"><h3>Runtime summary</h3></div>', unsafe_allow_html=True)
    columns = st.columns(5)
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value, border=True)
    st.caption(
        f"Model: {detector_metadata.get('model_name', 'YOLO11n')} · "
        f"Ultralytics {detector_metadata.get('ultralytics_version', 'version unavailable')} · "
        f"Detection threshold: {float(detector_metadata.get('confidence_threshold', 0)):.0%} · "
        f"Colour evidence threshold: "
        f"{float(detector_metadata.get('colour_confidence_threshold', 0)):.0%}"
    )
    with st.expander("Model and runtime metadata"):
        safe_metadata = {
            key: str(value)
            for key, value in detector_metadata.items()
            if key not in {"model_path"}
        }
        st.dataframe(
            pd.DataFrame(
                [{"Property": key.replace("_", " ").title(), "Value": value} for key, value in safe_metadata.items()]
            ),
            hide_index=True,
            alt="Model version, checksum, thresholds, and inference metadata",
        )


def _video_mime_type(extension: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }.get(extension.lower(), "application/octet-stream")
