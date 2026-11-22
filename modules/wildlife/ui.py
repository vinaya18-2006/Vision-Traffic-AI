"""Streamlit presentation for Wildlife Risk Intelligence."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Mapping, Protocol, TypeAlias

import cv2
import pandas as pd
import streamlit as st

from .contracts import (
    DetectionSummary,
    ImageAnalysisResult,
    VideoAnalysisResult,
    WildlifeDetection,
)
from .detector import WildlifePipeline, WildlifeYoloDetector, default_model_path
from .errors import MediaInputError, ModelLoadError, TaxonomyConfigError, VideoOutputError, WildlifeError
from .taxonomy import default_taxonomy_path, load_taxonomy
from .utils import decode_image, supported_image_extensions, supported_video_extensions
from .video import DEFAULT_MAX_VIDEO_FRAMES, infer_video

LOGGER = logging.getLogger(__name__)
AnalysisResult: TypeAlias = ImageAnalysisResult | VideoAnalysisResult


class _UploadedMedia(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


_GENERATION_KEY = "wildlife_reset_generation"
_RESULT_KEY = "wildlife_result"
_RESULT_MODE_KEY = "wildlife_result_mode"
_SOURCE_KEY = "wildlife_source_bytes"
_SOURCE_NAME_KEY = "wildlife_source_name"
_VIDEO_KEY = "wildlife_video_bytes"
_ERROR_KEY = "wildlife_error"


@st.cache_resource(max_entries=4)
def _cached_detector(model_path: str, animal_classes: tuple[str, ...]) -> WildlifeYoloDetector:
    """Cache each explicitly configured local animal detector."""
    return WildlifeYoloDetector(model_path, animal_classes=animal_classes)


def render_wildlife() -> None:
    """Render the standalone wildlife dashboard without running inference on import."""
    generation = st.session_state.setdefault(_GENERATION_KEY, 0)
    _render_header()
    _render_page_styles()

    mode = st.segmented_control(
        "Input type",
        options=["Image", "Video"],
        default="Image",
        key=f"wildlife_mode_{generation}",
        help="Choose whether to analyze one image or process a video frame by frame.",
    )
    selected_mode = mode or "Image"
    image_extensions = supported_image_extensions()
    video_extensions, backends = supported_video_extensions()
    accepted_extensions = image_extensions if selected_mode == "Image" else video_extensions
    upload = st.file_uploader(
        f"Upload a {selected_mode.lower()}",
        type=[extension.lstrip(".") for extension in accepted_extensions],
        key=f"wildlife_upload_{generation}_{selected_mode.lower()}",
        help="Only locally configured decoders and model weights are used.",
    )

    with st.sidebar:
        st.subheader("Wildlife analysis")
        confidence = st.slider(
            "Detection confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Detections below this model-score threshold are excluded.",
            key=f"wildlife_confidence_{generation}",
        )
        with st.expander("Model and processing settings"):
            model_path = st.text_input(
                "Local YOLO weights",
                value=str(default_model_path()),
                help="Select an existing local weights file. Model weights are never downloaded automatically.",
                key=f"wildlife_model_path_{generation}",
            )
            max_frames = st.number_input(
                "Maximum video frames",
                min_value=1,
                max_value=100000,
                value=DEFAULT_MAX_VIDEO_FRAMES,
                step=250,
                key=f"wildlife_max_frames_{generation}",
            )
        taxonomy_path = default_taxonomy_path()
        st.caption(f"Taxonomy: `{taxonomy_path}`")
        st.caption(
            f"Image formats: {', '.join(image_extensions) or 'none detected'}. "
            f"Video formats: {', '.join(video_extensions) or 'none detected'} "
            f"({', '.join(backends) or 'no OpenCV backends'})."
        )

    action_col, reset_col = st.columns([3, 1])
    with action_col:
        analyze = st.button(
            "Analyze media",
            type="primary",
            icon=":material/pets:",
            width="stretch",
            key=f"wildlife_analyze_{generation}",
        )
    with reset_col:
        reset = st.button(
            "Reset",
            icon=":material/refresh:",
            width="stretch",
            key=f"wildlife_reset_{generation}",
        )
    if reset:
        _reset_dashboard()
        st.rerun()
    if analyze:
        with st.spinner("Loading local model and analyzing wildlife detections..."):
            _run_analysis(
                selected_mode,
                upload,
                model_path,
                confidence,
                int(max_frames),
                taxonomy_path,
            )

    error = st.session_state.get(_ERROR_KEY)
    if error:
        st.error(error, icon=":material/error:")
    result: AnalysisResult | None = st.session_state.get(_RESULT_KEY)
    if result is None:
        _render_empty_state(selected_mode, upload)
        return

    result_mode = st.session_state.get(_RESULT_MODE_KEY, selected_mode)
    source_name = st.session_state.get(_SOURCE_NAME_KEY, "previous input")
    if result_mode != selected_mode or (upload is not None and upload.name != source_name):
        st.caption(
            f"Showing the last completed {result_mode.lower()} analysis for **{source_name}**. "
            "Analyze the selected input to refresh these results."
        )
    _render_results(
        result,
        st.session_state.get(_SOURCE_KEY),
        source_name,
        st.session_state.get(_VIDEO_KEY),
    )


def _run_analysis(
    mode: str,
    upload: _UploadedMedia | None,
    model_path_text: str,
    confidence_threshold: float,
    max_frames: int,
    taxonomy_path: Path,
) -> None:
    """Run only actual inference after the user explicitly submits an upload."""
    _clear_results()
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
        taxonomy = load_taxonomy(taxonomy_path)
        detector = _cached_detector(
            str(Path(model_path_text).expanduser().resolve()),
            taxonomy.animal_classes,
        )
        pipeline = WildlifePipeline(
            confidence_threshold=confidence_threshold,
            taxonomy=taxonomy,
            detector=detector,
        )
        if mode == "Image":
            result: AnalysisResult = pipeline.infer_image(source_bytes)
            video_bytes = None
        else:
            with tempfile.TemporaryDirectory(prefix="roadsight-wildlife-") as directory:
                source_path = Path(directory) / source_name
                output_path = Path(directory) / f"annotated{suffix}"
                source_path.write_bytes(source_bytes)
                result = infer_video(
                    pipeline,
                    source_path,
                    output_path,
                    max_frames=max_frames,
                )
                video_bytes = output_path.read_bytes()
        st.session_state[_RESULT_KEY] = result
        st.session_state[_RESULT_MODE_KEY] = mode
        st.session_state[_SOURCE_KEY] = source_bytes
        st.session_state[_SOURCE_NAME_KEY] = source_name
        if video_bytes is not None:
            st.session_state[_VIDEO_KEY] = video_bytes
    except (ModelLoadError, TaxonomyConfigError) as exc:
        LOGGER.exception("Wildlife model or taxonomy is unavailable")
        st.session_state[_ERROR_KEY] = f"Configuration unavailable: {exc}"
    except MediaInputError as exc:
        LOGGER.exception("Wildlife media could not be decoded")
        st.session_state[_ERROR_KEY] = f"Input processing failed: {exc}"
    except (VideoOutputError, WildlifeError) as exc:
        LOGGER.exception("Wildlife analysis could not be completed")
        st.session_state[_ERROR_KEY] = f"Analysis failed: {exc}"
    except (OSError, ValueError, cv2.error) as exc:
        LOGGER.exception("Wildlife analysis encountered an input or codec error")
        st.session_state[_ERROR_KEY] = f"Could not process this input: {exc}"


def _render_header() -> None:
    st.markdown(
        """
        <div class="wildlife-header">
          <div>
            <div class="wildlife-eyebrow">ROADSIGHT INTELLIGENCE · TASK 2</div>
            <h1>Wildlife Risk Intelligence</h1>
            <p>Evidence-based animal detections for safer roads and habitats.</p>
          </div>
          <div class="wildlife-badge">LOCAL MODEL · NO TRACKING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top:1.5rem; max-width:1440px;}
        .wildlife-header {
          display:flex; align-items:center; justify-content:space-between; gap:1rem;
          padding:1.4rem 1.6rem; margin-bottom:1rem; border-radius:20px;
          color:#f3f8f1; border:1px solid rgba(151,184,128,.32);
          background:linear-gradient(125deg,#15271e,#1c3828 58%,#244734);
          box-shadow:0 14px 36px rgba(18,43,28,.2);
        }
        .wildlife-eyebrow {font-size:.72rem;letter-spacing:.15em;font-weight:700;color:#b9d99b;}
        .wildlife-header h1 {margin:.4rem 0;font-size:2rem;letter-spacing:-.03em;}
        .wildlife-header p {margin:0;color:#d7e5d4;}
        .wildlife-badge {padding:.5rem .8rem;border:1px solid #78956e;border-radius:999px;
          color:#e7f3df;font-size:.75rem;font-weight:700;white-space:nowrap;}
        @media(max-width:700px) {.wildlife-header{align-items:flex-start;flex-direction:column;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state(mode: str, upload: _UploadedMedia | None) -> None:
    if st.session_state.get(_ERROR_KEY):
        return
    st.info(
        "Upload an image or video, choose the confidence threshold, and select **Analyze media**. "
        "No detection statistics are shown until model inference has completed."
    )
    if upload is None:
        return
    if mode == "Image":
        try:
            image = decode_image(upload.getvalue())
        except MediaInputError as exc:
            st.error(f"Could not preview this image: {exc}")
        else:
            st.image(
                image,
                channels="BGR",
                alt="Uploaded wildlife image before analysis",
                width="stretch",
            )
    else:
        st.caption("The selected video will be decoded and analyzed after submission.")


def _render_results(
    result: AnalysisResult,
    source_bytes: bytes | None,
    source_name: str,
    video_bytes: bytes | None,
) -> None:
    summary = result.summary
    is_video = isinstance(result, VideoAnalysisResult)
    st.success(
        f"Carnivores detected: **{summary.carnivores}**"
        + (" detections across processed frames." if is_video else " in this image."),
        icon=":material/pets:",
    )
    if is_video:
        st.caption(
            "Video totals are detections summed across processed frames, not unique animals. "
            "Object tracking is not implemented."
        )
    _render_kpis(summary, is_video)

    original_col, annotated_col = st.columns(2)
    if isinstance(result, ImageAnalysisResult):
        if source_bytes is not None:
            original_col.image(
                source_bytes,
                alt="Original uploaded wildlife image",
                caption="Original image",
                width="stretch",
            )
        annotated_col.image(
            result.annotated_image,
            channels="BGR",
            alt="Wildlife image annotated with taxonomy-coloured detection boxes",
            caption="Annotated image",
            width="stretch",
        )
        success, encoded = cv2.imencode(".png", result.annotated_image)
        if not success:
            st.error("The annotated image could not be encoded for download.")
        else:
            st.download_button(
                "Download annotated image",
                data=encoded.tobytes(),
                file_name=f"{Path(source_name).stem}_wildlife.png",
                mime="image/png",
                icon=":material/download:",
            )
        detections = result.detections
    else:
        original_col.image(
            result.original_preview_frame,
            channels="BGR",
            alt="First original video frame",
            caption="Original video · first processed frame",
            width="stretch",
        )
        annotated_col.image(
            result.annotated_preview_frame,
            channels="BGR",
            alt="First annotated video frame",
            caption="Annotated video · first processed frame",
            width="stretch",
        )
        if video_bytes is None:
            st.error("The processed video is unavailable for download.")
        else:
            st.download_button(
                "Download processed video",
                data=video_bytes,
                file_name=f"{Path(source_name).stem}_wildlife{Path(source_name).suffix.lower()}",
                mime=_video_mime(Path(source_name).suffix.lower()),
                icon=":material/download:",
            )
        _render_video_analytics(result)
        detections = tuple(
            detection for frame in result.frames for detection in frame.detections
        )

    _render_summaries(summary)
    _render_species_chart(summary.species_counts)
    _render_detection_details(detections)
    _render_taxonomy_notes(result.inference_metadata)


def _render_kpis(summary: DetectionSummary, is_video: bool) -> None:
    label = "Animal detections (frames)" if is_video else "Total animals"
    confidence = summary.average_confidence
    with st.container(horizontal=True):
        st.metric(label, summary.total_animals, border=True)
        st.metric(
            "Carnivore detections" if is_video else "Carnivores",
            summary.carnivores,
            border=True,
        )
        st.metric(
            "Non-carnivore detections" if is_video else "Non-carnivores",
            summary.non_carnivores,
            border=True,
        )
        st.metric(
            "Average detection confidence",
            f"{confidence:.1%}" if confidence is not None else "—",
            border=True,
        )


def _render_summaries(summary: DetectionSummary) -> None:
    st.subheader("Carnivore and non-carnivore summary")
    carnivore_col, other_col, unknown_col = st.columns(3)
    carnivore_col.metric("Carnivores", summary.carnivores)
    other_col.metric("Non-carnivores", summary.non_carnivores)
    unknown_col.metric("Unknown / not classified", summary.unknown)
    if summary.carnivore_species_counts:
        breakdown = ", ".join(
            f"{species}: {count}"
            for species, count in summary.carnivore_species_counts.items()
        )
        st.caption(f"Carnivore species breakdown: {breakdown}")
    if summary.unknown:
        st.warning(
            f"{summary.unknown} detections have unknown or context-dependent diet "
            "classification; they are not counted as carnivores."
        )


def _render_species_chart(species_counts: Mapping[str, int]) -> None:
    st.subheader("Species distribution")
    if not species_counts:
        st.info("No animal detections are available for this chart.")
        return
    chart_data = pd.DataFrame(
        [
            {"Species / model class": species, "Detections": count}
            for species, count in species_counts.items()
        ]
    )
    st.bar_chart(
        chart_data,
        x="Species / model class",
        y="Detections",
        alt="Animal detections by model class",
    )


def _render_detection_details(detections: tuple[WildlifeDetection, ...]) -> None:
    st.subheader("Detection details")
    if not detections:
        st.info("The model returned no detections at the selected confidence threshold.")
        return
    rows = [
        {
            "Species / model class": detection.species,
            "Confidence": detection.confidence,
            "Bounding box (x1, y1, x2, y2)": detection.bounding_box.as_xyxy(),
            "Classification": _classification_label(detection.classification),
        }
        for detection in detections
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        alt="Animal model labels, confidence, bounding boxes, and diet classifications",
        width="stretch",
    )


def _render_video_analytics(result: VideoAnalysisResult) -> None:
    st.subheader("Video analytics")
    frame_col, source_col, processing_col = st.columns(3)
    frame_col.metric("Frames processed", result.frame_count)
    source_col.metric(
        "Source FPS",
        f"{result.source_fps:.2f}" if result.source_fps is not None else "Unavailable",
    )
    processing_col.metric("Processing FPS", f"{result.processing_fps:.2f}")
    st.caption(
        f"Per-frame detections: {result.total_per_frame_detections}. "
        f"Unique tracked animals: not reported (tracking is not implemented)."
    )
    if result.frame_limit_reached:
        st.warning(
            f"The safety limit of {result.max_frames} frames was reached; remaining frames "
            "were not processed."
        )
    per_frame = pd.DataFrame(
        [
            {
                "Frame": frame.frame_index,
                "Animal detections": frame.animal_count,
                "Carnivore detections": frame.carnivore_count,
            }
            for frame in result.frames
        ]
    )
    st.dataframe(
        per_frame,
        hide_index=True,
        alt="Animal and carnivore detection totals for each processed video frame",
        width="stretch",
    )


def _render_taxonomy_notes(metadata: Mapping[str, object]) -> None:
    st.caption(
        f"Taxonomy version {metadata.get('taxonomy_version', 'unknown')} · "
        f"{metadata.get('taxonomy_source', 'mapping source unavailable')}"
    )
    with st.expander("Model vocabulary and classification notes"):
        st.write(
            "Labels are the model's class names; broad COCO classes such as “bird” are not "
            "fine-grained species identifications. Diet categories follow the separately "
            "versioned taxonomy file."
        )
        class_map = metadata.get("class_map", {})
        labels = class_map.values() if isinstance(class_map, Mapping) else ()
        st.write(
            "Animal labels available from the loaded model:",
            ", ".join(str(label) for label in labels),
        )
        st.write("Model:", metadata.get("model_name", "unknown"))
        st.write("Weights SHA-256:", metadata.get("model_sha256", "unavailable"))
        notes = metadata.get("taxonomy_notes", {})
        for species, note in notes.items() if isinstance(notes, Mapping) else ():
            st.caption(f"{species}: {note}")
        references = metadata.get("taxonomy_references", ())
        if isinstance(references, (tuple, list)):
            for reference in references:
                if not isinstance(reference, Mapping):
                    continue
                st.markdown(f"- [{reference['title']}]({reference['url']})")


def _reset_dashboard() -> None:
    _clear_results()
    st.session_state[_GENERATION_KEY] = st.session_state.get(_GENERATION_KEY, 0) + 1


def _clear_results() -> None:
    for key in (
        _RESULT_KEY,
        _RESULT_MODE_KEY,
        _SOURCE_KEY,
        _SOURCE_NAME_KEY,
        _VIDEO_KEY,
        _ERROR_KEY,
    ):
        st.session_state.pop(key, None)


def _classification_label(category: str) -> str:
    return {
        "carnivore": "Carnivore",
        "non_carnivore": "Non-carnivore",
        "unknown": "Unknown / not classified",
    }[category]


def _video_mime(extension: str) -> str:
    return {
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mp4": "video/mp4",
    }.get(extension, "application/octet-stream")
