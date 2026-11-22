"""Streamlit interface for static ASL fingerspelling recognition."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from .config import SignLanguageConfig, default_model_path
from .contracts import ImageRecognitionResult, VideoRecognitionResult
from .detector import (
    KerasLandmarkClassifier,
    MediaPipeHandExtractor,
    SignLanguagePipeline,
)
from .errors import SignLanguageError
from .preprocessing import (
    load_vocabulary,
    supported_image_extensions,
    supported_video_extensions,
)
from .video import DEFAULT_MAX_VIDEO_FRAMES, infer_video

LOGGER = logging.getLogger(__name__)
_RESULT_KEY = "sign_language_result"
_RESULT_MODE_KEY = "sign_language_result_mode"
_VIDEO_BYTES_KEY = "sign_language_video_bytes"
_ERROR_KEY = "sign_language_error"


@st.cache_resource(max_entries=4)
def _cached_classifier(model_path: str, labels: tuple[str, ...]) -> KerasLandmarkClassifier:
    return KerasLandmarkClassifier(model_path, labels)


@st.cache_resource(max_entries=1)
def _cached_extractor() -> MediaPipeHandExtractor:
    return MediaPipeHandExtractor()


def render_sign_language() -> None:
    """Render upload controls, model scope, authorization, and results."""
    st.subheader("Sign Language Detection")
    st.caption(
        "Static ASL fingerspelled-letter recognition only. This is not word, sentence, "
        "grammar, or unrestricted sign-language translation."
    )

    try:
        environment_config = SignLanguageConfig.from_environment()
        vocabulary = load_vocabulary()
    except SignLanguageError as exc:
        st.error(f"Sign-recognition configuration is invalid: {exc}")
        return

    model_path_text = str(default_model_path())
    with st.sidebar:
        st.subheader("ASL recognition")
        model_path_text = st.text_input(
            "Local landmark model",
            value=model_path_text,
            help="The pinned Keras checkpoint is loaded only after recognition is requested.",
            key="sign_language_model_path",
        )
        minimum_score = st.slider(
            "Minimum model score",
            min_value=0.0,
            max_value=1.0,
            value=float(environment_config.minimum_score),
            step=0.05,
            help="Below this uncalibrated model score, the result is shown as unknown.",
            key="sign_language_minimum_score",
        )
        max_frames = st.number_input(
            "Maximum video frames",
            min_value=1,
            max_value=100000,
            value=DEFAULT_MAX_VIDEO_FRAMES,
            step=250,
            key="sign_language_max_frames",
        )
        st.caption(
            "Authorized daily window: "
            f"{environment_config.allowed_start:%H:%M}–"
            f"{environment_config.allowed_end:%H:%M} "
            f"{environment_config.timezone_name} "
            "(start-inclusive/end-exclusive)."
        )

    st.write("**Sign language:** American Sign Language (ASL)")
    st.write("**Supported static letters:** " + ", ".join(vocabulary))
    st.warning(
        "J and Z require motion and are not supported. The publisher does not document "
        "the checkpoint's training dataset or signer split; its reported 0.9415 validation "
        "accuracy is unverified. Low-score rejection is not a validated out-of-vocabulary detector."
    )
    if not Path(model_path_text).expanduser().is_file():
        st.info(
            "Model unavailable. Provision the checksum-pinned local checkpoint with "
            "`.\\.venv\\Scripts\\python.exe -m modules.sign_language.provision_model`."
        )

    mode = st.segmented_control(
        "Input type",
        options=["Image", "Video"],
        default="Image",
        key="sign_language_input_type",
    )
    selected_mode = mode or "Image"
    image_extensions = supported_image_extensions()
    video_extensions, backends = supported_video_extensions()
    accepted_extensions = image_extensions if selected_mode == "Image" else video_extensions
    article = "an" if selected_mode == "Image" else "a"
    upload = st.file_uploader(
        f"Upload {article} {selected_mode.lower()}",
        type=[extension.lstrip(".") for extension in accepted_extensions],
        key="sign_language_upload",
        help="Input must be decodable by the installed Pillow/OpenCV backend.",
    )
    if selected_mode == "Video":
        st.caption(
            "Video is classified independently per frame. No temporal sign recognition, "
            "letter aggregation, or word formation is performed. Available OpenCV backends: "
            f"{', '.join(backends) or 'none'}."
        )

    if st.button(
        "Recognize static ASL letters",
        type="primary",
        icon=":material/sign_language:",
        width="stretch",
        key="sign_language_submit",
    ):
        _run_analysis(
            selected_mode,
            upload,
            model_path_text,
            minimum_score,
            int(max_frames),
            environment_config,
        )

    if st.session_state.get(_ERROR_KEY):
        st.error(st.session_state[_ERROR_KEY], icon=":material/error:")
    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        return
    result_mode = st.session_state.get(_RESULT_MODE_KEY)
    if result_mode != selected_mode:
        st.caption(f"Showing the last completed {result_mode.lower()} analysis.")
    if isinstance(result, ImageRecognitionResult):
        _render_image_result(result)
    else:
        _render_video_result(result, st.session_state.get(_VIDEO_BYTES_KEY))


def _run_analysis(
    mode: str,
    upload,
    model_path_text: str,
    minimum_score: float,
    max_frames: int,
    environment_config: SignLanguageConfig,
) -> None:
    for key in (_RESULT_KEY, _RESULT_MODE_KEY, _VIDEO_BYTES_KEY, _ERROR_KEY):
        st.session_state.pop(key, None)
    if upload is None:
        st.session_state[_ERROR_KEY] = "Choose an image or video before recognition."
        return

    config = replace(
        environment_config,
        model_path=Path(model_path_text).expanduser().resolve(),
        minimum_score=minimum_score,
    )
    try:
        config.require_authorized()
        vocabulary = load_vocabulary()
        classifier = _cached_classifier(str(config.model_path), vocabulary)
        extractor = _cached_extractor()
        pipeline = SignLanguagePipeline(
            config=config,
            extractor=extractor,
            classifier=classifier,
        )
        if mode == "Image":
            result = pipeline.infer_image(upload.getvalue())
            st.session_state[_VIDEO_BYTES_KEY] = None
        else:
            suffix = Path(upload.name).suffix.lower()
            with tempfile.TemporaryDirectory(prefix="roadsight-sign-") as directory:
                source_path = Path(directory) / f"source{suffix}"
                output_path = Path(directory) / f"annotated{suffix}"
                source_path.write_bytes(upload.getvalue())

                progress = st.progress(0.0, text="Preparing per-frame recognition...")

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
                video_bytes = output_path.read_bytes()
                progress.empty()
            st.session_state[_VIDEO_BYTES_KEY] = video_bytes
        st.session_state[_RESULT_KEY] = result
        st.session_state[_RESULT_MODE_KEY] = mode
    except SignLanguageError as exc:
        LOGGER.warning("Sign-language recognition was unavailable: %s", exc)
        st.session_state[_ERROR_KEY] = str(exc)
    except (OSError, ValueError) as exc:
        LOGGER.exception("Sign-language input could not be processed")
        st.session_state[_ERROR_KEY] = f"Could not process this input: {exc}"
    except Exception:
        LOGGER.exception("Unexpected sign-language recognition failure")
        st.session_state[_ERROR_KEY] = (
            "Recognition could not be completed. Check the media and local model configuration."
        )


def _render_image_result(result: ImageRecognitionResult) -> None:
    prediction = result.prediction
    st.write(f"**Result:** {prediction.label if prediction.status == 'recognized' else 'Unknown'}")
    st.write(f"**Status:** {prediction.status.replace('_', ' ').title()}")
    if prediction.confidence is not None:
        st.metric("Model score (not calibrated)", f"{prediction.confidence:.3f}")
    if prediction.warning:
        st.warning(prediction.warning)
    st.image(
        result.annotated_image,
        channels="BGR",
        alt="Uploaded image annotated with static ASL letter result",
    )
    st.json(dict(result.inference_metadata))


def _render_video_result(
    result: VideoRecognitionResult,
    video_bytes: bytes | None,
) -> None:
    st.write(
        f"Processed {result.frame_count} frames; each frame is classified independently."
    )
    st.image(
        result.annotated_preview_frame,
        channels="BGR",
        alt="First annotated video frame with static ASL letter result",
    )
    if video_bytes:
        extension = result.output_video_path.suffix.lower()
        if extension in {".mp4", ".mov"}:
            st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=f"annotated_asl_static{extension}",
            mime="video/mp4" if extension in {".mp4", ".mov"} else "video/x-msvideo",
        )
    st.dataframe(
        [
            {
                "frame": item.frame_index,
                "time_seconds": round(item.timestamp_seconds, 3),
                "label": item.prediction.label,
                "status": item.prediction.status,
                "model_score": item.prediction.confidence,
            }
            for item in result.frames
        ],
        hide_index=True,
    )
    st.json(dict(result.inference_metadata))
