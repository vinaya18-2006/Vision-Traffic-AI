"""Streamlit presentation for voice-emotion analysis."""

from __future__ import annotations

import streamlit as st

from .detector import MODEL_ID, MODEL_LABELS, VoiceEmotionPipeline, default_model_path
from .errors import InferenceError, MediaInputError, ModelLoadError


def render_voice_emotion() -> None:
    """Render the voice-emotion upload and explicit model status."""
    st.subheader("Voice emotion detection")
    st.caption(
        "Estimates one broad speech-emotion class; it does not establish the speaker's "
        "internal state or infer gender or identity."
    )
    st.info(
        f"Model: {MODEL_ID}. Labels: {', '.join(MODEL_LABELS.values())}. "
        f"Local weights directory: `{default_model_path()}`. "
        "Model files and Transformers dependencies are not provisioned automatically."
    )
    uploaded = st.file_uploader(
        "Upload a short PCM WAV voice clip",
        type=["wav"],
        help="Mono or stereo PCM WAV; up to 30 seconds and 25 MiB.",
    )

    with st.sidebar:
        confidence = st.slider(
            "Minimum model score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Adds a warning below this model score; it does not replace the predicted label.",
        )

    if uploaded is None:
        st.warning(
            "A local pretrained model is required. No prediction is available until its "
            "weights and reviewed dependencies are explicitly provisioned."
        )
        return

    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes, format="audio/wav", alt=f"Uploaded audio: {uploaded.name}")
    if st.button("Analyze audio", type="primary"):
        try:
            pipeline = VoiceEmotionPipeline(confidence_threshold=confidence)
            result = pipeline.infer_audio(audio_bytes, source_name=uploaded.name)
            st.success(f"Model classification: {result.label}")
            st.metric("Model score (not calibrated)", f"{result.confidence:.2%}")
            if result.warning:
                st.warning(result.warning)
            st.json(
                {
                    "label": result.label,
                    "model_score": round(result.confidence, 4),
                    "sample_rate": result.sample_rate,
                    "duration_seconds": round(result.duration_seconds or 0.0, 3),
                    "metadata": dict(result.inference_metadata),
                }
            )
        except (ModelLoadError, MediaInputError, InferenceError) as exc:
            st.error(f"Voice analysis unavailable: {exc}")
