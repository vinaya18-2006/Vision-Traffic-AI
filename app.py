"""Integrated Streamlit entry point for RoadSight Intelligence."""

import streamlit as st

from modules.car_detection.ui import render_task1
from modules.drowsiness.ui import render_drowsiness
from modules.person_analysis.ui import render_person_analysis
from modules.sign_language.ui import render_sign_language
from modules.voice_emotion.ui import render_voice_emotion
from modules.wildlife.ui import render_wildlife

st.set_page_config(
    page_title="VisionTraffic AI",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.sidebar.selectbox(
    "RoadSight Intelligence",
    [
        "Car Colour Detection",
        "Wildlife Risk Intelligence",
        "Voice Emotion Detection",
        "Person Attribute & Emotion Analysis",
        "Sign Language Detection",
        "Drowsiness Detection",
    ],
)
if page == "Wildlife Risk Intelligence":
    render_wildlife()
elif page == "Voice Emotion Detection":
    render_voice_emotion()
elif page == "Person Attribute & Emotion Analysis":
    render_person_analysis()
elif page == "Sign Language Detection":
    render_sign_language()
elif page == "Drowsiness Detection":
    render_drowsiness()
else:
    render_task1()
