"""Integrated Streamlit entry point for the implemented Task 1 module."""

import streamlit as st

from modules.car_detection.ui import render_task1

st.set_page_config(
    page_title="VisionTraffic AI",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_task1()
