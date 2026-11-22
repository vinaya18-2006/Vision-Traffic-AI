from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_navigation_exposes_drowsiness_unavailable_state() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    navigation = next(
        widget
        for widget in app.sidebar.selectbox
        if widget.label == "RoadSight Intelligence"
    )
    navigation.set_value("Drowsiness Detection")
    app.run(timeout=20)

    assert any(
        "Drowsiness inference is unavailable" in element.value
        for element in app.error
    )
    assert any(
        "no age model is configured" in element.value.lower()
        for element in app.info
    )
