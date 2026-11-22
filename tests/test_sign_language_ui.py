from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_navigation_exposes_sign_language_scope_and_model_notice() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert app.exception == []
    assert "Sign Language Detection" in app.selectbox[0].options
    app.selectbox[0].select("Sign Language Detection").run(timeout=30)

    assert app.exception == []
    assert any("Supported static letters" in element.value for element in app.markdown)
    assert any("J and Z require motion" in element.value for element in app.warning)
    assert any("not ASL translation" in element.value for element in app.caption)
