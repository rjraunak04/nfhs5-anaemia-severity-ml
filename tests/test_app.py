"""Smoke test for the recruiter-facing Streamlit app."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parents[1]


def test_streamlit_app_starts_with_truth_disclosure() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "NFHS-5 Anaemia Severity ML"
    assert any("not an NFHS research result" in item.value for item in app.markdown)
    assert any(
        metric.label == "Locked test" and metric.value == "Untouched" for metric in app.metric
    )
