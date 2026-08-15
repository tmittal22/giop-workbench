"""The Streamlit app, executed with Streamlit's own test harness.

This is not a screenshot test. It checks the one property that matters: the app path and
the API path produce the SAME numbers, so the GUI cannot quietly drift into being a
different calculation.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "app", "giop_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_loads_without_exceptions(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_has_every_panel(app):
    assert len(app.tabs) == 7


def test_demo_inversion_reproduces_the_published_eigenvalues():
    """run_giop.m states 0.0441 / 0.0033 / 0.3693. The GUI must return those too."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    for b in at.button:
        if "demo" in b.label.lower():
            b.click(); at.run(); break
    for b in at.button:
        if "Run inversion" in b.label:
            b.click(); at.run(); break
    assert not at.exception, [str(e.value) for e in at.exception]

    got = {m.label: m.value for m in at.metric}
    assert float(got["a_dg(443)  m⁻¹"]) == pytest.approx(0.0441, abs=5e-5)
    assert float(got["b_bp(443)  m⁻¹"]) == pytest.approx(0.0033, abs=5e-5)
    assert float(got["aph amplitude"]) == pytest.approx(0.3693, abs=5e-5)


def test_identifiability_panel_reports_the_condition_number():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    for b in at.button:
        if "demo" in b.label.lower():
            b.click(); at.run(); break
    for b in at.button:
        if "Run inversion" in b.label:
            b.click(); at.run(); break
    labels = [m.label for m in at.metric]
    assert "cond(A)" in labels
