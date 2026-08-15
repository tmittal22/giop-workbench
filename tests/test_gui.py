"""GUI smoke test: drive the real application through load -> invert -> export.

Skipped without a display. This is not a widget-level test; it checks the one property
that matters, that the GUI path produces the same numbers as the API path, so the GUI
cannot silently drift into being a different calculation.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY"), reason="no X display available"
)


@pytest.fixture
def app():
    tk = pytest.importorskip("tkinter")
    try:
        from giop.gui import GiopApp

        a = GiopApp()
    except tk.TclError as exc:                       # display present but unusable
        pytest.skip(f"tkinter could not open a window: {exc}")
    a.update()
    yield a
    a.destroy()


def test_gui_reproduces_the_golden_eigenvalues(app):
    app.load_demo()
    app.run()
    app.update()
    assert app.result is not None and app.result.converged
    np.testing.assert_allclose(
        app.result.x, [0.0441, 0.0033, 0.3693], atol=5e-5
    )


def test_gui_status_line_reports_the_prescribed_shapes(app):
    """The shapes are prescribed, so the GUI must show them. A user who cannot see
    S_dg and eta cannot know what the retrieval was conditional on."""
    app.load_demo()
    app.run()
    status = app.status.get()
    assert "S_dg" in status and "eta" in status and "QC" in status


def test_gui_refuses_an_invalid_configuration_without_crashing(app, monkeypatch):
    """Morel + LMI is a division by zero (PORTING_NOTES D2). The GUI must surface it
    as a message, not raise out of the event loop."""
    shown = {}
    monkeypatch.setattr("giop.gui.messagebox.showerror",
                        lambda title, msg: shown.update(title=title, msg=msg))
    app.load_demo()
    app.fq.set("morel")
    app.inv.set("lmi")
    app.run()
    assert "Configuration refused" in shown.get("title", "")
    assert "division by zero" in shown.get("msg", "")


def test_gui_warns_instead_of_inverting_with_no_data(app, monkeypatch):
    shown = {}
    monkeypatch.setattr("giop.gui.messagebox.showwarning",
                        lambda title, msg: shown.update(title=title))
    app.run()
    assert shown.get("title") == "No data"
