"""fit_shapes must not silently do nothing."""

import numpy as np
import pytest

from giop import giop
from giop.model import ConfigurationError

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.0014, 0.0025, 0.0045, 0.0056, 0.0087, 0.0052])
SIG = np.full(6, 1e-4)


def test_bounded_plus_fit_shapes_is_refused():
    """It was a SILENT NO-OP: the bounded path ignored fit_shapes and returned a
    fixed-shape answer bitwise identical to not asking, which reads as a successful
    shape fit. Refusing is the only safe behaviour until it is implemented."""
    with pytest.raises(ConfigurationError, match="not implemented for inv='bounded'"):
        giop(WL, RRS, 10.0, inv="bounded", sigma=SIG, fit_shapes=True)


def test_the_message_says_what_to_do_instead():
    try:
        giop(WL, RRS, 10.0, inv="bounded", sigma=SIG, fit_shapes=True)
    except ConfigurationError as exc:
        assert "fmin" in str(exc)
        assert "assumption" in str(exc)


def test_bounded_without_fit_shapes_still_works():
    """The guard must not break the normal path."""
    g = giop(WL, RRS, 10.0, inv="bounded", sigma=SIG)
    assert not g.failed
    assert np.all(g.x >= -1e-12)


def test_fmin_plus_fit_shapes_still_works():
    g = giop(WL, RRS, 10.0, inv="fmin", fit_shapes=True)
    assert np.isfinite(g.sdg) and np.isfinite(g.eta)
