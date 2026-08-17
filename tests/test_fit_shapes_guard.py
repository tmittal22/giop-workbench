"""fit_shapes must not silently do nothing, and must not come back WORSE than fixed.

History, both defects found on real turbid field data:

1. ``inv='bounded'`` ignored ``fit_shapes`` entirely and returned a fixed-shape answer
   bitwise identical to not asking for it. Guarded, then implemented.
2. ``inv='fmin'`` with ``fit_shapes`` DID search, but as a joint 5-D Nelder-Mead from the
   upstream oligotrophic start behind a ``return 1e6`` barrier. It railed both shapes and
   returned chi2_nu = 2431 against the fixed-shape 74.5 on the same spectrum. Freeing
   parameters cannot make the optimum worse -- the fixed-shape point is nested inside the
   free box -- so that number was an optimiser failure, and it had been read as evidence
   that the data cannot constrain the shapes.

The nesting test below is the one that fails against both versions.
"""

import numpy as np
import pytest

from giop import giop
from giop.model import ConfigurationError

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.0014, 0.0025, 0.0045, 0.0056, 0.0087, 0.0052])
SIG = np.full(6, 1e-4)

# A turbid hyperspectral spectrum, the case that exposed the defect: a green peak near
# 570 nm on a strongly absorbing blue end, which puts M_dg ~ 1 and M_bp ~ 0.08 rather
# than the upstream oligotrophic 0.01 / 0.001 the old start assumed.
WLH = np.arange(400.0, 701.0, 1.0)
RRSH = (0.0008 + 0.011 * np.exp(-0.5 * ((WLH - 572.0) / 62.0) ** 2)
        + 0.0016 * np.exp(-0.5 * ((WLH - 690.0) / 14.0) ** 2))
SIGH = 0.019 * RRSH


def _cost(wl, rrs, weight=None, **kw):
    """Misfit recomputed from the returned model spectrum, not read off ``.cost``.

    ``.cost`` is whatever objective that solver minimised, and it is reported on the
    SUBSURFACE scale, so it is not comparable across configurations. This recomputes one
    statistic above water.

    ``weight`` must be the objective the path being tested actually minimises --
    sigma-weighted for ``inv='bounded'`` with a sigma, unweighted for ``inv='fmin'``.
    Grading a path on an objective it did not minimise is not a nesting violation, and
    an earlier version of this test made exactly that mistake and read a legitimate
    1.05x as an optimiser failure.
    """
    g = giop(wl, rrs, 10.0, **kw)
    assert not g.failed
    w = 1.0 if weight is None else weight
    return float(np.sum(((g.rrs_model_above - rrs) / w) ** 2)), g


@pytest.mark.parametrize("inv", ["fmin", "bounded"])
def test_freeing_the_shapes_never_fits_worse_than_fixing_them(inv):
    """THE nesting property. Fixed shapes are a point inside the free search box."""
    kw = dict(inv=inv, sigma=SIGH) if inv == "bounded" else dict(inv=inv)
    w = SIGH if inv == "bounded" else None
    fixed, _ = _cost(WLH, RRSH, weight=w, **kw)
    free, gf = _cost(WLH, RRSH, weight=w, fit_shapes=True, n_starts=4, **kw)
    assert free <= fixed * 1.0001, (
        "fit_shapes returned chi2=%.4g against the fixed-shape %.4g, i.e. %.1fx WORSE "
        "than the special case nested inside its own search box. That is an optimiser "
        "failure, not a statement about the data." % (free, fixed, free / fixed))
    assert np.isfinite(gf.sdg) and np.isfinite(gf.eta)


def test_the_shapes_actually_move():
    """Not a no-op: the returned S_dg/eta differ from the configured ones."""
    fix = giop(WLH, RRSH, 10.0, inv="bounded", sigma=SIGH)
    fre = giop(WLH, RRSH, 10.0, inv="bounded", sigma=SIGH, fit_shapes=True, n_starts=4)
    assert (abs(fre.sdg - fix.sdg) > 1e-6) or (abs(fre.eta - fix.eta) > 1e-6)


def test_shapes_stay_inside_their_box():
    from giop.inversion import ETA_BOUNDS, SDG_BOUNDS

    g = giop(WLH, RRSH, 10.0, inv="bounded", sigma=SIGH, fit_shapes=True, n_starts=4)
    assert SDG_BOUNDS[0] - 1e-9 <= g.sdg <= SDG_BOUNDS[1] + 1e-9
    assert ETA_BOUNDS[0] - 1e-9 <= g.eta <= ETA_BOUNDS[1] + 1e-9


def test_n_starts_is_honoured_not_ignored():
    """The joint version took no n_starts argument at all, so this was unreachable."""
    lo, _ = _cost(WLH, RRSH, weight=SIGH, inv="bounded", sigma=SIGH, fit_shapes=True,
                  n_starts=1)
    hi, _ = _cost(WLH, RRSH, weight=SIGH, inv="bounded", sigma=SIGH, fit_shapes=True,
                  n_starts=6)
    assert hi <= lo * 1.0001, "more starts must not find a worse optimum"


def test_bounded_plus_fit_shapes_is_no_longer_refused():
    g = giop(WL, RRS, 10.0, inv="bounded", sigma=SIG, fit_shapes=True)
    assert not g.failed
    assert np.all(g.x >= -1e-12), "the bounded path must still enforce positivity"


def test_bounded_without_fit_shapes_still_works():
    g = giop(WL, RRS, 10.0, inv="bounded", sigma=SIG)
    assert not g.failed
    assert np.all(g.x >= -1e-12)


def test_fmin_plus_fit_shapes_still_works():
    g = giop(WL, RRS, 10.0, inv="fmin", fit_shapes=True)
    assert np.isfinite(g.sdg) and np.isfinite(g.eta)


def test_lmi_plus_fit_shapes_is_still_refused():
    """LMI is linear in the amplitudes; the shapes are not, so it cannot be done there."""
    with pytest.raises(ConfigurationError, match="nonlinear"):
        giop(WL, RRS, 10.0, inv="lmi", fit_shapes=True)
