"""Golden-master tests against the values published in upstream ``run_giop.m``.

``run_giop.m:88-89`` states the fmin branch "should report 0.0441, 0.0033, and 0.3693
to screen as eigenvectors for adg, bbp, and aph", and ``run_giop.m:103-104`` states the
lmi branch should report 0.0414, 0.0022, 0.1058. Those are the only published outputs
of the upstream code, and no MATLAB/Octave is available on this host, so they are the
primary anchor for the port.

Each test is paired with a control that breaks one specific piece of the physics and
asserts the golden test then FAILS, so a pass here is informative.
"""

import numpy as np
import pytest

from giop import GiopConfig, get_oc, giop
from giop.aphstar import bricaud1998

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])

# Published in run_giop.m. Quoted to 4 decimals, so that is the comparison tolerance.
GOLDEN_FMIN = np.array([0.0441, 0.0033, 0.3693])
GOLDEN_LMI = np.array([0.0414, 0.0022, 0.1058])
ATOL = 5e-5


@pytest.fixture(scope="module")
def chl_seed():
    return float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))


def test_oc4_seed_chlorophyll(chl_seed):
    """OC4 must produce the seed chl that the golden eigenvalues were generated with.

    This is not independently published, but it is load-bearing: the Bricaud aph*
    eigenvector is a function of it, so if OC4 were wrong the aph eigenvalue could not
    match 0.3693.
    """
    assert 0.5 < chl_seed < 0.55


def test_golden_fmin(chl_seed):
    res = giop(WL, RRS, chl_seed, qc=0.33)
    assert res.converged
    assert res.qc_passed is True
    np.testing.assert_allclose(res.x, GOLDEN_FMIN, atol=ATOL)


def test_golden_lmi(chl_seed):
    res = giop(WL, RRS, chl_seed, inv="lmi", qc=0.33)
    assert res.converged
    np.testing.assert_allclose(res.x, GOLDEN_LMI, atol=ATOL)


def test_golden_fmin_is_reproducible(chl_seed):
    """Nelder-Mead from a fixed start is deterministic; two calls must agree bitwise."""
    a = giop(WL, RRS, chl_seed, qc=0.33).x
    b = giop(WL, RRS, chl_seed, qc=0.33).x
    assert np.array_equal(a, b)


# --- controls: each breaks one thing and must make the golden test fail -----------


def test_control_bricaud_normalisation_wavelength(chl_seed):
    """The Bricaud aph* renormalisation is anchored at 442 nm, not 443 (THEORY.md G8).

    'Fixing' it to 443 (by interpolating the table there) moves the aph eigenvalue off
    the published value, which is what makes the 442 in ``get_bricaud_aph.m:24``
    load-bearing rather than a typo to clean up.
    """
    aphs442 = bricaud1998(chl_seed, WL, normalize=True)

    # renormalise at 443 instead, via interpolation of the un-normalised vector
    aphs_raw = bricaud1998(chl_seed, WL, normalize=False)
    from giop.aphstar import _bricaud_table
    from giop.matlab_compat import interp1

    dat = _bricaud_table()
    raw = dat[:, 3] * chl_seed ** (dat[:, 4] - 1.0)
    at443 = interp1(dat[:, 0], raw, 443.0)
    aphs443 = aphs_raw * 0.055 / at443

    assert not np.allclose(aphs442, aphs443, rtol=1e-6), (
        "442 and 443 normalisation must actually differ or this control is vacuous"
    )

    res = giop(WL, RRS, chl_seed, cfg=GiopConfig(aph=aphs443, qc=0.33))
    assert not np.allclose(res.x, GOLDEN_FMIN, atol=ATOL), (
        "golden test cannot fail: it passes even with the wrong normalisation anchor"
    )


def test_control_wrong_gordon_coefficients(chl_seed):
    """Perturbing (g0, g1) by 10 % must break the golden match."""
    res = giop(WL, RRS, chl_seed, fq=(0.0949 * 1.1, 0.0794), qc=0.33)
    assert not np.allclose(res.x, GOLDEN_FMIN, atol=ATOL)


def test_control_flat_air_sea_transmission(chl_seed):
    """The Gordon 2007 flat transmission is a different model and must not match."""
    res = giop(WL, RRS, chl_seed, trans="flat", qc=0.33)
    assert not np.allclose(res.x, GOLDEN_FMIN, atol=ATOL)


def test_control_lmi_and_fmin_disagree(chl_seed):
    """The two solvers are genuinely different estimators, not aliases.

    If this ever passes, one branch is silently calling the other and both golden
    tests collapse to one.
    """
    a = giop(WL, RRS, chl_seed, qc=0.33).x
    b = giop(WL, RRS, chl_seed, inv="lmi", qc=0.33).x
    assert not np.allclose(a, b, rtol=0.05)
