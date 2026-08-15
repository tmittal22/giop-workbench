"""MATLAB-semantics layer. THEORY.md sect. 9, PORTING_NOTES.md."""

import numpy as np
import pytest

from giop.matlab_compat import fastsmooth, interp1, nearest_band, v5cubic


class TestFastsmooth:
    """``fastsmooth.m`` ships its own worked examples in its header comments.

    Those two lines are a published golden master for this function, which matters
    because the upstream file has CR-only line endings and renders as one line with a
    mangled ``ends(k+halfw)=`` token; the port had to reconstruct the intent.
    """

    Y = np.array([1, 1, 1, 10, 10, 10, 1, 1, 1, 1], dtype=float)

    def test_boxcar_ends_zero(self):
        # fastsmooth([1 1 1 10 10 10 1 1 1 1],3) = [0 1 4 7 10 7 4 1 1 0]
        got = fastsmooth(self.Y, 3)
        np.testing.assert_allclose(got, [0, 1, 4, 7, 10, 7, 4, 1, 1, 0])

    def test_boxcar_ends_tapered(self):
        # fastsmooth([1 1 1 10 10 10 1 1 1 1],3,1,1) = [1 1 4 7 10 7 4 1 1 1]
        got = fastsmooth(self.Y, 3, 1, 1)
        np.testing.assert_allclose(got, [1, 1, 4, 7, 10, 7, 4, 1, 1, 1])

    def test_control_wrong_window_fails_the_golden(self):
        """A width-5 smooth must not reproduce the width-3 published answer."""
        got = fastsmooth(self.Y, 5)
        assert not np.allclose(got, [0, 1, 4, 7, 10, 7, 4, 1, 1, 0])


class TestInterp1:
    def test_pchip_and_v5cubic_agree_on_grid_nodes(self):
        """On uniform-grid nodes both interpolants return the tabulated value.

        This is why the published golden numbers cannot discriminate between MATLAB's
        modern and historical 'cubic': the demo wavelengths land on 1 nm table nodes.
        THEORY.md sect. 9.1.
        """
        x = np.arange(400.0, 701.0)
        y = np.exp(-((x - 500) ** 2) / 2000.0)
        nodes = np.array([412.0, 443, 490, 510, 555, 670])
        np.testing.assert_allclose(interp1(x, y, nodes, "pchip"),
                                   interp1(x, y, nodes, "v5cubic"), atol=1e-12)

    def test_pchip_and_v5cubic_differ_off_node(self):
        """Off-node they differ, so the choice is a real decision on field grids."""
        x = np.arange(400.0, 701.0, 5.0)
        y = np.exp(-((x - 500) ** 2) / 2000.0)
        q = np.array([412.3, 443.7, 501.1])
        a = interp1(x, y, q, "pchip")
        b = interp1(x, y, q, "v5cubic")
        assert np.max(np.abs(a - b)) > 1e-6

    def test_v5cubic_refuses_non_uniform_grid(self):
        """MATLAB's v5cubic needs uniform x. This is the inferred reason upstream's
        ``giop_kb.m:125`` changed the 6-band GSM aph* call from 'cubic' to 'pchip'."""
        x = np.array([412.0, 443, 490, 510, 555, 670])
        with pytest.raises(ValueError, match="uniformly spaced"):
            v5cubic(x, np.ones_like(x), np.array([500.0]))

    def test_returns_nan_outside_range_like_matlab(self):
        x = np.arange(400.0, 701.0)
        y = np.ones_like(x)
        out = interp1(x, y, np.array([399.0, 500.0, 701.0]))
        assert np.isnan(out[0]) and np.isnan(out[2]) and out[1] == 1.0

    def test_pchip_does_not_overshoot_where_v5cubic_does(self):
        """Shape preservation is the substantive difference between the two."""
        x = np.arange(0.0, 10.0)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
        q = np.linspace(0, 9, 200)
        assert interp1(x, y, q, "pchip").min() >= -1e-12
        assert interp1(x, y, q, "v5cubic").min() < -1e-6


class TestBandSelection:
    def test_nearest_band_beats_first_in_window_on_dense_grids(self):
        """Upstream takes the first index in [410, 415]; on a 1 nm grid that is 410.

        THEORY.md sect. 9.3 / PORTING_NOTES.md D5.
        """
        wl = np.arange(400.0, 701.0)
        first_in_window = int(np.flatnonzero((wl >= 410) & (wl <= 415))[0])
        assert wl[first_in_window] == 410.0
        assert wl[nearest_band(wl, 412.0, 3.0)] == 412.0

    def test_nearest_band_returns_none_when_out_of_tolerance(self):
        wl = np.array([600.0, 700.0])
        assert nearest_band(wl, 412.0, 3.0) is None

    def test_agrees_with_upstream_on_the_six_band_grid(self):
        """On the satellite band set the two rules must pick the same band, or the
        golden test and the port would be measuring different things."""
        wl = np.array([412.0, 443, 490, 510, 555, 670])
        for target, tol, lo, hi in ((412.0, 3.0, 410, 415),
                                    (443.0, 2.5, 441, 445),
                                    (551.5, 6.0, 546, 557)):
            first = int(np.flatnonzero((wl >= lo) & (wl <= hi))[0])
            assert nearest_band(wl, target, tol) == first
