"""Physics-level checks: water IOPs, the LMI solver identity, and closure/recovery."""

import numpy as np
import pytest
import scipy.io as sio
import scipy.linalg as sla
from importlib.resources import files

from giop import GiopConfig, giop
from giop.model import ConfigurationError, rrs_above_to_below, rrs_below_to_above, rrs_from_iops
from giop.water import a_water, bb_water


class TestPureWater:
    def test_aw_matches_the_independent_in_repo_copy(self):
        """``optics_coef.txt`` col 2 and ``pureH2O_iop.mat`` are two separate encodings
        of the same quantity shipped in the same repo. They must agree.

        This is the only independent oracle for a_w available offline, and it is a
        real one: the .mat file documents its own provenance (Pope & Fry 1997 over
        380-700 nm) in a NOTES field, and nothing in the code path reads it.
        """
        mat = sio.loadmat(files("giop.data").joinpath("pureH2O_iop.mat"))
        wl_ref = mat["wave"].ravel().astype(float)
        aw_ref = mat["aw"].ravel().astype(float)

        m = (wl_ref >= 400) & (wl_ref <= 700)
        got = a_water(wl_ref[m])
        rel = np.abs(got - aw_ref[m]) / aw_ref[m]
        assert np.nanmax(rel) < 0.02, f"max relative disagreement {np.nanmax(rel):.3%}"

    def test_aw_has_the_expected_red_rise(self):
        """a_w must rise by ~2 orders of magnitude from 440 to 700 nm."""
        assert a_water(700.0) / a_water(440.0) > 50

    def test_aw_is_nan_outside_the_table(self):
        assert np.isnan(a_water(200.0))
        assert np.isnan(a_water(2000.0))

    def test_bbw_power_law(self):
        """G6 exactly, and the b_w/2 consistency noted in water.optics_coef_table."""
        assert bb_water(400.0) == pytest.approx(0.0038)
        assert bb_water(443.0) == pytest.approx(0.0024447, rel=1e-4)
        # blue/red ratio follows the 4.32 exponent
        assert bb_water(400.0) / bb_water(800.0) == pytest.approx(2 ** 4.32, rel=1e-9)

    def test_bbw_table_mode_requires_a_table(self):
        with pytest.raises(ValueError, match="requires bbw_table"):
            bb_water(443.0, model="table")


class TestAirSeaInterface:
    def test_round_trip_is_exact(self):
        """G4a and G4c must be exact inverses; this is what lets the port report a
        model spectrum above water, which upstream never does (PORTING_NOTES D1)."""
        rrs = np.array([1e-4, 1e-3, 5e-3, 2e-2])
        back = rrs_below_to_above(rrs_above_to_below(rrs, "lee"), "lee")
        np.testing.assert_allclose(back, rrs, rtol=1e-12)

    def test_flat_round_trip_is_exact(self):
        rrs = np.array([1e-4, 5e-3])
        np.testing.assert_allclose(
            rrs_below_to_above(rrs_above_to_below(rrs, "flat"), "flat"), rrs, rtol=1e-12
        )

    def test_subsurface_is_larger_than_above_surface(self):
        """r_rs > R_rs always: the interface transmits less than unity."""
        rrs = np.array([1e-3, 5e-3])
        assert np.all(rrs_above_to_below(rrs, "lee") > rrs)


class TestLmiSolver:
    """THEORY.md sect. 6.2: upstream's four-line QR expression is the normal-equations
    solution computed through the QR factor, plus one refinement step."""

    @staticmethod
    def _system(seed=0):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(6, 3))
        b = rng.normal(size=6)
        return A, b

    def test_semi_normal_equations_equal_least_squares(self):
        A, b = self._system()
        _, R = sla.qr(A, mode="economic")
        x = sla.solve_triangular(R, sla.solve_triangular(R.T, A.T @ b, lower=True))
        r = b - A @ x
        x = x + sla.solve_triangular(
            R, sla.solve_triangular(R.T, A.T @ r, lower=True)
        )
        ref = sla.lstsq(A, b)[0]
        np.testing.assert_allclose(x, ref, rtol=1e-9)

    def test_control_normal_equations_without_qr_differ_on_ill_conditioning(self):
        """Control: the identity above is not vacuous. Explicitly forming A'A and
        inverting it loses accuracy on an ill-conditioned system, so the two routes
        are numerically distinct even though they are algebraically equal."""
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 3))
        A[:, 2] = A[:, 1] + 1e-8 * A[:, 2]   # near-collinear columns
        b = rng.normal(size=6)
        naive = np.linalg.inv(A.T @ A) @ (A.T @ b)
        ref = sla.lstsq(A, b)[0]
        assert not np.allclose(naive, ref, rtol=1e-6)


class TestForwardInverseConsistency:
    """Synthetic recovery. This is a self-consistency test, not an independent
    validation: it uses the same forward model to generate and to invert. It proves
    the solver finds the minimum it is supposed to find, nothing more. The
    independent anchor is test_golden.py."""

    WL = np.array([412.0, 443, 490, 510, 555, 670])

    def _synthesise(self, adg443, bbp443, mphi, chl_seed=0.5):
        from giop.model import eigenvectors
        from giop.water import a_water, bb_water

        cfg = GiopConfig()
        rin_dummy = np.ones_like(self.WL) * 1e-3
        idx = {"412": 0, "443": 1, "555": 4}
        adgs, bbps, aphs, _, _ = eigenvectors(
            self.WL, cfg, chl_seed, rin_dummy, rin_dummy, idx
        )
        aw, bbw = a_water(self.WL), bb_water(self.WL)
        a = aw + adg443 * adgs + mphi * aphs
        bb = bbw + bbp443 * bbps
        rin = rrs_from_iops(a, bb, 0.0949, 0.0794)
        return rrs_below_to_above(rin, "lee"), (adgs, bbps, aphs)

    @pytest.mark.parametrize(
        "truth", [(0.02, 0.002, 0.3), (0.10, 0.01, 2.0), (0.005, 0.0005, 0.05)]
    )
    def test_recovers_known_amplitudes(self, truth):
        adg443, bbp443, mphi = truth
        rrs, (adgs, bbps, aphs) = self._synthesise(*truth)
        # invert with the SAME eigenvectors, so only the amplitudes are unknown
        res = giop(self.WL, rrs, 0.5, cfg=GiopConfig(aph=aphs, sdg=0.018, eta="qaa"))
        # eta is data-derived, so pin it to what the synthesis used
        eta_used = np.log(bbps[0]) / np.log(443.0 / self.WL[0])
        res = giop(self.WL, rrs, 0.5,
                   cfg=GiopConfig(aph=aphs, sdg=0.018, eta=float(eta_used)))
        np.testing.assert_allclose(res.x, [adg443, bbp443, mphi], rtol=1e-3)

    def test_lmi_recovers_the_same_truth(self):
        truth = (0.02, 0.002, 0.3)
        rrs, (adgs, bbps, aphs) = self._synthesise(*truth)
        eta_used = np.log(bbps[0]) / np.log(443.0 / self.WL[0])
        res = giop(self.WL, rrs, 0.5, cfg=GiopConfig(
            aph=aphs, sdg=0.018, eta=float(eta_used), inv="lmi"))
        np.testing.assert_allclose(res.x, truth, rtol=1e-6)

    def test_control_wrong_eigenvector_breaks_recovery(self):
        """If the prescribed shape is wrong the amplitudes are wrong. This is
        assumption A2/A3 made visible, and it is why the recovery test above is only
        a solver check."""
        truth = (0.02, 0.002, 0.3)
        rrs, (adgs, bbps, aphs) = self._synthesise(*truth)
        eta_used = np.log(bbps[0]) / np.log(443.0 / self.WL[0])
        res = giop(self.WL, rrs, 0.5, cfg=GiopConfig(
            aph=aphs, sdg=0.030, eta=float(eta_used)))   # wrong S_dg
        assert not np.allclose(res.x, truth, rtol=0.05)


class TestConfigGuards:
    WL = np.array([412.0, 443, 490, 510, 555, 670])
    RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])

    def test_morel_plus_lmi_is_refused(self):
        """PORTING_NOTES D2: g1 = 0 under Morel and the LMI branch divides by g1.
        MATLAB returns Inf/NaN silently; the port refuses."""
        with pytest.raises(ConfigurationError, match="division by zero"):
            giop(self.WL, self.RRS, 0.5, fq="morel", inv="lmi")

    def test_missing_anchor_band_is_refused(self):
        with pytest.raises(ConfigurationError, match="requires a band within"):
            giop(np.array([600.0, 650, 700]), np.array([1e-3, 1e-3, 1e-3]), 0.5)

    def test_percent_reflectance_is_caught(self):
        """A reflectance factor in percent is the most likely field-data unit error."""
        with pytest.raises(ValueError, match="looks like percent"):
            giop(self.WL, self.RRS * 1000, 0.5)

    def test_micrometre_wavelengths_are_caught(self):
        with pytest.raises(ValueError, match="micrometres"):
            giop(self.WL / 1000.0, self.RRS, 0.5)

    def test_out_of_range_aph_window_is_explained(self):
        """aph* is tabulated 400-700 nm; a NaturaSpec spectrum runs far past that.

        The 1 nm grid here is what the instrument actually delivers, so the anchor
        bands are all present and the failure is genuinely the aph* window.
        """
        wl = np.arange(400.0, 900.0, 1.0)
        rrs = np.full_like(wl, 2e-3)
        with pytest.raises(ConfigurationError, match="400-700 nm"):
            giop(wl, rrs, 0.5)

    def test_a_coarse_grid_without_443_is_refused_clearly(self):
        """A 10 nm grid straddles 443 and GIOP cannot run on it. The message has to
        say so rather than failing later inside the eigenvector construction."""
        wl = np.arange(400.0, 700.0, 10.0)
        rrs = np.full_like(wl, 2e-3)
        with pytest.raises(ConfigurationError, match="within 2.5 nm of 443"):
            giop(wl, rrs, 0.5)
