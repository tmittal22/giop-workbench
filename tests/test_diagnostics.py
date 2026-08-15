"""Identifiability diagnostics and the two uncertainty estimators. THEORY.md sect. 6.3, 7."""

import numpy as np
import pytest

from giop import get_oc, giop
from giop.diagnostics import (
    DEGENERACY_THRESHOLD_DEG,
    condition_number,
    design_matrix,
    eigenvector_angles,
    report,
)
from giop.uncertainty import linearised_covariance, shape_ensemble

WL = np.array([412.0, 443, 490, 510, 555, 670])
RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])


@pytest.fixture(scope="module")
def result():
    chl = float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))
    return giop(WL, RRS, chl, qc=0.33)


class TestDiagnostics:
    def test_design_matrix_shape_and_signs(self, result):
        """b_bp enters with the opposite sign to the absorbers: more backscatter raises
        r_rs, more absorption lowers it. If that sign flipped, the LMI solve would
        return a b_bp of the wrong sign and still 'converge'."""
        A = design_matrix(result)
        assert A.shape == (len(WL), 3)
        assert np.all(A[:, 0] > 0) and np.all(A[:, 2] > 0)   # u * eigenvector
        assert np.all(A[:, 1] < 0)                            # (u - 1) * eigenvector

    def test_angles_are_measured_not_assumed(self, result):
        ang = eigenvector_angles(result)
        assert set(ang) == {"adg-bbp", "adg-aph", "bbp-aph"}
        assert all(0 <= v <= 90 for v in ang.values())
        # measured on this spectrum; pinned so a change in the eigenvectors is visible
        assert ang["adg-aph"] == pytest.approx(27.5, abs=0.5)
        assert ang["adg-bbp"] == pytest.approx(33.4, abs=0.5)

    def test_demo_spectrum_is_not_degenerate_at_fixed_shapes(self, result):
        assert min(eigenvector_angles(result).values()) > DEGENERACY_THRESHOLD_DEG
        assert condition_number(result) < 5e3

    def test_report_warns_that_angles_are_shape_conditional(self, result):
        """The failure mode this guards against is quoting a good condition number as
        if it meant the retrieval were accurate."""
        text = report(result)
        assert "shape_ensemble" in text or "DEGENERATE" in text

    def test_control_a_collinear_library_is_detected(self, result):
        """Control: make a_dg and a_phi nearly identical and the diagnostic must fire.
        Otherwise it cannot detect degeneracy at all."""
        import copy

        r = copy.copy(result)
        r.aphs = r.adgs * 0.5 + 1e-6 * r.aphs
        ang = eigenvector_angles(r)
        assert ang["adg-aph"] < DEGENERACY_THRESHOLD_DEG
        assert "DEGENERATE PAIRS" in report(r)


class TestLinearisedCovariance:
    def test_sigma_scales_linearly_with_measurement_noise(self, result):
        _, s1 = linearised_covariance(result, 1e-5)
        _, s2 = linearised_covariance(result, 2e-5)
        np.testing.assert_allclose(s2, 2 * s1, rtol=1e-9)

    def test_adg_and_aph_are_anticorrelated(self, result):
        """The CDOM/phytoplankton trade-off, as a number rather than an assertion."""
        cov, _ = linearised_covariance(result, 5e-5)
        rho = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])
        assert rho < -0.3

    def test_rejects_non_positive_sigma(self, result):
        with pytest.raises(ValueError, match="positive"):
            linearised_covariance(result, 0.0)


class TestShapeEnsemble:
    @pytest.fixture(scope="class")
    def ens(self):
        chl = float(get_oc(RRS[1], RRS[2], RRS[3], RRS[4], "oc4"))
        return shape_ensemble(WL, RRS, chl)

    def test_all_members_converge_on_the_demo_spectrum(self, ens):
        assert ens["n_members"] == 60 and ens["n_failed"] == 0

    def test_prescription_error_dominates_the_linearised_error(self, ens, result):
        """The central claim of THEORY.md sect. 7, measured.

        The ensemble spread on the aph amplitude must be far wider than the
        shape-conditional 1-sigma, or reporting the latter alone would be defensible.
        """
        _, sig = linearised_covariance(result, 5e-5)
        span = ens["aph_amplitude"]["max"] - ens["aph_amplitude"]["min"]
        assert span > 20 * sig[2]

    def test_negative_aph_members_are_counted_not_hidden(self, ens):
        assert ens["negative_fraction"]["aph_amplitude"] > 0.1
        assert ens["negative_fraction"]["adg443"] == 0.0

    def test_apg_is_better_constrained_than_its_split(self, ens):
        """a_pg = a_dg + a_phi should vary proportionally less across the ensemble than
        either component, which is why the total is the defensible product."""
        X = np.array([m[3] for m in ens["members"]])
        adg, aph = X[:, 0], X[:, 2]
        apg = adg + aph
        cv = lambda v: np.std(v) / abs(np.mean(v))  # noqa: E731
        assert cv(apg) < cv(aph)
