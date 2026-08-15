"""Sensor band definitions, spectral convolution, and the bounded solver."""

import numpy as np
import pytest

from giop import GiopConfig, get_oc, giop
from giop.sensors import SENSORS, band_table, convolve, modis_srf

WL = np.arange(350.0, 901.0, 1.0)
FLAT = np.full_like(WL, 5e-3)
DEMO_WL = np.array([412.0, 443, 490, 510, 555, 670])
DEMO_RRS = np.array([0.003478, 0.004074, 0.004465, 0.003588, 0.002494, 0.000051])


class TestModisSrf:
    def test_measured_srf_loads_with_the_expected_bands(self):
        wl, srf, labels = modis_srf()
        assert srf.shape[1] == 16 and len(labels) == 16
        assert wl.min() <= 400 and wl.max() >= 2000

    def test_each_label_falls_inside_its_own_bands_half_maximum(self):
        """The column order is an inference; this pins it against the data itself.

        A mislabelled column would silently return the wrong band's radiance, an error
        that never announces itself. The test is that the nominal label lies within the
        band's own half-maximum range, which is what "this is that band" means. A fixed
        tolerance in nm would be wrong here: the ocean-colour bands are ~10 nm wide while
        the SWIR bands are 25-52 nm wide and asymmetric (2130 has a measured centroid of
        2114), so the criterion has to scale with the band.
        """
        wl, srf, labels = modis_srf()
        for j, lab in enumerate(labels):
            r = srf[:, j]
            m = np.isfinite(r) & (r > 0)
            ww, rr = wl[m], r[m]
            edges = ww[rr >= rr.max() / 2.0]
            assert edges.min() <= lab <= edges.max(), (
                f"column {j} labelled {lab} has half-max range "
                f"{edges.min():.0f}-{edges.max():.0f} nm")

    def test_ocean_colour_bands_are_narrow_and_swir_bands_are_not(self):
        """Sanity on the response widths, which is what makes the tolerance above
        necessarily band-dependent."""
        wl, srf, labels = modis_srf()
        widths = {}
        for j, lab in enumerate(labels):
            r = srf[:, j]
            m = np.isfinite(r) & (r > 0)
            edges = wl[m][r[m] >= r[m].max() / 2.0]
            widths[lab] = edges.max() - edges.min()
        assert widths[443] <= 15 and widths[667] <= 15
        assert widths[2130] > 30


class TestConvolution:
    def test_a_flat_spectrum_convolves_to_itself(self):
        """Any correct weighted mean of a constant returns the constant."""
        for name in ("MODIS-Aqua", "Sentinel-3 OLCI", "SeaWiFS", "VIIRS-SNPP"):
            _, v, _ = convolve(WL, FLAT, name)
            good = np.isfinite(v)
            assert good.sum() >= 4
            np.testing.assert_allclose(v[good], 5e-3, rtol=1e-9)

    def test_native_mode_is_a_no_op(self):
        c, v, info = convolve(WL, FLAT, "Field hyperspectral (native)")
        assert info["method"] == "native"
        np.testing.assert_array_equal(c, WL)

    def test_bands_outside_the_measured_range_are_dropped_not_extrapolated(self):
        """A 400-500 nm spectrum cannot produce a 670 nm band value."""
        wl = np.arange(400.0, 501.0)
        _, v, info = convolve(wl, np.full_like(wl, 1e-3), "SeaWiFS", use_giop_bands=False)
        assert np.isnan(v).any()
        assert info["dropped"]

    def test_modis_uses_the_measured_srf_not_a_gaussian(self):
        _, _, info = convolve(WL, FLAT, "MODIS-Aqua")
        assert info["method"] == "measured SRF"
        _, _, info2 = convolve(WL, FLAT, "SeaWiFS")
        assert info2["method"] == "nominal Gaussian"

    def test_convolution_tracks_a_sloped_spectrum(self):
        """On a linear ramp the convolved value must equal the value at band centre,
        because a symmetric response integrates a straight line to its midpoint."""
        y = 1e-5 * (WL - 350.0)
        c, v, _ = convolve(WL, y, "SeaWiFS")
        expect = 1e-5 * (c - 350.0)
        np.testing.assert_allclose(v, expect, rtol=2e-2)

    def test_band_placement_actually_changes_the_answer(self):
        """OLCI 560 and SeaWiFS 555 must differ on a spectrum with green structure,
        otherwise this whole panel is decorative."""
        y = 5e-3 * np.exp(-((WL - 555.0) ** 2) / (2 * 25.0 ** 2)) + 1e-3
        _, v_olci, _ = convolve(WL, y, "Sentinel-3 OLCI")
        _, v_swf, _ = convolve(WL, y, "SeaWiFS")
        assert abs(v_olci[-2] - v_swf[-2]) > 0

    def test_every_sensor_declares_its_response_fidelity(self):
        for name, s in SENSORS.items():
            assert s.response in ("measured", "gaussian", "none")
            assert s.note, f"{name} has no note explaining its fidelity"

    def test_band_table_marks_the_giop_subset(self):
        rows = band_table("SeaWiFS")
        assert any(used for *_, used in rows)
        assert any(not used for *_, used in rows)


@pytest.fixture(scope="module")
def chl():
    """Module-level: pytest >= 8.4 rejects a class-scoped fixture written as an
    instance method, and older pytest never shared its state the way it looked like
    it did."""
    return float(get_oc(DEMO_RRS[1], DEMO_RRS[2], DEMO_RRS[3], DEMO_RRS[4], "oc4"))


class TestBoundedSolver:
    SIGMA = np.sqrt((0.05 * np.abs(DEMO_RRS)) ** 2 + 2e-4 ** 2)

    def test_agrees_with_giop_dc_where_the_answer_is_physical(self, chl):
        a = giop(DEMO_WL, DEMO_RRS, chl)
        b = giop(DEMO_WL, DEMO_RRS, chl, inv="bounded", sigma=self.SIGMA)
        np.testing.assert_allclose(b.x, a.x, rtol=0.02)

    def test_never_returns_negative_absorption(self, chl):
        """The whole point. GIOP-DC returns aph = -0.275 at S_dg = 0.010."""
        for s in (0.010, 0.012, 0.014, 0.018, 0.025):
            b = giop(DEMO_WL, DEMO_RRS, chl, sdg=s, inv="bounded", sigma=self.SIGMA)
            assert np.all(b.x >= -1e-12), f"S_dg={s} gave {b.x}"

    def test_control_giop_dc_does_go_negative(self, chl):
        """If GIOP-DC never went negative, the bounded solver would be solving nothing."""
        a = giop(DEMO_WL, DEMO_RRS, chl, sdg=0.010)
        assert a.x[2] < 0

    def test_rejects_bad_sigma(self, chl):
        with pytest.raises(ValueError):
            giop(DEMO_WL, DEMO_RRS, chl, inv="bounded",
                 sigma=np.zeros_like(DEMO_RRS))

    def test_unknown_solver_is_refused_with_the_valid_set(self):
        from giop.model import ConfigurationError
        with pytest.raises(ConfigurationError, match="bounded"):
            giop(DEMO_WL, DEMO_RRS, 0.5, inv="nonsense")
