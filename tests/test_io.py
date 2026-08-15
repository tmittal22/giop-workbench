"""Ingestion layer: .sed parsing, above-water R_rs, resampling.

The synthetic .sed files here are written to the layout and column vocabulary
extracted from the string table of ``DARWin2.exe`` in the NaturaSpec Plus installer
(``Data:`` marker, tab-separated table, ``Wvl`` / ``Rad. (Ref.)`` / ``Rad. (Target)``
/ ``Reflect. %``, ``0.000000e+000`` 3-digit exponents). They are not a substitute for
a real instrument file; ``test_reads_a_real_file`` is skipped until one is present in
``tests/data/``.
"""

import numpy as np
import pytest
from pathlib import Path

from giop.io import (
    RHO_MOBLEY1999,
    align,
    bin_spectrum,
    gaussian_resample,
    read_sed,
    reflectance_factor_to_rrs,
    residual_correction,
    rrs_above_water,
)
from giop.io.rrs import rrs_from_sed_triplet

WL = np.arange(350.0, 1001.0, 1.0)


def write_sed(path, wl, rad_ref, rad_target, reflect_pct=None, comment="test",
              marker="Data:", extra_header=True):
    lines = []
    if extra_header:
        lines += [
            "Version: 2.1",
            f"File Name: C:\\DARWin\\{path.name}",
            "Instrument: NaturaSpecPlus_SN0000",
            "Detectors: 512,256,256",
            "Measurement: REFLECTANCE",
            "Date: 08/14/2026,08/14/2026",
            "Time: 12:00:00,12:00:30",
            "Temperature (C): 25.00,-5.00,-5.00",
            "Battery Voltage: 7.50",
            "Averages: 10,10",
            "Integration: 20,1,1",
            "Dark Mode: AUTO",
            "Foreoptic: LENS 4 DEG {RADIANCE}",
            "Radiometric Calibration: RADIANCE",
            "Units: W/m^2/sr/nm",
            "Wavelength Range: 350,2500",
            "Latitude: 40.79",
            "Longitude: -77.86",
            "Altitude: 350.00",
            "GPS Time: n/a",
            "Satellites: n/a",
            f"Comment: {comment}",
            "Channels: 651",
        ]
    cols = ["Wvl", "Rad. (Ref.)", "Rad. (Target)"]
    data = [wl, rad_ref, rad_target]
    if reflect_pct is not None:
        cols.append("Reflect. %")
        data.append(reflect_pct)
    lines.append(marker)
    lines.append("\t".join(cols))
    for row in zip(*data):
        lines.append("\t".join(_darwin_float(v) for v in row))
    path.write_text("\n".join(lines), encoding="latin-1")
    return path


def _darwin_float(v, digits=12):
    """DARWin's 3-digit-exponent style, e.g. ``0.000000e+000``.

    Written at higher precision than the instrument does so that file rounding is not
    the limiting error when two processing paths are compared against each other.
    """
    s = f"{v:.{digits}e}"
    mant, _, exp = s.partition("e")
    sign, mag = exp[0], int(exp[1:])
    return f"{mant}e{sign}{mag:03d}"


@pytest.fixture
def sed_triplet(tmp_path):
    """A water target, a sky scan and a panel scan sharing one reference panel."""
    ed = 1.0 * np.exp(-((WL - 550) ** 2) / (2 * 300.0 ** 2)) + 0.2   # smooth solar-ish
    l_panel = ed * 0.99 / np.pi
    l_sky = 0.02 * ed / np.pi
    rrs_true = 4e-3 * np.exp(-((WL - 490) ** 2) / (2 * 60.0 ** 2))
    l_water = rrs_true * ed + RHO_MOBLEY1999 * l_sky

    t = write_sed(tmp_path / "water.sed", WL, l_panel, l_water,
                  100 * l_water / l_panel, comment="water")
    s = write_sed(tmp_path / "sky.sed", WL, l_panel, l_sky,
                  100 * l_sky / l_panel, comment="sky")
    p = write_sed(tmp_path / "panel.sed", WL, l_panel, l_panel,
                  100 * np.ones_like(WL), comment="panel")
    return t, s, p, rrs_true


class TestSedReader:
    def test_parses_header_and_columns(self, sed_triplet):
        spec = read_sed(sed_triplet[0])
        assert spec.header["Instrument"] == "NaturaSpecPlus_SN0000"
        assert spec.comment == "water"
        assert spec.latitude == pytest.approx(40.79)
        assert spec.longitude == pytest.approx(-77.86)
        assert len(spec.wavelength) == len(WL)
        np.testing.assert_allclose(spec.wavelength, WL)
        assert "rad_target" in spec.columns and "rad_ref" in spec.columns

    def test_percent_reflectance_is_converted_to_fraction(self, sed_triplet):
        """``Reflect. %`` is 0-100 and ``Reflect. [1.0]`` is 0-1. Getting this wrong
        is a factor-100 error that would sail through the inversion as 'turbid'."""
        spec = read_sed(sed_triplet[2])          # panel vs itself = 100 %
        assert spec.reflectance_scale == 100.0
        np.testing.assert_allclose(spec.reflectance, 1.0, atol=1e-9)

    def test_unit_scale_variant(self, tmp_path):
        p = tmp_path / "u.sed"
        lines = ["Comment: x", "Data:", "Wvl\tReflect. [1.0]",
                 "400.0\t0.5", "401.0\t0.5"]
        p.write_text("\n".join(lines))
        spec = read_sed(p)
        assert spec.reflectance_scale == 1.0
        np.testing.assert_allclose(spec.reflectance, 0.5)

    def test_irr_alias_is_recognised(self, tmp_path):
        """DARWin writes ``Irr. (Ref.)``. Parsers that expect ``Irrad. (Ref.)`` drop
        the column silently; this pins the alias that the instrument actually uses."""
        p = tmp_path / "i.sed"
        p.write_text("Data:\nWvl\tIrr. (Ref.)\n400.0\t1.5\n401.0\t1.6\n")
        spec = read_sed(p)
        assert "irr_ref" in spec.columns
        np.testing.assert_allclose(spec.irradiance_reference, [1.5, 1.6])

    def test_missing_data_marker_is_a_clear_error(self, tmp_path):
        p = tmp_path / "bad.sed"
        p.write_text("Version: 2.1\nWvl\tReflect. %\n400\t50\n")
        with pytest.raises(ValueError, match="no 'Data:' marker"):
            read_sed(p)

    def test_missing_column_names_what_is_available(self, tmp_path):
        p = tmp_path / "c.sed"
        p.write_text("Data:\nWvl\tReflect. %\n400.0\t50\n401.0\t50\n")
        spec = read_sed(p)
        with pytest.raises(KeyError, match="Rad. \\(Target\\)"):
            _ = spec.radiance_target

    def test_clip_restricts_all_columns_together(self, sed_triplet):
        spec = read_sed(sed_triplet[0]).clip(400, 700)
        assert spec.wavelength.min() >= 400 and spec.wavelength.max() <= 700
        assert all(len(v) == len(spec.wavelength) for v in spec.columns.values())

    @pytest.mark.skipif(
        not list((Path(__file__).parent / "data").glob("*.sed")),
        reason="no real .sed file staged in tests/data/",
    )
    def test_reads_a_real_file(self):
        for p in (Path(__file__).parent / "data").glob("*.sed"):
            spec = read_sed(p)
            assert len(spec.wavelength) > 100
            assert np.all(np.diff(spec.wavelength) > 0)


class TestAboveWaterRrs:
    def test_recovers_the_synthetic_rrs(self, sed_triplet):
        """End-to-end closure of G16/G17 on a spectrum built from a known R_rs."""
        t, s, p, rrs_true = sed_triplet
        res = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                   panel_reflectance=0.99, rho=RHO_MOBLEY1999)
        np.testing.assert_allclose(res.rrs, rrs_true, atol=1e-9)

    def test_control_wrong_rho_biases_the_result(self, sed_triplet):
        """rho is the dominant error term (A10). If the retrieval were insensitive to
        it, the test above would not be testing the glint subtraction at all."""
        t, s, p, rrs_true = sed_triplet
        res = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                   panel_reflectance=0.99, rho=0.05)
        assert not np.allclose(res.rrs, rrs_true, atol=1e-9)
        assert np.all(res.rrs <= rrs_true + 1e-12)   # over-subtraction biases low

    def test_ratio_path_matches_radiance_path(self, sed_triplet):
        """The two ingestion routes must agree when one panel serves all scans."""
        t, s, p, _ = sed_triplet
        a = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                 panel_reflectance=0.99, use="radiance")
        b = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                 panel_reflectance=0.99, use="ratio")
        # Agreement is limited only by the written precision of the file, which the
        # helper deliberately sets well above DARWin's own.
        np.testing.assert_allclose(a.rrs, b.rrs, rtol=1e-8, atol=1e-15)

    def test_panel_reflectance_scales_linearly(self, sed_triplet):
        t, s, p, _ = sed_triplet
        a = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                 panel_reflectance=0.99)
        b = rrs_from_sed_triplet(read_sed(t), read_sed(s), read_sed(p),
                                 panel_reflectance=0.50)
        np.testing.assert_allclose(b.rrs, a.rrs * 0.50 / 0.99, rtol=1e-12)

    def test_over_subtraction_is_reported_not_hidden(self):
        wl = np.arange(400.0, 700.0)
        lt = np.full_like(wl, 1.0)
        ls = np.full_like(wl, 100.0)     # absurd sky, forces L_t < rho*L_sky
        lp = np.full_like(wl, 10.0)
        res = rrs_above_water(wl, lt, ls, lp)
        assert any("exceeds the measured upwelling radiance" in n for n in res.notes)

    def test_mismatched_grids_are_refused(self):
        wl = np.arange(400.0, 700.0)
        with pytest.raises(ValueError, match="same wavelength grid"):
            rrs_above_water(wl, np.ones_like(wl), np.ones(10), np.ones_like(wl))


class TestResidualCorrection:
    def test_nir_zero_removes_a_constant_offset(self):
        # Signal identically zero beyond 700 nm, so the recovered offset is exact and
        # the tolerance means something. A Gaussian tail would leave real signal in
        # the window and the correction would (correctly) remove some of it.
        wl = np.arange(400.0, 900.0)
        true = np.where(wl <= 700, 3e-3 * np.exp(-((wl - 500) ** 2) / (2 * 50.0 ** 2)), 0.0)
        off, method, notes = residual_correction(wl, true + 5e-4, "nir_zero")
        assert off == pytest.approx(5e-4, rel=1e-9)
        assert "Invalid in turbid water" in notes[0]

    def test_nir_zero_removes_real_signal_in_turbid_water(self):
        """The documented failure mode, made explicit: with genuinely non-zero NIR
        R_rs the correction subtracts water-leaving signal. THEORY.md sect. 10.3."""
        wl = np.arange(400.0, 900.0)
        turbid = np.full_like(wl, 2e-3)      # flat, non-zero everywhere including NIR
        off, _, _ = residual_correction(wl, turbid, "nir_zero")
        assert off == pytest.approx(2e-3, rel=1e-9)

    def test_none_is_the_default_and_does_nothing(self):
        wl = np.arange(400.0, 900.0)
        off, method, _ = residual_correction(wl, np.ones_like(wl), "none")
        assert off == 0.0 and method == "none"

    def test_nir_zero_needs_nir_bands(self):
        wl = np.arange(400.0, 700.0)
        with pytest.raises(ValueError, match="needs bands in"):
            residual_correction(wl, np.ones_like(wl), "nir_zero")


class TestResampling:
    def test_align_is_a_no_op_on_a_shared_grid(self):
        wl = np.arange(400.0, 700.0)
        a, b = np.sin(wl), np.cos(wl)
        grid, (oa, ob) = align([(wl, a), (wl, b)])
        assert grid is wl and oa is a and ob is b

    def test_bin_reports_empty_bins_as_nan(self):
        wl = np.arange(400.0, 500.0)
        vals = np.ones_like(wl)
        _, binned, n = bin_spectrum(wl, vals, [450.0, 900.0], width=10.0)
        assert binned[0] == pytest.approx(1.0) and n[0] == 11
        assert np.isnan(binned[1]) and n[1] == 0

    def test_gaussian_resample_preserves_a_constant(self):
        wl = np.arange(400.0, 700.0)
        _, out = gaussian_resample(wl, np.full_like(wl, 2.5), [450.0, 550.0], fwhm=10)
        np.testing.assert_allclose(out, 2.5, rtol=1e-9)

    def test_gaussian_resample_refuses_edge_centres(self):
        wl = np.arange(400.0, 700.0)
        _, out = gaussian_resample(wl, np.ones_like(wl), [395.0], fwhm=10)
        assert np.isnan(out[0])


class TestLambertianHelper:
    def test_reflectance_factor_conversion(self):
        np.testing.assert_allclose(reflectance_factor_to_rrs(np.pi), 1.0)

    def test_percent_input_warns(self):
        with pytest.warns(UserWarning, match="percent"):
            reflectance_factor_to_rrs(np.array([50.0]))

    def test_a_legitimate_bright_factor_does_not_warn(self):
        """Control: the percent guard must not fire on valid fractional data, or it
        trains the user to ignore it."""
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            reflectance_factor_to_rrs(np.array([0.99, np.pi]))
